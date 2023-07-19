import logging
from collections import OrderedDict
from torch import nn
import torch
from fusion import ConcatenateDenseSE, BNSEModule, general_weight_initialization, LMF, GMU
from headers.hierarchical import HierarchicalClassifier, HierarchicalClassifierSimple
import torch.nn.functional as F
from vision import SqueezeContextGating, ContextGating
from vision.models import TSN_fix_dim_seq

from text.bert_base import TextBERT_seq
from audio import PANNs
from vilt.vision_transformer import VisionTransformer
from torch.nn import GELU, LayerNorm

logger = logging.getLogger('template_video_classification')

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertOnlyMLMHead(nn.Module):
    def __init__(self, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertLMPredictionHead(nn.Module):
    def __init__(self, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(
            torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class RegionFeatureRegression(nn.Module):
    "for MRM"
    def __init__(self, hidden_size, feat_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12))
        self.out = nn.Linear(hidden_size, feat_dim)

    def forward(self, input_):
        hidden = self.net(input_)
        output = self.out(hidden)
        return output


def copyStateDict(state_dict):
    blocks_new_state_dict = OrderedDict()
    norm_new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'blocks' in k:
            name = ".".join(k.split(".")[1:])
            blocks_new_state_dict[name] = v
        if k.startswith('norm'):
            name = ".".join(k.split(".")[1:])
            norm_new_state_dict[name] = v

    return blocks_new_state_dict, norm_new_state_dict

class Uniter(nn.Module):
    def __init__(self,
                 args,
                 label_dict,
                 deploy=False):
        super(Uniter, self).__init__()
        self.args = args
        self.seq_dim = 768

        self.fcn_dim = args.fusion_embedding_size
        self.num_mixtures = 2
        self.se_reduction = 4
        self.p_drop = args.dropout
        self.per_class = True
        self.only_vision = args.only_vision
        self.deploy = deploy

        self.vision_model = TSN_fix_dim_seq(self.args, self.seq_dim)
        self.title_model = TextBERT_seq(args)
        self.ocr_model = self.title_model

        self.audio_model = PANNs.Transfer_Cnn14()
        audio_model_path = args.audio_pretrian
        if not self.deploy:
            audio_model_checkpoint = torch.load(audio_model_path, map_location=torch.device('cpu'))#, map_location='cpu')
            self.audio_model.load_state_dict(audio_model_checkpoint['model'], strict=False)

        self.audio_fc = torch.nn.Linear(in_features=self.audio_model.audio_embedding_size, out_features=self.seq_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.seq_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.token_type_embeddings = nn.Embedding(4, self.seq_dim)
        self.token_type_embeddings.apply(self.init_weights)

        self.transformer = VisionTransformer()

        fusion_checkpoint = torch.load(args.fusion_pretrain, map_location=torch.device('cpu'))  # , map_location='cpu')
        blocks_new_state_dict, norm_new_state_dict = copyStateDict(fusion_checkpoint)
        self.transformer.blocks.load_state_dict(blocks_new_state_dict, strict=True)
        self.transformer.norm.load_state_dict(norm_new_state_dict, strict=True)

        self.mlm_cls = BertOnlyMLMHead(self.title_model.bert.embeddings.word_embeddings.weight)
        self.feat_regress = RegionFeatureRegression(self.seq_dim, self.seq_dim)

        self.pooler = Pooler(self.seq_dim)
        self.itm_output = nn.Linear(self.seq_dim, 2)

        if args.se_gating_type == 'BNSEModule':
            self.se_gating = BNSEModule(self.seq_dim, self.se_reduction)
        elif args.se_gating_type == 'SqueezeContextGating':
            self.se_gating = SqueezeContextGating(self.seq_dim, self.se_reduction)
        elif args.se_gating_type == 'ContextGating':
            self.se_gating = ContextGating(self.seq_dim)
        else:
            raise ValueError("args.se_gating_type is illegal")

        if args.classifier_type == 'hierarchicalClassifier':
            self.classifier_expression = HierarchicalClassifier(self.seq_dim, 256,
                                                                label_dict['expression'], self.args.dropout)
            self.classifier_material = HierarchicalClassifier(self.seq_dim, 256,
                                                              label_dict['material'], self.args.dropout)
            self.classifier_person = HierarchicalClassifier(self.seq_dim, 256,
                                                            label_dict['person'], self.args.dropout)
            self.classifier_style = HierarchicalClassifier(self.seq_dim, 256,
                                                           label_dict['style'], self.args.dropout)
            self.classifier_topic = HierarchicalClassifier(self.seq_dim, 256,
                                                           label_dict['topic'], self.args.dropout)
        elif args.classifier_type == 'hierarchicalClassifier_simple':
            self.classifier_expression = HierarchicalClassifierSimple(self.seq_dim, 256,
                                                                      label_dict['expression'], self.args.dropout)
            self.classifier_material = HierarchicalClassifierSimple(self.seq_dim, 256,
                                                                    label_dict['material'], self.args.dropout)
            self.classifier_person = HierarchicalClassifierSimple(self.seq_dim, 256,
                                                                  label_dict['person'], self.args.dropout)
            self.classifier_style = HierarchicalClassifierSimple(self.seq_dim, 256,
                                                                 label_dict['style'], self.args.dropout)
            self.classifier_topic = HierarchicalClassifierSimple(self.seq_dim, 256,
                                                                 label_dict['topic'], self.args.dropout)
        else:
            raise ValueError("args.classifier_type is illegal")

        self._init_weights(self.classifier_expression,
                           self.classifier_material,
                           self.classifier_person,
                           self.classifier_style,
                           self.classifier_topic)
        self.summary()

    def _init_weights(self, *components):
        for component in components:
            for module in component.modules():
                general_weight_initialization(module)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_embedding(self, vision, audio, title_input_ids, title_token_type_ids, title_attention_mask,
                ocr_input_ids, ocr_token_type_ids, ocr_attention_mask, return_vision_embedding=False):
        """Encoder, Pool, Predit
            expected shape of 'features': (n_batch, 5, input_dim)
        """
        # vision = batch['images']
        # audio = batch['audio']
        #
        # title_input_ids_mask = batch['title_input_ids_mask']
        # title_attention_mask_mask = batch['title_attention_mask_mask']
        # title_token_type_ids_mask = batch['title_token_type_ids_mask']
        # title_txt_labels_mask = batch['title_txt_labels_mask']
        #
        # ocr_input_ids_mask = batch['ocr_input_ids_mask']
        # ocr_attention_mask_mask = batch['ocr_attention_mask_mask']
        # ocr_token_type_ids_mask = batch['ocr_token_type_ids_mask']
        # ocr_txt_labels_mask = batch['ocr_txt_labels_mask']

        video_embedding = self.vision_model(vision)
        vision_embedding = video_embedding

        title_embedding = self.title_model((title_input_ids, title_attention_mask, title_token_type_ids))
        ocr_embedding = self.ocr_model((ocr_input_ids, ocr_attention_mask, ocr_token_type_ids))

        audio_embedding = self.audio_model(audio)
        audio_embedding = self.audio_fc(audio_embedding)[:, None, :]

        video_mask = torch.ones(video_embedding.shape[:-1]).to(video_embedding)
        audio_mask = torch.ones(audio_embedding.shape[:-1]).to(audio_embedding)

        video_embedding = video_embedding + self.token_type_embeddings(
            torch.full_like(video_mask, 0).type(torch.cuda.LongTensor))
        title_embedding = title_embedding + self.token_type_embeddings(
            torch.full_like(title_attention_mask, 1).type(torch.cuda.LongTensor))
        ocr_embedding = ocr_embedding + self.token_type_embeddings(
            torch.full_like(ocr_attention_mask, 2).type(torch.cuda.LongTensor))
        audio_embedding = audio_embedding + self.token_type_embeddings(
            torch.full_like(audio_mask, 3).type(torch.cuda.LongTensor))

        cls_tokens = self.cls_token.expand(video_embedding.shape[0], -1, -1)

        co_embeds = torch.cat([cls_tokens, video_embedding, title_embedding, ocr_embedding, audio_embedding], dim=1)
        co_masks = torch.cat([torch.ones(cls_tokens.shape[0], 1).to(cls_tokens), video_mask, title_attention_mask, ocr_attention_mask, audio_mask], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        if return_vision_embedding:
            return x, vision_embedding
        else:
            return x

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def _get_feat_target(self, img_feat, img_masks):
        img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)  # (n, m, d)
        feat_dim = img_feat.size(-1)
        feat_targets = img_feat[img_masks_ext].contiguous().view(
            -1, feat_dim)  # (s, d)
        return feat_targets

    def forward_mlm(self, vision, audio, title_input_ids, title_token_type_ids, title_attention_mask, title_txt_labels_mask,
                ocr_input_ids, ocr_token_type_ids, ocr_attention_mask, ocr_txt_labels_mask):

        num_segments = vision.size()[1] // 3
        title_sequence_len = title_input_ids.size(1)
        ocr_sequence_len = ocr_input_ids.size(1)

        sequence_output = self.forward_embedding(vision, audio, title_input_ids, title_token_type_ids, title_attention_mask,
                                ocr_input_ids, ocr_token_type_ids, ocr_attention_mask)

        text_sequence_output = sequence_output[:, (1 + num_segments):
                                                  (1 + num_segments + title_sequence_len + ocr_sequence_len), :]
        text_labels = torch.cat([title_txt_labels_mask, ocr_txt_labels_mask], 1)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(text_sequence_output,
                                                    text_labels != -1)

        prediction_scores = self.mlm_cls(masked_output)
        masked_lm_loss = F.cross_entropy(prediction_scores,
                                         text_labels[text_labels != -1],
                                         reduction='mean')
        return masked_lm_loss

    def forward_itm(self, vision, audio, title_input_ids, title_token_type_ids, title_attention_mask,
                    ocr_input_ids, ocr_token_type_ids, ocr_attention_mask, itm_target):

        sequence_output = self.forward_embedding(vision, audio, title_input_ids, title_token_type_ids,
                                                 title_attention_mask,
                                                 ocr_input_ids, ocr_token_type_ids, ocr_attention_mask)

        pooled_output = self.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)
        itm_loss = F.cross_entropy(itm_scores, itm_target, reduction='mean')
        return itm_loss

    def forward_mrfr(self, vision, audio, title_input_ids, title_token_type_ids, title_attention_mask,
                ocr_input_ids, ocr_token_type_ids, ocr_attention_mask, mrfr_img_mask):
        num_segments = vision.size()[1] // 3
        sequence_output, vision_embedding = self.forward_embedding(vision, audio, title_input_ids, title_token_type_ids,
                                                                   title_attention_mask,
                                                                   ocr_input_ids, ocr_token_type_ids, ocr_attention_mask,
                                                                   return_vision_embedding=True)
        masked_output = self._compute_masked_hidden(sequence_output[:, 1: num_segments + 1, :],
                                                    mrfr_img_mask)

        prediction_feat = self.feat_regress(masked_output)
        feat_targets = self._get_feat_target(vision_embedding, mrfr_img_mask)
        mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                               reduction='mean')
        return mrfr_loss

    def forward_cls(self, vision, audio, title_input_ids, title_token_type_ids, title_attention_mask,
                    ocr_input_ids, ocr_token_type_ids, ocr_attention_mask):
        sequence_output = self.forward_embedding(vision, audio,
                                                 title_input_ids, title_token_type_ids, title_attention_mask,
                                                 ocr_input_ids, ocr_token_type_ids, ocr_attention_mask)

        cls_feats = self.pooler(sequence_output)
        fcn_output = self.se_gating(cls_feats)

        fusion_output_expression = self.classifier_expression(fcn_output)
        fusion_output_material = self.classifier_material(fcn_output)
        fusion_output_person = self.classifier_person(fcn_output)
        fusion_output_style = self.classifier_style(fcn_output)
        fusion_output_topic = self.classifier_topic(fcn_output)

        return fusion_output_expression, \
               fusion_output_material, \
               fusion_output_person, \
               fusion_output_style, \
               fusion_output_topic



    def forward(self, batch, task):
        """Encoder, Pool, Predit
            expected shape of 'features': (n_batch, 5, input_dim)
        """
        images = batch['images']
        audio = batch['audio']

        title_input_ids = batch['title_input_ids']
        title_input_ids_mask = batch['title_input_ids_mask']
        title_txt_labels_mask = batch['title_txt_labels_mask']
        title_token_type_ids_mask = batch['title_token_type_ids_mask']
        title_attention_mask_mask = batch['title_attention_mask_mask']

        ocr_input_ids = batch['ocr_input_ids']
        ocr_input_ids_mask = batch['ocr_input_ids_mask']
        ocr_txt_labels_mask = batch['ocr_txt_labels_mask']
        ocr_token_type_ids_mask = batch['ocr_token_type_ids_mask']
        ocr_attention_mask_mask = batch['ocr_attention_mask_mask']

        random_images = batch['random_images']
        itm_target = batch['itm_target']

        mrfr_img_mask = batch['mrfr_img_mask']

        loss = 0.
        for i in task:
            if i == "mlm":
                mlm_loss = self.forward_mlm(images, audio,
                                            title_input_ids_mask,
                                            title_token_type_ids_mask,
                                            title_attention_mask_mask,
                                            title_txt_labels_mask,
                                            ocr_input_ids_mask,
                                            ocr_token_type_ids_mask,
                                            ocr_attention_mask_mask,
                                            ocr_txt_labels_mask)
                loss += mlm_loss
            if i == "itm":
                itm_loss = self.forward_itm(random_images, audio,
                                            title_input_ids,
                                            title_token_type_ids_mask,
                                            title_attention_mask_mask,
                                            ocr_input_ids,
                                            ocr_token_type_ids_mask,
                                            ocr_attention_mask_mask,
                                            itm_target
                                            )
                loss += itm_loss
            if i == "mrfr":
                mrfr_loss = self.forward_mrfr(images, audio,
                                              title_input_ids,
                                              title_token_type_ids_mask,
                                              title_attention_mask_mask,
                                              ocr_input_ids,
                                              ocr_token_type_ids_mask,
                                              ocr_attention_mask_mask,
                                              mrfr_img_mask)
                loss += mrfr_loss
            if i == "cls":
                fusion_output_expression, \
                fusion_output_material, \
                fusion_output_person, \
                fusion_output_style, \
                fusion_output_topic = self.forward_cls(images, audio,
                                                       title_input_ids,
                                                       title_token_type_ids_mask,
                                                       title_attention_mask_mask,
                                                       ocr_input_ids,
                                                       ocr_token_type_ids_mask,
                                                       ocr_attention_mask_mask
                                                       )
                if self.deploy:
                    return tuple(fusion_output_expression), \
                           tuple(fusion_output_material), \
                           tuple(fusion_output_person), \
                           tuple(fusion_output_style), \
                           tuple(fusion_output_topic)
                else:
                    return {'predict': {'video': (fusion_output_expression,
                                                  fusion_output_material,
                                                  fusion_output_person,
                                                  fusion_output_style,
                                                  fusion_output_topic)}
                            }

        return loss

    @property
    def input_size(self):
        return self.vision_model.input_size

    @property
    def input_mean(self):
        return self.vision_model.input_mean

    @property
    def input_std(self):
        return self.vision_model.input_std

    @property
    def crop_size(self):
        return self.vision_model.input_size

    @property
    def scale_size(self):
        return int(self.vision_model.input_size / 0.875)

    def summary(self):
        logger.info("""
    Initializing Template Video Classify Framework
        frame base model:       {}
        num_segments:           {}
        consensus_module:       {}
        dropout_ratio:          {}
        frame_feature_dim:      {}
            """.format(self.args.arch,
                       self.args.num_segments,
                       self.args.consensus_type,
                       self.args.dropout,
                       self.vision_model.feature_dim))


if __name__ == '__main__':
    # model = VideoCNNModel()
    print()


