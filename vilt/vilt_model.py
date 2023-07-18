import logging
from functools import partial
from collections import OrderedDict
from torch import nn
import torch
from fusion import ConcatenateDenseSE, BNSEModule, general_weight_initialization, LMF, GMU
from vision.models import TSN_fix_dim_seq
import torch.nn.functional as F
from vision import SqueezeContextGating, ContextGating
from text.bert_base import TextBERT_seq
from audio import PANNs
from vilt.vision_transformer import VisionTransformer


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


class Vilt(nn.Module):
    def __init__(self,
                 args,
                 deploy=False):
        super(Vilt, self).__init__()
        self.args = args
        self.seq_dim = 768

        self.fcn_dim = args.fusion_embedding_size
        self.num_mixtures = 2
        self.se_reduction = 4
        self.p_drop = args.dropout
        self.per_class = True
        self.moe = args.use_moe
        self.only_vision = args.only_vision
        self.deploy = deploy

        if args.dim == "person":
            self.n_classes = args.num_class_person
        elif args.dim == "expression":
            self.n_classes = args.num_class_expression
        elif args.dim == "style":
            self.n_classes = args.num_class_style
        elif args.dim == "topic":
            self.n_classes = args.num_class_topic

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


        self.pooler = Pooler(self.seq_dim)

        if args.se_gating_type == 'BNSEModule':
            self.se_gating = BNSEModule(self.seq_dim, self.se_reduction)
        elif args.se_gating_type == 'SqueezeContextGating':
            self.se_gating = SqueezeContextGating(self.seq_dim, self.se_reduction)
        elif args.se_gating_type == 'ContextGating':
            self.se_gating = ContextGating(self.seq_dim)
        else:
            raise ValueError("args.se_gating_type is illegal")

        self.classifier = self._get_classifier(self.seq_dim, num_class=self.n_classes)
        self._init_weights(self.classifier)

        self.summary()

    def _get_classifier(self, consensus_output_size, num_class=-1, reduce_dim=256):
        if num_class == -1:
            num_class = self.n_classes
        if self.args.use_MLP:
            classifier = nn.Sequential(
                nn.Linear(consensus_output_size, reduce_dim),
                nn.Dropout(p=self.args.dropout),
                nn.ReLU(),
                nn.Linear(reduce_dim, num_class)
            )
        else:
            classifier = nn.Sequential(
                nn.Dropout(p=self.args.dropout),
                nn.Linear(consensus_output_size, num_class)
            )

        # for m in classifier.children():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.constant_(m.bias, 0)
        return classifier

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



    def forward(self, vision, audio, title_input_ids, title_token_type_ids, title_attention_mask,
                ocr_input_ids, ocr_token_type_ids, ocr_attention_mask):
        """Encoder, Pool, Predit
            expected shape of 'features': (n_batch, 5, input_dim)
        """
        video_embedding = self.vision_model(vision)
        title_embedding = self.title_model((title_input_ids, title_attention_mask, title_token_type_ids))
        ocr_embedding = self.ocr_model((ocr_input_ids, ocr_attention_mask, ocr_token_type_ids))
        audio_embedding = self.audio_model(audio)
        audio_embedding = self.audio_fc(audio_embedding)[:, None, :]

        video_mask = torch.ones(video_embedding.shape[:-1]).to(video_embedding)
        audio_mask = torch.ones(audio_embedding.shape[:-1]).to(audio_embedding)

        video_embedding = video_embedding + self.token_type_embeddings(torch.full_like(video_mask, 0).type(torch.cuda.LongTensor))
        title_embedding = title_embedding + self.token_type_embeddings(torch.full_like(title_attention_mask, 1).type(torch.cuda.LongTensor))
        ocr_embedding = ocr_embedding + self.token_type_embeddings(torch.full_like(ocr_attention_mask, 2).type(torch.cuda.LongTensor))
        audio_embedding = audio_embedding + self.token_type_embeddings(torch.full_like(audio_mask, 3).type(torch.cuda.LongTensor))

        cls_tokens = self.cls_token.expand(video_embedding.shape[0], -1, -1)

        co_embeds = torch.cat([cls_tokens, video_embedding, title_embedding, ocr_embedding, audio_embedding], dim=1)
        co_masks = torch.cat([torch.ones(cls_tokens.shape[0], 1).to(cls_tokens), video_mask, title_attention_mask, ocr_attention_mask, audio_mask], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)

        cls_feats = self.pooler(x)
        fcn_output = self.se_gating(cls_feats)
        output = self.classifier(fcn_output)

        if self.deploy:
            return output
        else:
            return {'predict': {'video': output}
                    }
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
        num_class:            {}
            """.format(self.args.arch,
                       self.args.num_segments,
                       self.args.consensus_type,
                       self.args.dropout,
                       self.vision_model.feature_dim,
                       self.n_classes))


if __name__ == '__main__':
    # model = VideoCNNModel()
    print()


