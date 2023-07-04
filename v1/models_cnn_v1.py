import logging
from torch import nn
import torch
from fusion import ConcatenateDenseSE, BNSEModule, general_weight_initialization, LMF, GMU
from vision.models import TSNNeXtVLADSE, TSNNetVLADSE, Timesformer, TSMNetVLADSE, TSMRelationModuleBaseTsn, \
    TSNRelationModuleMultiScale, TSNRelationModuleBaseTsn, VideoSwin, TSNAVG
import torch.nn.functional as F
from vision import SqueezeContextGating, ContextGating
from text.bert_base import TextBERT
from audio import PANNs
logger = logging.getLogger('template_video_classification')


class VideoCNNModel(nn.Module):
    def __init__(self,
                 args,
                 deploy=False):
        super(VideoCNNModel, self).__init__()
        self.args = args
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

        if self.args.consensus_type == "netvlad":
            self.vision_model = TSNNetVLADSE(self.args)
        elif self.args.consensus_type == "nextvlad":
            self.vision_model = TSNNeXtVLADSE(self.args)
        elif self.args.consensus_type == "trn":
            self.vision_model = TSNRelationModuleMultiScale(self.args)
        elif self.args.consensus_type == "trn_base_tsn":
            self.vision_model = TSNRelationModuleBaseTsn(self.args)
        elif self.args.consensus_type == "tsmnetvlad":
            assert 'resnet' in args.arch.lower(), 'tsm arch目前只支持 resnet'
            self.vision_model = TSMNetVLADSE(self.args)
        elif self.args.consensus_type == "tsm_trn_base_tsn":
            assert 'resnet' in args.arch.lower(), 'tsm arch目前只支持 resnet'
            self.vision_model = TSMRelationModuleBaseTsn(self.args)
        elif self.args.consensus_type == "timesformer":
            assert self.num_segments == 8, 'timesformer 对应 num_segments 必须为8'
            self.vision_model = Timesformer(self.args, deploy)
        elif self.args.consensus_type == "videoswin":
            #videoSwin 论文中fnum_segments用的32，感觉太多了，我们还是用10
            #经分析，最好还是8的倍数，可见笔记分析
            assert self.args.num_segments % 8 == 0, 'videoswin 对应 num_segments 为8的倍数'
            self.vision_model = VideoSwin(self.args, deploy)
        elif self.args.consensus_type == "avg":
            self.vision_model = TSNAVG(self.args)
        else:
            raise ValueError("args.consensus_type is illegal")

        # assert 'resnet' in args.arch.lower(), 'motion的 arch目前只支持 resnet，如果换arch为其他模型，'\
        #                                       '需要把motion的arch与vision的arch写成两个，这先共用'
        # assert self.args.consensus_type != "tsmnetvlad", "有了motion模态，vision最好不用tsmnetvlad，不然这俩重复了"
        # self.motion_model = TSMNetVLADSE(self.args)



        if args.bert_fintuing:
            self.title_model = TextBERT(args)
            self.ocr_model = TextBERT(args)
        else:
            self.title_model = TextBERT(args)
            self.ocr_model = self.title_model

        self.audio_model = PANNs.Transfer_Cnn14()
        audio_model_path = args.audio_pretrian
        if not self.deploy:
            audio_model_checkpoint = torch.load(audio_model_path, map_location=torch.device('cpu'))#, map_location='cpu')
            self.audio_model.load_state_dict(audio_model_checkpoint['model'], strict=False)

        if args.only_vision:
            assert args.fusion_type == 'concat', 'only_vision args.fusion_type only support concat'
            self.intermediate_fc = nn.Sequential(
                # nn.Dropout(p=self.args.dropout),
                nn.Linear(
                    self.vision_model.get_output_dim(),
                    self.fcn_dim),
                nn.ReLU()
            )
        else:
            if args.fusion_type == 'concat':
                self.intermediate_fc = nn.Sequential(
                    # nn.Dropout(p=self.args.dropout),
                    nn.Linear(
                        self.vision_model.get_output_dim()
                        + self.title_model.get_output_dim()
                        + self.ocr_model.get_output_dim()
                        + self.audio_model.audio_embedding_size,
                        # + self.motion_model.get_output_dim(),
                        self.fcn_dim),
                    nn.ReLU()
                )
            elif args.fusion_type == 'lmf':
                self.intermediate_fc = nn.Sequential(
                    LMF([self.vision_model.get_output_dim(),
                         self.title_model.get_output_dim(),
                         self.ocr_model.get_output_dim(),
                         self.audio_model.audio_embedding_size], self.fcn_dim),
                    nn.ReLU()
                )
            elif args.fusion_type == 'gmu':
                self.intermediate_fc = nn.Sequential(
                    GMU([self.vision_model.get_output_dim(),
                         self.title_model.get_output_dim(),
                         self.ocr_model.get_output_dim(),
                         self.audio_model.audio_embedding_size], self.fcn_dim),
                    nn.ReLU()
                )
            else:
                raise ValueError("args.fusion_type is illegal")

        if args.se_gating_type == 'BNSEModule':
            self.se_gating = BNSEModule(self.fcn_dim, self.se_reduction)
        elif args.se_gating_type == 'SqueezeContextGating':
            self.se_gating = SqueezeContextGating(self.fcn_dim, self.se_reduction)
        elif args.se_gating_type == 'ContextGating':
            self.se_gating = ContextGating(self.fcn_dim)
        else:
            raise ValueError("args.se_gating_type is illegal")

        if self.moe:
            self.expert_fc = nn.Sequential(
                nn.BatchNorm1d(self.fcn_dim),
                nn.Dropout(self.args.dropout),
                nn.Linear(self.fcn_dim, self.n_classes * self.num_mixtures)
            )
            if self.per_class:
                self.gating_fc = nn.Sequential(
                    nn.BatchNorm1d(self.fcn_dim),
                    nn.Dropout(self.args.dropout),
                    nn.Linear(self.fcn_dim, self.n_classes * (self.num_mixtures + 1))
                )  # contains one gate for the dummy 'expert' (always predict none)
            else:
                self.gating_fc = nn.Sequential(
                    nn.BatchNorm1d(self.fcn_dim),
                    nn.Dropout(self.args.dropout),
                    nn.Linear(self.fcn_dim, (self.num_mixtures + 1))
                )  # contains one gate for the dummy 'expert' (always predict none)
        else:
            self.classifier = self._get_classifier(self.fcn_dim, num_class=self.n_classes)
        if args.is_distillation:
            self.classifier_vision = self._get_classifier(self.vision_model.get_output_dim(), num_class=self.n_classes)
            self.classifier_title = self._get_classifier(self.title_model.get_output_dim(), num_class=self.n_classes)
            self.classifier_ocr = self._get_classifier(self.ocr_model.get_output_dim(), num_class=self.n_classes)
            self.classifier_audio = self._get_classifier(self.audio_model.audio_embedding_size, num_class=self.n_classes)
            self._init_weights(self.classifier_vision, self.classifier_title, self.classifier_ocr, self.classifier_audio)

        if self.moe:
            if args.fusion_type == 'concat':
                self._init_weights(self.intermediate_fc, self.expert_fc, self.gating_fc)
            else:
                self._init_weights(self.expert_fc, self.gating_fc)
        else:
            if args.fusion_type == 'concat':
                self._init_weights(self.intermediate_fc, self.classifier)
            else:
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

    def forward(self, vision, audio, title_input_ids, title_token_type_ids, title_attention_mask,
                ocr_input_ids, ocr_token_type_ids, ocr_attention_mask):
        """Encoder, Pool, Predict
            expected shape of 'features': (n_batch, 5, input_dim)
        """
        # vision, audio, title_input_ids, title_token_type_ids, title_attention_mask, \
        # ocr_input_ids, ocr_token_type_ids, ocr_attention_mask = features

        video_embedding = self.vision_model(vision)
        if self.only_vision:
            fcn_output = self.intermediate_fc(video_embedding)
        else:
            title_embedding = self.title_model((title_input_ids, title_attention_mask, title_token_type_ids))
            ocr_embedding = self.ocr_model((ocr_input_ids, ocr_attention_mask, ocr_token_type_ids))
            audio_embedding = self.audio_model(audio)
            # motion_embedding = self.motion_model(vision)
            # fcn_output = self.intermediate_fc(
            #     torch.cat([video_embedding, title_embedding, ocr_embedding, audio_embedding, motion_embedding], dim=1))
            if self.args.fusion_type == 'concat':
                fcn_output = self.intermediate_fc(
                    torch.cat([video_embedding, title_embedding, ocr_embedding, audio_embedding], dim=1))
            else:
                fcn_output = self.intermediate_fc([video_embedding, title_embedding, ocr_embedding, audio_embedding])

        fcn_output = self.gating_fc(fcn_output)

        if self.moe:
            # shape (n_batch, n_classes, num_mixtures)
            expert_logits = self.expert_fc(fcn_output).view(
                -1, self.n_classes, self.num_mixtures)
            if self.per_class:
                expert_distributions = F.softmax(
                    self.gating_fc(fcn_output).view(
                        -1, self.n_classes, self.num_mixtures + 1
                    ), dim=-1
                )
            else:
                expert_distributions = F.softmax(
                    self.gating_fc(fcn_output), dim=-1
                ).unsqueeze(1)
            output = (
                    expert_logits * expert_distributions[..., :self.num_mixtures]
            ).sum(dim=-1)
        else:
            output = self.classifier(fcn_output)
        if self.args.is_distillation:
            output_vision = self.classifier_vision(video_embedding)
            output_audio = self.classifier_audio(audio_embedding)
            if self.args.bert_fintuing:
                output_title = self.classifier_title(title_embedding)
                output_ocr = self.classifier_ocr(ocr_embedding)
            else:
                output_title = None
                output_ocr = None
        else:
            output_vision = None
            output_title = None
            output_ocr = None
            output_audio = None

        if self.deploy:
            return output
        else:
            return {'predict': {'video': output,
                                'vision': output_vision,
                                'title': output_title,
                                'ocr': output_ocr,
                                'audio': output_audio}
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
                       self.args.num_class))


if __name__ == '__main__':
    # model = VideoCNNModel()
    print()


