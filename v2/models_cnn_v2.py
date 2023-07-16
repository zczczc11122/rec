import logging
from torch import nn
import torch
from fusion import ConcatenateDenseSE, BNSEModule, general_weight_initialization, LMF, GMU
from vision.models import TSNNeXtVLADSE, TSNNetVLADSE, Timesformer, TSMNetVLADSE, TSMRelationModuleBaseTsn, \
    TSNRelationModuleMultiScale, TSNRelationModuleBaseTsn, VideoSwin, TSNAVG
from headers.hierarchical import HierarchicalClassifier, HierarchicalClassifierSimple
import torch.nn.functional as F
from vision import SqueezeContextGating, ContextGating
from text.bert_base import TextBERT
from audio import PANNs
logger = logging.getLogger('template_video_classification')


class VideoCNNModel(nn.Module):
    def __init__(self,
                 args,
                 label_dict,
                 deploy=False):
        super(VideoCNNModel, self).__init__()
        self.args = args
        self.fcn_dim = args.fusion_embedding_size
        self.num_mixtures = 2
        self.se_reduction = 4
        self.p_drop = args.dropout
        self.per_class = True
        self.only_vision = args.only_vision
        self.deploy = deploy

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
            assert self.args.num_segments == 8, 'timesformer 对应 num_segments 必须为8'
            self.vision_model = Timesformer(self.args, deploy)
        elif self.args.consensus_type == "videoswin":
            assert self.args.num_segments % 8 == 0, 'videoswin 对应 num_segments 为8的倍数'
            self.vision_model = VideoSwin(self.args, deploy)
        elif self.args.consensus_type == "avg":
            self.vision_model = TSNAVG(self.args)
        else:
            raise ValueError("args.consensus_type is illegal")

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

        if args.classifier_type == 'hierarchicalClassifier':
            self.classifier_expression = HierarchicalClassifier(self.fcn_dim, 256,
                                                                label_dict['expression'], self.args.dropout)
            self.classifier_material = HierarchicalClassifier(self.fcn_dim, 256,
                                                                label_dict['material'], self.args.dropout)
            self.classifier_person = HierarchicalClassifier(self.fcn_dim, 256,
                                                                label_dict['person'], self.args.dropout)
            self.classifier_style = HierarchicalClassifier(self.fcn_dim, 256,
                                                                label_dict['style'], self.args.dropout)
            self.classifier_topic = HierarchicalClassifier(self.fcn_dim, 256,
                                                                label_dict['topic'], self.args.dropout)
        elif args.classifier_type == 'hierarchicalClassifier_simple':
            self.classifier_expression = HierarchicalClassifierSimple(self.fcn_dim, 256,
                                                                label_dict['expression'], self.args.dropout)
            self.classifier_material = HierarchicalClassifierSimple(self.fcn_dim, 256,
                                                              label_dict['material'], self.args.dropout)
            self.classifier_person = HierarchicalClassifierSimple(self.fcn_dim, 256,
                                                            label_dict['person'], self.args.dropout)
            self.classifier_style = HierarchicalClassifierSimple(self.fcn_dim, 256,
                                                           label_dict['style'], self.args.dropout)
            self.classifier_topic = HierarchicalClassifierSimple(self.fcn_dim, 256,
                                                           label_dict['topic'], self.args.dropout)
        else:
            raise ValueError("args.classifier_type is illegal")
        if args.fusion_type == 'concat':
            self._init_weights(self.intermediate_fc,
                               self.classifier_expression,
                               self.classifier_material,
                               self.classifier_person,
                               self.classifier_style,
                               self.classifier_topic)
        else:
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

    def forward(self, vision, audio, title_input_ids, title_token_type_ids, title_attention_mask,
                ocr_input_ids, ocr_token_type_ids, ocr_attention_mask):
        """Encoder, Pool, Predit
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

        fcn_output = self.se_gating(fcn_output)

        fusion_output_expression = self.classifier_expression(fcn_output)
        fusion_output_material = self.classifier_material(fcn_output)
        fusion_output_person = self.classifier_person(fcn_output)
        fusion_output_style = self.classifier_style(fcn_output)
        fusion_output_topic = self.classifier_topic(fcn_output)

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


