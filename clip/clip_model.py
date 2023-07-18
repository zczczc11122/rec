import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from vision.models import TSNNeXtVLADSE, TSNNetVLADSE, Timesformer, TSMNetVLADSE, TSMRelationModuleBaseTsn, \
    TSNRelationModuleMultiScale, TSNRelationModuleBaseTsn, VideoSwin, TSNAVG
from text.bert_base import TextBERT
from transformers import BertTokenizer

logger = logging.getLogger('template_video_classification')

class CLIP(nn.Module):
    def __init__(self,
                 args,
                 id2_label):
        super().__init__()
        self.args = args
        self.p_drop = args.dropout
        self.bert_fintuing = args.bert_fintuing
        self.clip_type = args.clip_type
        if args.clip_type == 'only_label':
            label_list = []
            #因为用的是中文bert，所以这里用中文表示
            for i in range(len(id2_label)):
                label_list.append(f"视频标签为{id2_label[i]}")
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
            text_idx = self.tokenizer.encode_plus(label_list,
                                                        max_length=10,
                                                        padding='max_length',
                                                        truncation=True,
                                                        return_tensors='pt')
            self.label_input_ids = torch.LongTensor(text_idx['input_ids'])
            self.label_token_type_ids = torch.LongTensor(text_idx['token_type_ids'])
            self.label_attention_mask = torch.LongTensor(text_idx['attention_mask'])

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
            self.vision_model = Timesformer(self.args, False)
        elif self.args.consensus_type == "videoswin":
            assert self.args.num_segments % 8 == 0, 'videoswin 对应 num_segments 为8的倍数'
            self.vision_model = VideoSwin(self.args, False)
        elif self.args.consensus_type == "avg":
            self.vision_model = TSNAVG(self.args)
        else:
            raise ValueError("args.consensus_type is illegal")

        self.text_model = TextBERT(args)
        vision_dim = self.vision_model.get_output_dim()
        text_dim = self.text_model.get_output_dim()
        self.vision_proj = nn.Linear(vision_dim, 1024)
        self.text_proj = nn.Linear(text_dim, 1024)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.summary()

    def forward(self, vision, text_input_ids=None, text_token_type_ids=None, text_attention_mask=None):
        """Encoder, Pool, Predit
            expected shape of 'features': (n_batch, 5, input_dim)
        """
        video_embedding = self.vision_model(vision)
        video_embedding = self.vision_proj(video_embedding)
        video_embedding = F.normalize(video_embedding, dim=-1)
        if self.clip_type == 'only_label':
            text_feature = self.text_model((self.label_input_ids, self.label_attention_mask, self.label_token_type_ids))
        else:
            text_feature = self.text_model((text_input_ids, text_attention_mask, text_token_type_ids))
        text_feature = self.text_proj(text_feature)
        text_feature = F.normalize(text_feature, dim=-1)
        return video_embedding, text_feature, self.logit_scale.exp()

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
        dropout_ratio:          {}
        frame_feature_dim:      {}
            """.format(self.args.arch,
                       self.args.num_segments,
                       self.args.dropout,
                       self.vision_model.feature_dim))


if __name__ == '__main__':
    # model = VideoCNNModel()
    print()


