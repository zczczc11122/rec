from typing import Any
import math
from collections import OrderedDict
from vision.backbone import Backbone
from vision.consensus import NetVLADConsensus, NetXtVLADConsensus, RelationModuleMultiScale, RelationModuleBaseTsn, AverageConsensus
from vision import SqueezeContextGating
import torch.nn as nn
import torch
from vision.timesformer.models.vit import vit_base_patch16_224
from vision.motion.tsm import TSM
from vision.video_swintransformer.swin_transformer import SwinTransformer3D

class TSN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        base_model, feature_dim, input_size, input_mean, input_std = Backbone().prepare_backbone(args.arch)
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.input_mean = input_mean
        self.input_std = input_std

    def forward(self, x): # batch_size * (num_segments * channel) * height * weight
        num_segments = x.size()[1] // 3
        # batch_size * num_segments, channel, height, weight
        frame_emb = self.base_model(x.view(-1, 3, x.size()[-2], x.size()[-1]))
        if self.args.arch.startswith('efficientnet'):
            frame_emb = frame_emb.squeeze(-1).squeeze(-1)
        # batch_size * num_segments, feature_size
        frame_emb = frame_emb.view((-1, num_segments) + frame_emb.size()[1:])
        return frame_emb

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

class TSN_fix_dim_seq(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.args = args
        self.backbone = TSN(args)
        self.fc = torch.nn.Linear(in_features=self.backbone.feature_dim, out_features=dim)

        self.feature_dim = dim
        self.input_size = self.backbone.input_size
        self.input_mean = self.backbone.input_mean
        self.input_std = self.backbone.input_std

    def forward(self, x):
        frame_emb = self.backbone(x)
        frame_emb = self.fc(frame_emb)

        return frame_emb

    def _forward_unimplemented(self, *input: Any) -> None:
        pass



class TSNAVG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.backbone = TSN(args)
        self.base_model = self.backbone.base_model
        self.feature_dim = self.backbone.feature_dim
        self.input_size = self.backbone.input_size
        self.input_mean = self.backbone.input_mean
        self.input_std = self.backbone.input_std

        self.AVGConsensus = AverageConsensus(self.feature_dim)

        self.vision_embedding_size = self.AVGConsensus.get_output_dim()
        if self.args.vlad_add_se:
            self.se = SqueezeContextGating(self.vision_embedding_size)

    def forward(self, x): # batch_size * (num_segments * channel) * height * weight
        frame_emb = self.backbone(x)
        video_emb = self.AVGConsensus(frame_emb)
        if self.args.vlad_add_se:
            video_emb = self.se(video_emb)
        return video_emb

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def get_output_dim(self):
        return self.vision_embedding_size


class TSNNetVLADSE(nn.Module):
    def __init__(self, args):
        super().__init__()
        # if not args.only_vision:
        #     if not args.vlad_add_final_fc:
        #         raise ValueError("if not args.only_vision, args.vlad_add_final_fc must be True, or vision feature dim is too long")
        self.args = args

        self.backbone = TSN(args)
        self.base_model = self.backbone.base_model
        self.feature_dim = self.backbone.feature_dim
        self.input_size = self.backbone.input_size
        self.input_mean = self.backbone.input_mean
        self.input_std = self.backbone.input_std

        self.netvlad = NetVLADConsensus(feature_size=self.feature_dim,
                                        num_segments=args.num_segments,
                                        num_clusters=args.vlad_cluster_size,
                                        output_feature_size=args.vlad_hidden_size,
                                        add_final_fc=args.vlad_add_final_fc)
        self.vlad_hidden_size = self.netvlad.get_output_dim()
        if self.args.vlad_add_se:
            self.se = SqueezeContextGating(self.vlad_hidden_size)
        self.vision_embedding_size = self.vlad_hidden_size

    def forward(self, x):  # batch_size * (num_segments * channel) * height * weight
        frame_emb = self.backbone(x)
        video_emb = self.netvlad(frame_emb)
        if self.args.vlad_add_se:
            video_emb = self.se(video_emb)
        return video_emb

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def get_output_dim(self):
        return self.vision_embedding_size

class TSNNeXtVLADSE(nn.Module):
    def __init__(self, args):
        super().__init__()
        # if not args.only_vision:
        #     if not args.vlad_add_final_fc:
        #         raise ValueError("if not args.only_vision, args.vlad_add_final_fc must be True, or vision feature dim is too long")
        self.args = args

        self.backbone = TSN(args)
        self.base_model = self.backbone.base_model
        self.feature_dim = self.backbone.feature_dim
        self.input_size = self.backbone.input_size
        self.input_mean = self.backbone.input_mean
        self.input_std = self.backbone.input_std

        self.netvlad = NetXtVLADConsensus(feature_size=self.feature_dim,
                                          num_segments=args.num_segments,
                                          num_clusters=args.vlad_cluster_size,
                                          output_feature_size=args.vlad_hidden_size,
                                          add_final_fc=args.vlad_add_final_fc)
        self.vlad_hidden_size = self.netvlad.get_output_dim()
        if self.args.vlad_add_se:
            self.se = SqueezeContextGating(self.vlad_hidden_size)
        self.vision_embedding_size = self.vlad_hidden_size

    def forward(self, x):  # batch_size * (num_segments * channel) * height * weight
        frame_emb = self.backbone(x)
        video_emb = self.netvlad(frame_emb)
        if self.args.vlad_add_se:
            video_emb = self.se(video_emb)
        return video_emb

    '''
    def _forward_unimplemented(self, *input: Any) -> None:
        pass
    '''
    def get_output_dim(self):
        return self.vision_embedding_size

class TSNRelationModuleMultiScale(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.backbone = TSN(args)
        self.base_model = self.backbone.base_model
        self.feature_dim = self.backbone.feature_dim
        self.input_size = self.backbone.input_size
        self.input_mean = self.backbone.input_mean
        self.input_std = self.backbone.input_std

        self.trn = RelationModuleMultiScale(img_feature_dim=self.feature_dim,
                                                num_frames=args.num_segments,
                                                output_feature_size=args.trn_hidden_size)
        self.vision_embedding_size = self.trn.get_output_dim()

    def forward(self, x):  # batch_size * (num_segments * channel) * height * weight
        frame_emb = self.backbone(x)
        video_emb = self.trn(frame_emb)

        return video_emb

    def get_output_dim(self):
        return self.vision_embedding_size

class TSNRelationModuleBaseTsn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.backbone = TSN(args)
        self.base_model = self.backbone.base_model
        self.feature_dim = self.backbone.feature_dim
        self.input_size = self.backbone.input_size
        self.input_mean = self.backbone.input_mean
        self.input_std = self.backbone.input_std

        self.trn = RelationModuleBaseTsn(feature_dim=self.feature_dim,
                                         output_feature_dim=args.trn_hidden_size,
                                         num_segments=args.num_segments)
        self.vision_embedding_size = self.trn.get_output_dim()

    def forward(self, x):  # batch_size * (num_segments * channel) * height * weight
        frame_emb = self.backbone(x)
        video_emb = self.trn(frame_emb)

        return video_emb

    def get_output_dim(self):
        return self.vision_embedding_size


class TSMNetVLADSE(nn.Module):
    def __init__(self, args):
        super().__init__()
        # if not args.only_vision:
        #     if not args.vlad_add_final_fc:
        #         raise ValueError("if not args.only_vision, args.vlad_add_final_fc must be True, or vision feature dim is too long")
        self.args = args
        self.backbone = TSM(base_model=args.arch,
                            num_segments=args.num_segments,
                            is_shift=args.tsm_is_shift,
                            non_local=args.tsm_non_local)

        self.feature_dim = self.backbone.feature_dim
        self.input_size = self.backbone.input_size
        self.input_mean = self.backbone.input_mean
        self.input_std = self.backbone.input_std

        self.netvlad = NetVLADConsensus(feature_size=self.feature_dim,
                                        num_segments=args.num_segments,
                                        num_clusters=args.vlad_cluster_size,
                                        output_feature_size=args.vlad_hidden_size,
                                        add_final_fc=args.vlad_add_final_fc)
        self.vlad_hidden_size = self.netvlad.get_output_dim()
        if self.args.vlad_add_se:
            self.se = SqueezeContextGating(self.vlad_hidden_size)
        self.vision_embedding_size = self.vlad_hidden_size

    def forward(self, x):  # batch_size * (num_segments * channel) * height * weight
        frame_emb = self.backbone(x)
        video_emb = self.netvlad(frame_emb)
        if self.args.vlad_add_se:
            video_emb = self.se(video_emb)
        return video_emb

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def get_output_dim(self):
        return self.vision_embedding_size

class TSMRelationModuleBaseTsn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = TSM(base_model=args.arch,
                            num_segments=args.num_segments,
                            is_shift=args.tsm_is_shift,
                            non_local=args.tsm_non_local)

        self.feature_dim = self.backbone.feature_dim
        self.input_size = self.backbone.input_size
        self.input_mean = self.backbone.input_mean
        self.input_std = self.backbone.input_std

        self.trn = RelationModuleBaseTsn(feature_dim=self.feature_dim,
                                         output_feature_dim=args.trn_hidden_size,
                                         num_segments=args.num_segments)
        self.vision_embedding_size = self.trn.get_output_dim()

    def forward(self, x):  # batch_size * (num_segments * channel) * height * weight
        frame_emb = self.backbone(x)
        video_emb = self.trn(frame_emb)
        return video_emb

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def get_output_dim(self):
        return self.vision_embedding_size


class Timesformer(nn.Module):
    def __init__(self, args, deploy):
        super(Timesformer, self).__init__()
        self.backbone = vit_base_patch16_224()
        self.feature_dim = self.backbone.feature_dim
        self.input_size = self.backbone.input_size
        self.input_mean = self.backbone.input_mean
        self.input_std = self.backbone.input_std

        if not (args.resume or deploy):
            model_checkpoint = torch.load(args.timsf_pretrian, map_location=torch.device('cpu'))
            self.backbone.load_state_dict(model_checkpoint['state_dict'], strict=True)
        self.backbone.model.head = nn.Identity()

    def forward(self, x):
        video_emb = self.backbone(x)
        return video_emb

    def get_output_dim(self):
        return self.feature_dim

class VideoSwin(nn.Module):
    def __init__(self, args, deploy):
        super(VideoSwin, self).__init__()
        self.backbone = SwinTransformer3D(patch_size=(2,4,4),
                                          embed_dim=128,
                                          depths=[2, 2, 18, 2],
                                          num_heads=[4, 8, 16, 32],
                                          window_size=(8, 7, 7),
                                          mlp_ratio=4.,
                                          qkv_bias=True,
                                          drop_rate=0.,
                                          attn_drop_rate=0.,
                                          drop_path_rate=0.2,
                                          patch_norm=True)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.feature_dim = 1024
        self.input_size = 224
        self.input_mean = [i / 255. for i in [123.675, 116.28, 103.53]]
        self.input_std = [i / 255. for i in [58.395, 57.12, 57.375]]

        def copyStateDict(state_dict):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('backbone'):
                    new_k = '.'.join(k.split('.')[1:])
                    new_state_dict[new_k] = v
            return new_state_dict

        if not (args.resume or deploy):
            model_checkpoint = torch.load(args.video_swin_pretrian, map_location=torch.device('cpu'))
            self.backbone.load_state_dict(copyStateDict(model_checkpoint['model_state']), strict=True)

    def forward(self, x):
        x = self.backbone(x)
        video_emb = self.avg_pool(x)
        video_emb = video_emb.view(video_emb.shape[0], -1)
        return video_emb

    def get_output_dim(self):
        return self.feature_dim


class PositionEmbedding(nn.Module):

    '''
    Position embedding for self-attention
    refer: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    d_model: word embedding size or output size of the self-attention blocks
    max_len: the max length of the input squeezec
    '''

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)    # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # [1, max_len]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)   # not the parameters of the Module

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TSNTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        base_model, feature_dim, input_size, input_mean, input_std = Backbone().prepare_backbone(args.arch)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.POS = PositionEmbedding(d_model=feature_dim, max_len=16)
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.input_mean = input_mean
        self.input_std = input_std

    def forward(self, x): # batch_size * (num_segments * channel) * height * weight
        num_segments = x.size()[1] // 3
        # batch_size * num_segments, channel, height, weight
        frame_emb = self.base_model(x.view(-1, 3, x.size()[-2], x.size()[-1]))
        if self.args.arch.startswith('efficientnet'):
            frame_emb = frame_emb.squeeze(-1).squeeze(-1)
        # batch_size * num_segments, feature_size
        frame_emb = frame_emb.view((-1, num_segments) + frame_emb.size()[1:])
        # batch_size, num_segments, feature_size
        frame_emb = self.POS(frame_emb)
        video_emb = self.transformer_encoder(frame_emb)
        # batch_size, [1,] consensus_feature_size
        video_emb = video_emb[:,0,:]
        return video_emb
