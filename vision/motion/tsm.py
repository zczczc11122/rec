import torchvision
import numpy as np
from torch import nn
from vision.motion.temporal_shift import make_temporal_shift
from vision.motion.non_local import make_non_local
from vision.motion.basic_ops import ConsensusModule

class TSM(nn.Module):
    def __init__(self, num_segments, base_model='resnet18',
                 is_shift=False, shift_div=8, shift_place='blockres', non_local=False, consensus_type='avg',
                 temporal_pool=False):
        super(TSM, self).__init__()
        self.is_shift = is_shift
        self.non_local = non_local
        self.num_segments = num_segments
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool

        input_size = 224
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]

        self._prepare_base_model(base_model)

        # self.consensus = ConsensusModule(consensus_type)

    def _prepare_base_model(self, base_model_name):
        # print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model_name:
            self.base_model = getattr(torchvision.models, base_model_name)(True)
            if self.is_shift:
                # print('Adding temporal shift...')
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            if self.non_local:
                # print('Adding non_local module...')
                make_non_local(self.base_model, self.num_segments, base_model_name)

            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError('Unknown base model: {}'.format(base_model_name))

    def forward(self, x):
        # batch_size * num_segments, channel, height, weight
        frame_emb = self.base_model(x.view(-1, 3, x.size()[-2], x.size()[-1]))
        base_out = frame_emb.view((-1, self.num_segments) + frame_emb.size()[1:])
        # output = self.consensus(base_out)
        return base_out #output.squeeze(1)

    def get_output_dim(self):
        return self.feature_dim




