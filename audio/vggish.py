import torch.nn as nn
import logging
from torch.utils.data.dataloader import default_collate
logger = logging.getLogger('template_video_classification')

class VGGish(nn.Module):
    """
    PyTorch implementation of the VGGish model.
    Adapted from: https://github.com/harritaylor/torch-vggish
    The following modifications were made: (i) correction for the missing ReLU layers, (ii) correction for the
    improperly formatted data when transitioning from NHWC --> NCHW in the fully-connected layers, and (iii)
    correction for flattening in the fully-connected layers.
    """

    def __init__(self, num_class=7, num_sec=5, drop_ratio=0.5, have_clssifier=False):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 24, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_ratio),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_ratio)
        )
        if have_clssifier:
            self.clssifier = nn.Linear(128 * num_sec, num_class)
        self.num_sec = num_sec
        self.have_clssifier = have_clssifier
        self.audio_embedding_size = 128 * num_sec

    def forward(self, x):
        #x = default_collate(x)
        #print(x.shape)
        #print(x[0])
        x = x.view(-1, 1, 96, 64)
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, self.num_sec, 128)
        x = x.view(x.size(0), -1)
        if self.have_clssifier:
            x = self.clssifier(x)
            return {'predict': {'fusion': x}}
        else:
            #print(x.shape)
            return x

    @property
    def input_size(self):
        return 1

    @property
    def input_mean(self):
        return 1

    @property
    def input_std(self):
        return 1

    @property
    def crop_size(self):
        return 1

    @property
    def scale_size(self):
        return 1

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


def main():
    pass


if __name__ == '__main__':
    main()