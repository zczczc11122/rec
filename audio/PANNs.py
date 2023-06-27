import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from audio.audio_models import Cnn14, init_layer
#from audio_models import Cnn14, init_layer


class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50,
                 fmax=14000, freeze_base=False):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527

        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,
                          fmax, audioset_classes_num)
        self.audio_embedding_size = 2048
        # Transfer to another task layer
        #self.fc_transfer = nn.Linear(2048, self.audio_embedding_size, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        # self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        return embedding

if __name__ == '__main__':
    model = Transfer_Cnn14().cuda()
    for _ in range(10):
        input = torch.rand((20, 32000 * 30)).cuda()
        out = model(input)
        print(out.shape)