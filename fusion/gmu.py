import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_

class GMU(nn.Module):
    '''
    模仿GMU源码写的（theano），不过源码中没实现多特征融合，只有2个特征融合，照着论文的图自己改了点
    '''

    def __init__(self, hidden_dims, output_dim):
        super(GMU, self).__init__()

        self.vision_hidden = hidden_dims[0]
        self.title_hidden = hidden_dims[1]
        self.ocr_hidden = hidden_dims[2]
        self.audio_hidden = hidden_dims[3]

        self.output_dim = output_dim

        self.vision_factor = Parameter(torch.Tensor(self.vision_hidden, self.output_dim))
        self.title_factor = Parameter(torch.Tensor(self.title_hidden, self.output_dim))
        self.ocr_factor = Parameter(torch.Tensor(self.ocr_hidden, self.output_dim))
        self.audio_factor = Parameter(torch.Tensor(self.audio_hidden, self.output_dim))

        self.gate_factor = Parameter(torch.Tensor(sum(hidden_dims), self.output_dim * len(hidden_dims)))

        self.activation = torch.nn.Tanh()
        self.gate_activation = torch.nn.Sigmoid()

        # init teh factors
        xavier_normal_(self.vision_factor)
        xavier_normal_(self.title_factor)
        xavier_normal_(self.ocr_factor)
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.gate_factor)

    def forward(self, input_f):
        vision, title, ocr, audio = input_f

        x = torch.cat((vision, title, ocr, audio), dim=1)
        z = self.gate_activation(torch.matmul(x, self.gate_factor))

        h_vision = self.activation(torch.matmul(vision, self.vision_factor))
        h_title = self.activation(torch.matmul(title, self.title_factor))
        h_ocr = self.activation(torch.matmul(ocr, self.ocr_factor))
        h_audio = self.activation(torch.matmul(audio, self.audio_factor))

        h = h_vision * z[:, 0 * self.output_dim: 1 * self.output_dim] + \
            h_title * z[:, 1 * self.output_dim: 2 * self.output_dim] + \
            h_ocr * z[:, 2 * self.output_dim: 3 * self.output_dim] + \
            h_audio * z[:, 3 * self.output_dim: 4 * self.output_dim]

        return h