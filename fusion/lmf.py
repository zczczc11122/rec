import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_

class LMF(nn.Module):
    '''
    Low-rank MutilModal Fusion
    '''

    def __init__(self, hidden_dims, output_dim, rank=4):
        super(LMF, self).__init__()

        self.vision_hidden = hidden_dims[0]
        self.title_hidden = hidden_dims[1]
        self.ocr_hidden = hidden_dims[2]
        self.audio_hidden = hidden_dims[3]

        self.rank = rank
        self.output_dim = output_dim

        self.vision_factor = Parameter(torch.Tensor(self.rank, self.vision_hidden + 1, self.output_dim))
        self.title_factor = Parameter(torch.Tensor(self.rank, self.title_hidden + 1, self.output_dim))
        self.ocr_factor = Parameter(torch.Tensor(self.rank, self.ocr_hidden + 1, self.output_dim))
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal_(self.vision_factor)
        xavier_normal_(self.title_factor)
        xavier_normal_(self.ocr_factor)
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, input_f):
        vision, title, ocr, audio = input_f
        batch_size = vision.data.shape[0]
        # if vision.is_cuda:
        #     DTYPE = torch.cuda.FloatTensor
        # else:
        #     DTYPE = torch.FloatTensor
        device = vision.device
        _vision_h = torch.cat((torch.ones([batch_size, 1], dtype=torch.float, requires_grad=False).to(device), vision), dim=1)
        _title_h = torch.cat((torch.ones([batch_size, 1], dtype=torch.float, requires_grad=False).to(device), title), dim=1)
        _ocr_h = torch.cat((torch.ones([batch_size, 1], dtype=torch.float, requires_grad=False).to(device), ocr), dim=1)
        _audio_h = torch.cat((torch.ones([batch_size, 1], dtype=torch.float, requires_grad=False).to(device), audio), dim=1)

        fusion_vision = torch.matmul(_vision_h, self.vision_factor)
        fusion_title = torch.matmul(_title_h, self.title_factor)
        fusion_ocr = torch.matmul(_ocr_h, self.ocr_factor)
        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_zy = fusion_vision * fusion_title * fusion_ocr * fusion_audio

        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)

        return output