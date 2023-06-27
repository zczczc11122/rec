import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#猜测代码来源：https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/TextCNN.py

class TextCNN(nn.Module):

    def __init__(self, args):
        super().__init__()
        num_filters = args.textcnn_num_filters
        filter_sizes = [int(s) for s in args.textcnn_filter_sizes.split(',')]
        text_embedding_size = args.text_output_size
        self.text_embedding_size = args.text_output_size
        self.word_embedding_pretrained = torch.tensor(np.load(args.word_embedding_path)['embeddings'].astype('float32'))
        self.word_embedding = nn.Embedding.from_pretrained(self.word_embedding_pretrained, freeze=True)
        word_embedding_dim = self.word_embedding.embedding_dim
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, word_embedding_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(args.text_dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), text_embedding_size)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.word_embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
