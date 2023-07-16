import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#猜测代码来源：https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/TextRCNN.py

class TextRCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_size = args.textrcnn_hidden_size
        num_layers = args.textrcnn_num_layers
        pad_size = args.text_max_size
        text_embedding_size = args.text_output_size

        self.word_embedding_pretrained = torch.tensor(np.load(args.word_embedding_path)['embeddings'].astype('float32'))
        self.word_embedding = nn.Embedding.from_pretrained(self.word_embedding_pretrained, freeze=True)
        self.word_embedding_dim = self.word_embedding.embedding_dim

        self.lstm = nn.LSTM(self.word_embedding_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=0.3)
        self.max_pool = nn.MaxPool1d(pad_size)
        self.linear = nn.Linear(hidden_size * 2 + self.word_embedding_dim, text_embedding_size)

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
