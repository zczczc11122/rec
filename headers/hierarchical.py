# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

class HierachicalClassifier(nn.Module):

    def __init__(self, input_dim, out_dim, label_map, dropout_rate=0.1):
        super(HierachicalClassifier, self).__init__()
        # 每层编码网络
        self.hie_net = nn.ModuleList()
        # 每层输出网络
        self.hie_out = nn.ModuleList()
        # 每次连接网络
        self.hie_link = nn.ModuleList()

        for level in sorted(label_map.keys()):
            # level_label_num为每层标签大小
            self.hie_net.append(nn.Sequential(nn.Dropout(dropout_rate),
                                              nn.Linear(input_dim, out_dim),
                                              nn.GELU(),
                                              nn.LayerNorm(out_dim, eps=1e-12)))
            self.hie_out.append(nn.Sequential(nn.Dropout(dropout_rate),
                                nn.Linear(out_dim, len(label_map[level]['id2cls']))))

            if level > 0:
                self.hie_link.append(nn.Sequential(nn.Dropout(dropout_rate),
                                     nn.Linear(len(label_map[level -1]['id2cls']) + out_dim * (level + 1), out_dim),
                                     nn.GELU(),
                                     nn.LayerNorm(out_dim, eps=1e-12))
                )

    def forward(self, x):

        # x: b* input_dim
        level_feats = []
        level_outs = []
        for level, (level_net, level_out) in enumerate(zip(self.hie_net, self.hie_out)):
            # 单独获取每层的特征
            level_feat = level_net(x)
            level_feats.append(level_feat)

            if level == 0:
                first_out_feat = level_out(level_feat)
                level_outs.append(first_out_feat)
            else:
                temp_out = torch.cat(level_feats + [level_outs[-1]], dim=1)
                level_feat_ = self.hie_link[level-1](temp_out)
                # 弹出最后一个，添加一个最新的
                level_feats.pop(-1)
                level_feats.append(level_feat_)
                # 输出
                level_out_feat = level_out(level_feat_)
                level_outs.append(level_out_feat)
        return level_outs

class HierachicalClassifierSimple(nn.Module):

    def __init__(self, input_dim, out_dim, label_map, dropout_rate=0.1):
        super(HierachicalClassifierSimple, self).__init__()
        self.first_linear = nn.Sequential(nn.Dropout(p=dropout_rate),
                                          nn.Linear(input_dim, 512),
                                          nn.ReLU(inplace=True))
        self.last_linear = nn.Sequential(nn.Dropout(p=dropout_rate),
                                          nn.Linear(input_dim, 1024),
                                          nn.ReLU(inplace=True))
        levels = sorted(label_map.keys())

        self.first_classifier = nn.Sequential(nn.Dropout(p=dropout_rate),
                                              nn.Linear(512, len(label_map[levels[0]]['id2cls'])))
        self.last_classifier = nn.Sequential(nn.Dropout(p=dropout_rate),
                                              nn.Linear(1024, len(label_map[levels[-1]]['id2cls'])))

    def forward(self, x):
        first_level_hidden = self.first_linear(x)
        first_level_output = self.first_classifier(first_level_hidden)

        last_level_hidden = self.last_linear(x)
        last_level_output = self.last_classifier(last_level_hidden)

        level_outs = [first_level_output, last_level_output]

        return level_outs




