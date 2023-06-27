from typing import Any

import torch.nn as nn
from vision import SqueezeContextGating

def general_weight_initialization(module: nn.Module):
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        # print("Initing linear")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class ConcatenateDense(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=4096, out_features=args.fusion_output_size),
            #nn.Linear(in_features=args.vlad_hidden_size + args.text_output_size, out_features=args.fusion_output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fusion_layer(x)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class ConcatenateDenseSE(nn.Module):
    #SqueezeContextGating
    def __init__(self, vlad_hidden_size, text_output_size, audo_output_size, args):
        super().__init__()
        print(vlad_hidden_size + text_output_size + text_output_size + audo_output_size)
        self.fusion_layer = nn.Sequential(
            nn.Dropout(p=args.dropout),
            # nn.Linear(in_features=vlad_hidden_size + text_output_size + text_output_size + audo_output_size,
            nn.Linear(in_features=4096,
                      out_features=args.fusion_output_size),
            nn.ReLU()
        )
        self.se = SqueezeContextGating(args.fusion_output_size)

    def forward(self, x):
        print(x.shape)
        x = self.fusion_layer(x)
        x = self.se(x)
        return x

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class BNSEModule(nn.Module):
    """Squeeze-and-excite context gating
    """

    def __init__(self, channels, reduction):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(channels)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            general_weight_initialization(module)

    def forward(self, x):
        module_input = x
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x