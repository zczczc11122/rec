from typing import Any

import torch

class SqueezeContextGating(torch.nn.Module):

    def __init__(self, feature_size,
                 reduction_ratio=8,
                 add_batch_norm=False) -> None:
        super().__init__()
        self.add_batch_norm = add_batch_norm
        self.gating_weights_1 = torch.nn.Linear(in_features=feature_size,
                                                out_features=feature_size // reduction_ratio, bias=False)
        self.gating_weights_2 = torch.nn.Linear(in_features=feature_size // reduction_ratio,
                                                out_features=feature_size, bias=False)
        self.bn = torch.nn.BatchNorm1d(num_features=feature_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        gates = self.gating_weights_1(x)

        if self.add_batch_norm:
            gates = self.bn(gates)
        gates = self.gating_weights_2(gates)
        gates = self.sigmoid(gates)

        activation = torch.mul(x, gates)

        return activation

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

class ContextGating(torch.nn.Module):
    #https://github.com/antoine77340/LOUPE/blob/master/loupe.py#L59

    def __init__(self, feature_size,
                 add_batch_norm=True):
        super().__init__()
        self.add_batch_norm = add_batch_norm
        if add_batch_norm:
            self.gating_weights = torch.nn.Linear(in_features=feature_size,
                                                   out_features=feature_size, bias=False)
            self.bn = torch.nn.BatchNorm1d(num_features=feature_size)
        else:
            self.gating_weights = torch.nn.Linear(in_features=feature_size,
                                                  out_features=feature_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        if self.add_batch_norm:
            gates = self.gating_weights(x)
            gates = self.bn(gates)
        else:
            gates = self.gating_weights(x)
        gates = self.sigmoid(gates)
        activation = torch.mul(x, gates)

        return activation
