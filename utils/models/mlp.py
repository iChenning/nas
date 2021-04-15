import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features=2048, hidden_features=4096, out_features=256):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features, hidden_features, bias=False),
                                 nn.BatchNorm1d(hidden_features),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_features, out_features, bias=False))

    def forward(self, f):
        return self.mlp(f)
