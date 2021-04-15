import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['dot', 'cos']


class DotScratch(nn.Module):
    def __init__(self, n_cla, emb_size=2048):
        super(DotScratch, self).__init__()
        self.fc = nn.Linear(emb_size, n_cla)

    def forward(self, f):
        logits = self.fc(f)
        return logits


class CosScratch(nn.Module):
    def __init__(self, n_cla, emb_size=2048, scale=100.):
        super(CosScratch, self).__init__()
        self.scale = scale
        self.fc = torch.nn.Parameter(torch.Tensor(n_cla, emb_size))
        nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))

    def forward(self, f):
        logits = F.linear(F.normalize(f), F.normalize(self.fc)) * self.scale
        return logits


def dot(n_cla, emb_size, scale):
    return DotScratch(n_cla, emb_size)


def cos(n_cla, emb_size, scale):
    return CosScratch(n_cla, emb_size, scale)

