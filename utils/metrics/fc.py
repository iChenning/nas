import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['dot', 'dot_mean', 'cos', 'cos_mean']


class DotScratch(nn.Module):
    def __init__(self, opt, emb_size=2048):
        super(DotScratch, self).__init__()
        n_cla = opt.n_classes if 'n_classes' in opt.keys() else opt.n_way
        self.fc = nn.Linear(emb_size, n_cla)

    def forward(self, f):
        logits = self.fc(f)
        return logits


class DotMean(nn.Module):
    def __init__(self, opt, emb_size, features, labels):
        super(DotMean, self).__init__()
        n_cla = opt.n_classes if 'n_classes' in opt.keys() else opt.n_way

        weight = torch.zeros(n_cla, emb_size)
        weight_nums = torch.zeros(n_cla, 1)
        for i_ in range(features.shape[0]):
            weight[labels[i_], :] += features[i_, :]  # 此处要求labels必须是从0开始且连续的
            weight_nums[labels[i_]] += 1.
        weight /= weight_nums
        self.fc = torch.nn.Parameter(weight)

    def forward(self, f):
        logits = F.linear(f, self.fc)
        return logits


class CosScratch(nn.Module):
    def __init__(self, opt, emb_size=2048):
        super(CosScratch, self).__init__()
        n_cla = opt.n_classes if 'n_classes' in opt.keys() else opt.n_way
        self.scale = opt.fc.scale
        self.fc = torch.nn.Parameter(torch.Tensor(n_cla, emb_size))
        nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))

    def forward(self, f):
        logits = F.linear(F.normalize(f), F.normalize(self.fc)) * self.scale
        # print('classifier-inter batch-size:', logits.shape[0])
        return logits


class CosMean(nn.Module):
    def __init__(self, opt, emb_size, features, labels):
        super(CosMean, self).__init__()
        n_cla = opt.n_classes if 'n_classes' in opt.keys() else opt.n_way
        self.scale = opt.fc.scale
        weight = torch.zeros(n_cla, emb_size)
        weight_nums = torch.zeros(n_cla, 1)
        for i_ in range(features.shape[0]):
            weight[labels[i_], :] += features[i_, :]  # 此处要求labels必须是从0开始且连续的
            weight_nums[labels[i_]] += 1.
        weight /= weight_nums
        weight = F.normalize(weight)
        self.fc = torch.nn.Parameter(weight)

    def forward(self, f):
        logits = F.linear(F.normalize(f), F.normalize(self.fc)) * self.scale
        return logits


def dot(opt, emb_size):
    return DotScratch(opt, emb_size)


def dot_mean(opt, emb_size, features, labels):
    return DotMean(opt, emb_size, features, labels)


def cos(opt, emb_size):
    return CosScratch(opt, emb_size)


def cos_mean(opt, emb_size, features, labels):
    return CosMean(opt, emb_size, features, labels)


def metric_fc(opt, emb_size=2048, features=None, labels=None):
    if opt.fc.type == 'dot':
        if opt.fc.init_type == 'mean':
            fc = DotMean(opt, emb_size, features, labels)
            print("=>Initialized dot-fc by mean!")
        elif opt.fc.init_type == 'load':
            fc = DotScratch(opt, emb_size)
            state_dict = torch.load(opt.fc.load_path, map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v
            fc.load_state_dict(new_state_dict).to(opt.device)
            print("=>Initialized dot-fc by pretrained-fc!")
        else:
            fc = DotScratch(opt, emb_size)
            print("=>Initialized dot-fc by random!")

    elif opt.fc.type == 'cos':
        if opt.fc.init_type == 'mean':
            fc = CosMean(opt, emb_size, features, labels)
            print("=>Initialized cos-fc by mean!")
        elif opt.fc.init_type == 'load':
            fc = CosScratch(opt, emb_size)
            state_dict = torch.load(opt.fc.load_path, map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v
            fc.load_state_dict(new_state_dict).to(opt.device)
            print("=>Initialized cos-fc by pretrained-fc!")
        else:
            fc = CosScratch(opt, emb_size)
            print("=>Initialized cos-fc by random!")
    else:
        assert False, 'opt.fc.type must belong to {dot cos}'

    return fc
