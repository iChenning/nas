import torch


__all__ = ['cross_entropy', 'auto_weight']

def loss_fun(opt):
    if opt.loss.type == 'cross_entropy':
        loss = torch.nn.CrossEntropyLoss()
        print('=>cross_entropy!')
    elif opt.loss.type == 'weight':
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        print('=>weight_cross_entropy!')
    else:
        assert False, 'opt.loss.type must belong to {cross_entropy weight}'

    return loss


def cross_entropy():
    return torch.nn.CrossEntropyLoss()


def auto_weight():
    return torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')


def weight_cal(opt, s, label):
    with torch.no_grad():
        tmp = s.clone()
        tmp = torch.nn.functional.softmax(tmp, dim=1)
        tmp_label = tmp[list(range(tmp.shape[0])), label].clone()
        tmp_argmax = torch.argmax(tmp, dim=1)
        tmp[list(range(tmp.shape[0])), label] = -10000.
        tmp_sec_max = tmp[list(range(tmp.shape[0])), torch.argmax(tmp, dim=1)]
        weight = (tmp_label - tmp_sec_max) ** opt.loss.weight_times
        weight[torch.where(~(label == tmp_argmax))] = 1.
    weight = torch.autograd.Variable(weight, requires_grad=True)
    return weight
