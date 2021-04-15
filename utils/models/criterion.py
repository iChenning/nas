import torch


__all__ = ['cross_entropy', 'auto_weight']


class CrossEntropy(torch.nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, s, label):
        return self.criterion(s, label)


def cross_entropy(weight_times):
    return CrossEntropy()


class AutoWeight(torch.nn.Module):
    def __init__(self, weight_times=2.):
        super(AutoWeight, self).__init__()
        self.weight_times = weight_times
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    def forward(self, s, label):
        with torch.no_grad():
            tmp = s.clone()
            tmp = torch.nn.functional.softmax(tmp, dim=1)
            tmp_label = tmp[list(range(tmp.shape[0])), label].clone()
            tmp_argmax = torch.argmax(tmp, dim=1)
            tmp[list(range(tmp.shape[0])), label] = -10000.
            tmp_sec_max = tmp[list(range(tmp.shape[0])), torch.argmax(tmp, dim=1)]
            weight = (tmp_label - tmp_sec_max) ** self.weight_times
            weight[torch.where(~(label == tmp_argmax))] = 1.
        weight = torch.autograd.Variable(weight, requires_grad=True)

        loss = self.criterion(s, label) * (torch.log(1 / (weight + 1e-7)) + 1.)
        loss = torch.mean(loss)
        return loss


def auto_weight(weight_times):
    return AutoWeight(weight_times=weight_times)

