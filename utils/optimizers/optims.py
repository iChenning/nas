import torch

from utils.optimizers.sam import SAM


__all__ = ['sgd', 'sgd_sam']


def sgd(modules, lr=0.01, momentum=0.9, weight_decay=3e-4):
    params = []
    for module in modules:
        params.append({'params': module.parameters()})
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    return optimizer


def sgd_sam(modules, lr=0.01, momentum=0.9, weight_decay=3e-4):
    params = []
    for module in modules:
        params.append({'params': module.parameters()})
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    return optimizer
