import torch
from torch.optim.lr_scheduler import LambdaLR
import math

from utils.optimizers.sam import SAM


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def sgd(opt, modules):
    params = []
    for module in modules:
        params.append({'params': module.parameters()})
    optimizer = torch.optim.SGD(params,
                                lr=opt.optim.lr_init, momentum=opt.optim.momentum,
                                weight_decay=opt.optim.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.optim.lr_mul, gamma=opt.optim.lr_gamma)
    return (optimizer, scheduler)


def sgd_sam(opt, modules):
    params = []
    for module in modules:
        params.append({'params': module.parameters()})
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer,
                    lr=opt.optim.lr_init, momentum=opt.optim.momentum,
                    weight_decay=opt.optim.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.optim.lr_mul, gamma=opt.optim.lr_gamma)
    return (optimizer, scheduler)
