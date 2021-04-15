import torch
import os
import argparse
import random
import numpy as np
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp
import logging
import sys
import time

from utils.data.data_dir import default_dir
from utils.data.augment import hard_trans
from utils.data.dataset import MyDataset
from torch.utils.data import DataLoader
from utils.log import logger_init
from utils.seed_init import rand_seed

from ea_nas.res_supernet import resnet_nas, choose_rand
import utils.models as models
import utils.optimizers as optimizers


def run(args, encoder, fc, criterion, optimizer, scheduler, trainloader, validloader, logger):
    # train
    for i_epoch in range(args.max_epoch):
        encoder.train()
        fc.train()
        trainloader.sampler.set_epoch(i_epoch)
        correct = torch.tensor(0.0).cuda()
        total = torch.tensor(0.0).cuda()
        start_time = time.time()

        for i_iter, data in enumerate(trainloader):
            img, label = data[:2]
            img, label = img.cuda(), label.cuda()

            choose = epoch_choose(i_epoch)

            optimizer.zero_grad()
            f = encoder(img, choose)
            s = fc(f)
            loss = criterion(s, label)
            loss.backward()
            optimizer.step()

            # acc
            _, predicted = torch.max(s.data, 1)
            correct += predicted.eq(label.data).sum()
            total += label.size(0)

            # print info
            log_fre = len(trainloader) if args.log_fre == -1 else args.log_fre
            if (i_iter + 1) % log_fre == 0:
                correct_tmp = correct.clone()
                total_tmp = total.clone()
                dist.reduce(correct_tmp, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(total_tmp, dst=0, op=dist.ReduceOp.SUM)
                if args.local_rank == 0:
                    eta = (time.time() - start_time) / 60.
                    logger.info("Training: Epoch[{:0>3}/{:0>3}] "
                                "Iter[{:0>3}/{:0>3}] "
                                "lr: {:.5f} "
                                "Loss: {:.4f} "
                                "Acc:{:.2%} "
                                "Run-T:{:.2f}m".format(
                        i_epoch + 1, args.max_epoch,
                        i_iter + 1, len(trainloader),
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        loss.cpu().item() / dist.get_world_size(),
                        correct.cpu().item() / total.cpu().item(),
                        eta))

        scheduler.step()
        if args.local_rank == 0:
            torch.save(encoder.state_dict(), '%s/encoder.pth' % args.log_dir)
            torch.save(fc.state_dict(), '%s/fc.pth' % args.log_dir)

        # valid
        if i_epoch % 10 == 0:
            choose_valid = epoch_choose(0)
            valid(args, encoder, fc, validloader, logger, choose_valid)


def epoch_choose(i_epoch, s_epoch=60, times=1):
    epoches = [1 * times, 1 * times, 1 * times, 1 * times, 1 * times,
               4 * times, 4 * times, 4 * times, 4 * times, 4 * times,
               2 * times,
               1 * times, 1 * times]
    epoches = np.array(epoches)
    epoches = np.cumsum(epoches)
    if i_epoch < s_epoch:
        stages = (tuple(range(6, 5, -1)),
                  tuple(range(8, 7, -1)),
                  tuple(range(12, 11, -1)),
                  tuple(range(6, 5, -1)))
        channel_scales = (1.5,)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[0]:
        stages = (tuple(range(6, 5, -1)),
                  tuple(range(8, 7, -1)),
                  tuple(range(12, 11, -1)),
                  tuple(range(6, 5, -1)))
        channel_scales = (1.5, 1.25)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[1]:
        stages = (tuple(range(6, 5, -1)),
                  tuple(range(8, 7, -1)),
                  tuple(range(12, 11, -1)),
                  tuple(range(6, 5, -1)))
        channel_scales = (1.5, 1.25, 1.0)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[2]:
        stages = (tuple(range(6, 5, -1)),
                  tuple(range(8, 7, -1)),
                  tuple(range(12, 11, -1)),
                  tuple(range(6, 5, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[3]:
        stages = (tuple(range(6, 5, -1)),
                  tuple(range(8, 7, -1)),
                  tuple(range(12, 11, -1)),
                  tuple(range(6, 5, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75, 0.5)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[4]:
        stages = (tuple(range(6, 5, -1)),
                  tuple(range(8, 7, -1)),
                  tuple(range(12, 11, -1)),
                  tuple(range(6, 5, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[5]:
        stages = (tuple(range(6, 4, -1)),
                  tuple(range(8, 6, -1)),
                  tuple(range(12, 10, -1)),
                  tuple(range(6, 4, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[6]:
        stages = (tuple(range(6, 3, -1)),
                  tuple(range(8, 5, -1)),
                  tuple(range(12, 9, -1)),
                  tuple(range(6, 3, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[7]:
        stages = (tuple(range(6, 2, -1)),
                  tuple(range(8, 4, -1)),
                  tuple(range(12, 8, -1)),
                  tuple(range(6, 2, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[8]:
        stages = (tuple(range(6, 1, -1)),
                  tuple(range(8, 3, -1)),
                  tuple(range(12, 7, -1)),
                  tuple(range(6, 1, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[9]:
        stages = (tuple(range(6, 0, -1)),
                  tuple(range(8, 2, -1)),
                  tuple(range(12, 6, -1)),
                  tuple(range(6, 0, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[10]:
        stages = (tuple(range(6, 0, -1)),
                  tuple(range(8, 1, -1)),
                  tuple(range(12, 5, -1)),
                  tuple(range(6, 0, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[11]:
        stages = (tuple(range(6, 0, -1)),
                  tuple(range(8, 1, -1)),
                  tuple(range(12, 4, -1)),
                  tuple(range(6, 0, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
        kernels = [(1, 3), (1, 5), (1, 7)]
    elif i_epoch < s_epoch + epoches[12]:
        stages = (tuple(range(6, 0, -1)),
                  tuple(range(8, 1, -1)),
                  tuple(range(12, 3, -1)),
                  tuple(range(6, 0, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
        kernels = [(1, 3), (1, 5), (1, 7)]
    else:
        stages = (tuple(range(6, 0, -1)),
                  tuple(range(8, 1, -1)),
                  tuple(range(12, 2, -1)),
                  tuple(range(6, 0, -1)))
        channel_scales = (1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
        kernels = [(1, 3), (1, 5), (1, 7)]
    choose = choose_rand(stages, channel_scales, kernels)
    return choose


def valid(args, encoder, fc, validloader, logger, choose_valid):
    with torch.no_grad():
        encoder.eval()
        fc.eval()
        correct = torch.tensor([0.0]).cuda()
        total = torch.tensor([0.0]).cuda()
        for data in validloader:
            img, label = data[:2]
            img, label = img.cuda(), label.cuda()

            feature = encoder(img, choose_valid)
            s = fc(feature)

            # acc
            _, predicted = torch.max(s.data, 1)
            correct += predicted.eq(label.data).sum()
            total += label.size(0)
        dist.reduce(correct, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total, dst=0, op=dist.ReduceOp.SUM)
        if args.local_rank == 0:
            logger.info('valid-acc:{:.2%}'.format(correct.cpu().item() / total.cpu().item()))
            logger.info('--------------------------')


def main(args):
    # dist init
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # dataloader
    train_dir, valid_dir = default_dir[args.dataset]
    train_trans, valid_trans = hard_trans(args.img_size)

    trainset = MyDataset(train_dir, transform=train_trans)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=False,
                             num_workers=8, pin_memory=True, sampler=train_sampler)

    validset = MyDataset(valid_dir, transform=valid_trans)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(validset)
    validloader = DataLoader(validset, batch_size=args.bs, shuffle=False,
                             num_workers=8, pin_memory=True, sampler=valid_sampler)

    # model
    supernet_choose = choose_rand(([6], [8], [12], [6]), (1.5,), [(1, 7)])
    supernet = resnet_nas(supernet_choose)
    fc = models.__dict__['cos'](trainloader.dataset.n_classes, supernet.emb_size, 100.0)
    criterion = models.__dict__['cross_entropy'](1.0)
    supernet = supernet.cuda()
    fc = fc.cuda()
    criterion = criterion.cuda()

    # optimizer
    optimizer = optimizers.__dict__['sgd']((supernet, fc), args.optim_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.optim_step, gamma=0.1)

    # ddp
    supernet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(supernet)
    supernet = ddp(supernet, device_ids=[args.local_rank], find_unused_parameters=True)
    fc = ddp(fc, device_ids=[args.local_rank], find_unused_parameters=True)

    # log
    if args.local_rank == 0:
        time_str = datetime.strftime(datetime.now(), '%y-%m-%d-%H-%M-%S')
        args.log_dir = os.path.join('logs', args.dataset + '_' + time_str)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        logger = logger_init(args.log_dir)
    else:
        logger = None
    if args.local_rank == 0:
        logger.info(args)
        logger.info('n_classes:%d' % trainloader.dataset.n_classes)

    # train and valid
    run(args, supernet, fc, criterion, optimizer, scheduler, trainloader, validloader, logger)
    if args.local_rank == 0:
        logger.info(args.log_dir)


if __name__ == "__main__":
    # -------------------   参数读取   ------------------------------
    parser = argparse.ArgumentParser("finetuning with Pytorch")
    # data
    parser.add_argument('--dataset', type=str, default='image3403',
                        help='The dataset_name')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size.')
    parser.add_argument('--bs', type=int, default=64,
                        help='The batch-size of single-gpu training and validating.')

    # optimizer
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer type.')
    parser.add_argument('--optim_lr', type=float, default=2e-1,
                        help='The init learning rate.')
    parser.add_argument('--optim_step', type=list, default=[100, 160, 200],
                        help='optim step.')
    parser.add_argument('--max_epoch', type=int, default=250,
                        help='max epoch.')

    # GPU
    parser.add_argument("--local_rank", default=0, type=int,
                        help='master rank')

    # info
    parser.add_argument('--log_fre', type=int, default=50,
                        help='log show fre.')

    args = parser.parse_args()
    rand_seed(0)
    main(args)
