import torch
import os
import math
import argparse
import random
import numpy as np
import copy
from datetime import datetime
import torch.multiprocessing as mp
import torch.distributed as dist
import time
import logging
import sys

from utils.data.data_dir import default_dir
from utils.data.augment import hard_trans
from utils.data.dataset import MyDataset
from torch.utils.data import DataLoader

from ea_nas.res_search import resnet_nas, choose_rand, remap_res2arch, load_supernet, load_normal
import utils.models as models
import utils.optimizers as optimizers

from utils.log_new import LogSupervised

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def logger_init(folder_dir):
    logger = logging.getLogger('run')
    logger.setLevel(level=logging.INFO)
    # streamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # fileHandler
    file_path = os.path.join(folder_dir, 'log.log')
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def rand_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def valid(args, encoder, fc, validloader):
    with torch.no_grad():
        encoder.eval()
        fc.eval()

        rank = dist.get_rank()
        correct = torch.tensor([0.0]).to(rank)
        total = torch.tensor([0.0]).to(rank)
        for data in validloader:
            img, label = data[:2]
            img, label = img.to(rank), label.to(rank)

            feature = encoder(img)
            s = fc(feature)

            # acc
            _, predicted = torch.max(s.data, 1)
            correct += predicted.eq(label.data).sum()
            total += s.shape[0]

        # print
        dist.reduce(correct, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            print('acc:{:.2%}'.format(
                correct.cpu().item() / total.cpu().item()))


def run(args, encoder, fc, criterion, optimizer, scheduler, trainloader, validloader, logger):
    # train
    for i_epoch in range(args.max_epoch):
        encoder.train()
        fc.train()
        trainloader.sampler.set_epoch(i_epoch)  # 不可少

        rank = dist.get_rank()
        correct = torch.tensor(0.0).cuda(rank)
        total = torch.tensor(0.0).cuda(rank)
        start_time = torch.tensor(time.time()).cuda(rank)
        for i_iter, data in enumerate(trainloader):
            img, label = data[:2]
            img, label = img.cuda(rank), label.cuda(rank)

            f = encoder(img)
            s = fc(f)
            loss = criterion(s, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # acc
            _, predicted = torch.max(s.data, 1)
            correct += predicted.eq(label.data).sum()
            total += s.shape[0]
            eta = (time.time() - start_time) / (i_iter + 1) * (len(trainloader) * (
                    args.max_epoch - i_epoch) - i_iter) / 3600

            # print
            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(correct, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(eta, dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                logger.info('loss:{:.4f} '
                      'acc:{:.2%} '
                      'ETA:{:.2f}h'.format(
                    loss.cpu().item() / args.world_size,
                    correct.cpu().item() / total.cpu().item(),
                    eta.cpu().item() / args.world_size))

        valid(args, encoder, fc, validloader)


def main(rank, args):
    # dist init
    dist.init_process_group("nccl", init_method='tcp://localhost:12345', rank=rank, world_size=args.world_size)

    # dataloader
    train_dir, valid_dir = default_dir[args.dataset]
    train_trans, valid_trans = hard_trans(args.img_size)

    trainset = MyDataset(train_dir, transform=train_trans)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=args.world_size, rank=rank)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=False,
                             num_workers=8, pin_memory=True, sampler=train_sampler)

    validset = MyDataset(valid_dir, transform=valid_trans)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        validset, num_replicas=args.world_size, rank=rank)
    validloader = DataLoader(validset, batch_size=args.bs, shuffle=False,
                             num_workers=8, pin_memory=True, sampler=valid_sampler)

    # model
    # ----------------------修改此处根据步骤2获得结构-------------------------------
    arch_choose = [[3, [[1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)]]],
                   [4, [[1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)]]],
                   [6, [[1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)]]],
                   [3, [[1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)]]]]
    # ----------------------------------------------------------------------------
    arch = resnet_nas(arch_choose)
    if args.arch_dir is not None:
        arch.load_state_dict(load_normal(args.arch_dir))
        print('load success!')
    fc = models.__dict__[args.fc](trainloader.dataset.n_classes, arch.emb_size, args.fc_scale)
    if args.fc_dir is not None:
        fc.load_state_dict(load_normal(args.fc_dir))
    criterion = models.__dict__[args.criterion](args.criterion_times)

    arch = arch.cuda(rank)
    fc = fc.cuda(rank)
    criterion = criterion.cuda(rank)

    # optimizer
    if torch.cuda.is_available() and args.multi_gpu and torch.cuda.device_count() > 1:
        args.optim_lr *= round(math.sqrt(torch.cuda.device_count()))
    optimizer = optimizers.__dict__[args.optim]((arch, fc), args.optim_lr)
    scheduler = optimizers.__dict__[args.optim_lr_mul](optimizer, args.warmup_epoch,
                                                       args.max_epoch, len(trainloader))

    arch = torch.nn.SyncBatchNorm.convert_sync_batchnorm(arch)
    arch = torch.nn.parallel.DistributedDataParallel(arch, device_ids=[rank], find_unused_parameters=True)
    fc = torch.nn.parallel.DistributedDataParallel(fc, device_ids=[rank], find_unused_parameters=True)


    if rank == 0:
        time_str = datetime.strftime(datetime.now(), '%y-%m-%d-%H-%M-%S')
        args.log_dir = os.path.join('logs', args.dataset + '_' + time_str)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        logger = logger_init(args.log_dir)

    if rank != 0:
        dist.barrier()
    # train and valid
    run(args, arch, fc, criterion, optimizer, scheduler, trainloader, validloader, logger)


if __name__ == "__main__":
    # -------------------   参数读取   ------------------------------
    parser = argparse.ArgumentParser("finetuning with Pytorch")
    # data
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='The dataset_name')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Input image size.')
    parser.add_argument('--bs', type=int, default=128,
                        help='The batch-size of single-gpu training and validating.')

    # model
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='The architecture of encoder!')
    parser.add_argument('--arch_dir', type=str, default=None,
                        help="The pretrained model's save-path.")
    parser.add_argument('--fc', type=str, default='cos',
                        help='The metric of classifier!')
    parser.add_argument('--fc_dir', type=str, default=None,
                        help="The pretrained fc's save-path.")
    parser.add_argument('--fc_scale', type=float, default=100.,
                        help='It does not work for fc-dot.')
    parser.add_argument('--criterion', type=str, default='cross_entropy',
                        help="The criterion type.")
    parser.add_argument('--criterion_times', type=float, default=2.,
                        help='It only works for auto_weight.')

    # optimizer
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer type.')
    parser.add_argument('--optim_lr', type=float, default=1e-3,
                        help='The init learning rate.')
    parser.add_argument('--optim_lr_mul', type=str, default='warm_cos',
                        help='The init learning rate.')
    parser.add_argument('--warmup_epoch', type=int, default=3,
                        help='warmup epoch.')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='max epoch.')

    # GPU
    parser.add_argument('--multi_gpu', type=bool, default=True,
                        help='Is running on multi-gpu?')

    # info
    parser.add_argument('--log_note', type=str, default=None,
                        help='logger info.')
    parser.add_argument('--log_fre', type=int, default=25,
                        help='log show fre.')
    args = parser.parse_args()

    rand_seed(0)
    args.world_size = torch.cuda.device_count()  # 单机多卡版本


    mp.spawn(main, nprocs=args.world_size, args=(args,))
