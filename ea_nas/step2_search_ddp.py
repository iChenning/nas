import torch
import os
import argparse
import random
import numpy as np
import copy
import time
import logging
import sys
import gc
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp

from utils.data.data_dir import default_dir
from utils.data.augment import hard_trans
from utils.data.dataset import MyDataset
from torch.utils.data import DataLoader
from utils.log import logger_init
from utils.seed_init import rand_seed

from grow_nas.res_search import resnet_nas, choose_rand, load_supernet, remap_res2arch
import grow_nas.search_ga as ga
import utils.models as models
import utils.optimizers as optimizers


def search_step_ga(args, parents, supernet, logger):
    # data
    target_train_dir, target_valid_dir = default_dir[args.target_dataset]
    target_train_trans, target_valid_trans = hard_trans(args.target_img_size)

    target_trainset = MyDataset(target_train_dir, transform=target_train_trans)
    train_sampler = torch.utils.data.distributed.DistributedSampler(target_trainset)
    target_trainloader = DataLoader(target_trainset, batch_size=args.target_bs,
                                    shuffle=False, num_workers=8, pin_memory=True, sampler=train_sampler)

    target_validset = MyDataset(target_valid_dir, transform=target_valid_trans)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(target_validset)
    target_validloader = DataLoader(target_validset, batch_size=args.target_bs,
                                    shuffle=False, num_workers=8, pin_memory=True, sampler=valid_sampler)

    parents_acc = []
    for i_ga, v_ga in enumerate(parents):
        if args.local_rank == 0:
            logger.info('current progress:' + str(i_ga + 1) + '/' + str(len(parents)) + '...')
            logger.info(v_ga)

        # model
        arch = resnet_nas(v_ga)
        arch = copy.deepcopy(remap_res2arch(supernet, arch))
        fc = models.__dict__['cos'](target_trainloader.dataset.n_classes, arch.emb_size, 100.0)
        criterion = models.__dict__['cross_entropy'](1.0)
        arch = arch.cuda()
        fc = fc.cuda()
        criterion = criterion.cuda()

        # optimizer
        optimizer = optimizers.__dict__['sgd']((arch, fc), args.optim_lr)
        scheduler = optimizers.__dict__['warm_cos'](optimizer, args.target_warmup_epoch,
                                                    args.target_max_epoch, len(target_trainloader))

        # ddp
        arch = torch.nn.SyncBatchNorm.convert_sync_batchnorm(arch)
        arch = ddp(arch, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        fc = ddp(fc, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        # train
        for i_epoch in range(args.target_max_epoch):
            arch.train()
            fc.train()
            target_trainloader.sampler.set_epoch(i_epoch)
            correct = torch.tensor(0.0).cuda()
            total = torch.tensor(0.0).cuda()
            start_time = time.time()

            for i_iter, data in enumerate(target_trainloader):
                img, label = data[:2]
                img, label = img.cuda(), label.cuda()

                optimizer.zero_grad()
                f = arch(img)
                s = fc(f)
                loss = criterion(s, label)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # acc
                _, predicted = torch.max(s.data, 1)
                correct += predicted.eq(label.data).sum()
                total += label.size(0)

                # print info
                log_fre = len(target_trainloader) if args.target_log_fre == -1 else args.target_log_fre
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
                            i_epoch + 1, args.target_max_epoch,
                            i_iter + 1, len(target_trainloader),
                            optimizer.state_dict()['param_groups'][0]['lr'],
                            loss.cpu().item() / dist.get_world_size(),
                            correct.cpu().item() / total.cpu().item(),
                            eta))

        # valid
        with torch.no_grad():
            arch.eval()
            fc.eval()
            correct = torch.tensor([0.0]).cuda()
            total = torch.tensor([0.0]).cuda()

            for data in target_validloader:
                img, label = data[:2]
                img, label = img.cuda(), label.cuda()

                feature = arch(img)
                s = fc(feature)

                # acc
                _, predicted = torch.max(s.data, 1)
                correct += predicted.eq(label.data).sum()
                total += label.size(0)

            dist.all_reduce(correct, op=dist.ReduceOp.SUM)  # 这个地方就是个坑，之前用的reduce，导致各线程不一样，中途会卡住
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            if args.local_rank == 0:
                logger.info('valid-acc:{:.2%}'.format(correct.cpu().item() / total.cpu().item()))
                logger.info('--------------------------')
            parents_acc.append(correct / total)

        # 释放显存防止显存溢出
        del arch, fc, criterion, optimizer, scheduler
        del data, img, label, correct, total
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)
    return parents_acc


def search(args, logger):
    # 导入seed_net
    supernet = resnet_nas(args.supernet_choose)
    if args.supernet_dir is None:
        assert False, 'need source_arch_dir'
    supernet.load_state_dict(load_supernet(args.supernet_dir))

    # parents
    if args.local_rank == 0:
        logger.info('---------parents--------')
        logger.info('init need a little time...')
    all_population = [] if args.all_pop is None else args.all_pop
    parents = ga.init_popu(args.seed_choose,
                           args.stages, args.channel_scales, args.kernels,
                           seed_len=args.target_ga_len)
    all_population.extend(parents)
    parents_acc = search_step_ga(args, parents, supernet, logger)

    # ga search
    for i_evol in range(args.target_ga_iter):
        if args.local_rank == 0:
            logger.info('---------ga search--------')
            logger.info('ga search:' + str(i_evol + 1) + '/' + str(args.target_ga_iter))
        children = ga.cross_mutation(args.seed_choose, parents,
                                     args.stages, args.channel_scales, args.kernels,
                                     all_population, children_len=args.target_ga_len)
        if len(children) == 0:
            break
        all_population.extend(children)
        children_acc = search_step_ga(args, children, supernet, logger)

        parents.extend(children)
        parents_acc.extend(children_acc)
        parents, parents_acc = ga.update_popu(parents, parents_acc, k=args.target_ga_len)

    seed_choose, _ = ga.update_popu(parents, parents_acc, k=3)

    # print
    if args.local_rank == 0:
        logger.info('---------seed_choose--------')
        logger.info(seed_choose[0])
        logger.info('----------------------------')
    return seed_choose[0], all_population



def main(args):
    # dist init
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # log
    if args.local_rank == 0:
        time_str = datetime.strftime(datetime.now(), '%y-%m-%d-%H-%M-%S')
        args.log_dir = os.path.join('logs', args.target_dataset + '_' + time_str)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        logger = logger_init(args.log_dir)
    else:
        logger = None
    if args.local_rank == 0:
        logger.info(args)
        logger.info('-----------------------------------------------------------')

    # search
    search(args, logger)


if __name__ == "__main__":
    # -------------------   参数读取   ------------------------------
    parser = argparse.ArgumentParser("finetuning with Pytorch")

    # target data
    parser.add_argument('--target_dataset', type=str, default='cifar100',
                        help='The target dataset_name')
    parser.add_argument('--target_img_size', type=int, default=128,
                        help='Input image size.')
    parser.add_argument('--target_bs', type=int, default=128,  # 128
                        help='The batch-size of single-gpu training and validating.')
    parser.add_argument('--target_warmup_epoch', type=int, default=0,
                        help='warmup epoch.')
    parser.add_argument('--target_max_epoch', type=int, default=10,
                        help='max epoch.')
    parser.add_argument('--target_log_fre', type=int, default=-1,
                        help='log show fre.')
    parser.add_argument('--target_ga_iter', type=int, default=40,
                        help='target_ga_iter.')
    parser.add_argument('--target_ga_len', type=int, default=40,
                        help='target_ga_len.')

    parser.add_argument('--supernet_dir', type=str, default=None,
                        help='supernet dir.')

    # optimizer
    parser.add_argument('--optim_lr', type=float, default=1e-3,
                        help='The init learning rate.')

    # multi-gpu
    parser.add_argument("--local_rank", default=0, type=int,
                        help='master rank')

    args = parser.parse_args()
    rand_seed(0)

    args.stages = (tuple(range(6, 0, -1)),
                   tuple(range(8, 1, -1)),
                   tuple(range(12, 2, -1)),
                   tuple(range(6, 0, -1)))
    args.channel_scales = (1.5, 1.25, 1., 0.75, 0.5, 0.25)
    args.kernels = [(1, 7), (1, 5), (1, 3)]
    args.supernet_choose = choose_rand(([6], [8], [12], [6]), (1.5,), [(1, 7)])
    args.seed_choose = choose_rand(([3], [4], [6], [3]), (1.0,), [(1, 3)])
    args.all_pop = None

    main(args)
