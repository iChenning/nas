import torch
import os
import argparse
import copy
import time
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
from grow_nas.res_search import resnet_nas, choose_rand, load_normal, remap_res2arch

import grow_nas.search_ga as ga
import utils.models as models
import utils.optimizers as optimizers


def search_step_ga(args, parents, seednet, logger):
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
        arch = copy.deepcopy(remap_res2arch(seednet, arch))
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
    seednet = resnet_nas(args.seed_choose)
    if args.source_arch_dir is None:
        assert False, 'need source_arch_dir'
    seednet.load_state_dict(load_normal(args.source_arch_dir))

    # parents
    if args.local_rank == 0:
        logger.info('---------parents--------')
    all_population = [] if args.all_pop is None else args.all_pop
    parents = ga.init_popu(args.seed_choose,
                           args.stages, args.channel_scales, args.kernels,
                           seed_len=args.target_ga_len)
    all_population.extend(parents)
    parents_acc = search_step_ga(args, parents, seednet, logger)

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
        children_acc = search_step_ga(args, children, seednet, logger)

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


def pretrain_finetune(args, logger):
    # data
    source_train_dir, source_valid_dir = default_dir[args.source_dataset]
    source_train_trans, source_valid_trans = hard_trans(args.source_img_size)

    source_trainset = MyDataset(source_train_dir, transform=source_train_trans)
    train_sampler = torch.utils.data.distributed.DistributedSampler(source_trainset)
    source_trainloader = DataLoader(source_trainset, batch_size=args.source_bs,
                                    shuffle=False, num_workers=8, pin_memory=True, sampler=train_sampler)

    source_validset = MyDataset(source_valid_dir, transform=source_valid_trans)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(source_validset)
    source_validloader = DataLoader(source_validset, batch_size=args.target_bs,
                                    shuffle=False, num_workers=8, pin_memory=True, sampler=valid_sampler)

    # model, 导入seed_net
    basenet = resnet_nas(args.base_choose)
    if args.source_arch_dir is None:
        assert False, 'need source_arch_dir'
    basenet.load_state_dict(load_normal(args.source_arch_dir))
    seednet = resnet_nas(args.seed_choose)
    seednet = copy.deepcopy(remap_res2arch(basenet, seednet))
    fc = models.__dict__['cos'](source_trainloader.dataset.n_classes, seednet.emb_size, 100.0)
    criterion = models.__dict__['cross_entropy'](1.0)
    seednet = seednet.cuda()
    fc = fc.cuda()
    criterion = criterion.cuda()

    # optimizer
    optimizer = optimizers.__dict__['sgd']((seednet, fc), args.optim_lr)
    scheduler = optimizers.__dict__['warm_cos'](optimizer, args.source_warmup_epoch,
                                                args.source_max_epoch, len(source_trainloader))

    # ddp
    seednet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(seednet)
    seednet = ddp(seednet, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    fc = ddp(fc, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # train
    for i_epoch in range(args.source_max_epoch):
        seednet.train()
        fc.train()
        source_trainloader.sampler.set_epoch(i_epoch)
        correct = torch.tensor(0.0).cuda()
        total = torch.tensor(0.0).cuda()
        start_time = time.time()

        for i_iter, data in enumerate(source_trainloader):
            img, label = data[:2]
            img, label = img.cuda(), label.cuda()

            optimizer.zero_grad()
            f = seednet(img)
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
            log_fre = len(source_trainloader) if args.source_log_fre == -1 else args.source_log_fre
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
                        i_epoch + 1, args.source_max_epoch,
                        i_iter + 1, len(source_trainloader),
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        loss.cpu().item() / dist.get_world_size(),
                        correct.cpu().item() / total.cpu().item(),
                        eta))
        if args.local_rank == 0:
            if not os.path.exists('tmp_model'):
                os.makedirs('tmp_model')
            torch.save(seednet.state_dict(), 'tmp_model/encoder.pth')
            torch.save(fc.state_dict(), 'tmp_model/fc.pth')

        # valid
        with torch.no_grad():
            seednet.eval()
            fc.eval()
            correct = torch.tensor([0.0]).cuda()
            total = torch.tensor([0.0]).cuda()
            for data in source_validloader:
                img, label = data[:2]
                img, label = img.cuda(), label.cuda()

                feature = seednet(img)
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

    # 释放显存防止显存溢出
    del basenet, seednet, fc, criterion, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

    source_arch_dir = 'tmp_model/encoder.pth'
    return source_arch_dir


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

    # search and pretrained-fine-tuning
    for i in range(args.whole_iter):
        # search
        if args.local_rank == 0:
            logger.info('----------search----------')
            logger.info('whole progress:' + str(i + 1) + '/' + str(args.whole_iter))
            logger.info('--------------------------')
        seed_choose, args.all_pop = search(args, logger)
        args.base_choose = args.seed_choose
        args.seed_choose = seed_choose

        # pretrained fine-tuning
        if args.local_rank == 0:
            logger.info('----------pretrained----------')
            logger.info('whole progress:' + str(i + 1) + '/' + str(args.whole_iter))
            logger.info('------------------------------')
        args.source_arch_dir = pretrain_finetune(args, logger)

    # # release process
    # dist.destroy_process_group()


if __name__ == "__main__":
    # -------------------   参数读取   ------------------------------
    parser = argparse.ArgumentParser("finetuning with Pytorch")
    parser.add_argument('--whole_iter', type=int, default=10,
                        help='The source dataset_name')

    # target data
    parser.add_argument('--target_dataset', type=str, default='cifar100_gz',
                        help='The target dataset_name')
    parser.add_argument('--target_img_size', type=int, default=128,
                        help='Input image size.')
    parser.add_argument('--target_bs', type=int, default=128,  # 128
                        help='The batch-size of single-gpu training and validating.')
    parser.add_argument('--target_warmup_epoch', type=int, default=0,
                        help='warmup epoch.')
    parser.add_argument('--target_max_epoch', type=int, default=5,
                        help='max epoch.')
    parser.add_argument('--target_log_fre', type=int, default=-1,
                        help='log show fre.')
    parser.add_argument('--target_ga_iter', type=int, default=2,
                        help='target_ga_iter.')
    parser.add_argument('--target_ga_len', type=int, default=40,
                        help='target_ga_len.')

    # source data
    parser.add_argument('--source_dataset', type=str, default='image3403',
                        help='The source dataset_name')
    parser.add_argument('--source_img_size', type=int, default=224,
                        help='Input image size.')
    parser.add_argument('--source_bs', type=int, default=128,
                        help='The batch-size of single-gpu training and validating.')
    parser.add_argument('--source_arch_dir', type=str, default='../expert/models/resnet50_pytorch.pth',
                        help="The pretrained model's save-path.")
    parser.add_argument('--source_warmup_epoch', type=int, default=0,
                        help='warmup epoch.')
    parser.add_argument('--source_max_epoch', type=int, default=10,
                        help='max epoch.')
    parser.add_argument('--source_log_fre', type=int, default=50,
                        help='log show fre.')

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
    args.seed_choose = choose_rand(([3], [4], [6], [3]), (1.0,), [(1, 3)])
    args.all_pop = None

    main(args)
