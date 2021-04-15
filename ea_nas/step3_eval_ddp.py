import torch
import os
import argparse
from datetime import datetime
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp

from utils.data.data_dir import default_dir
from utils.data.augment import hard_trans
from utils.data.dataset import MyDataset
from torch.utils.data import DataLoader
from utils.log import logger_init
from utils.seed_init import rand_seed

from ea_nas.res_search import resnet_nas, choose_rand, remap_res2arch, load_supernet, load_normal
import utils.models as models
import utils.optimizers as optimizers


def valid(args, encoder, fc, validloader, logger):
    with torch.no_grad():
        encoder.eval()
        fc.eval()

        correct = torch.tensor([0.0]).cuda()
        total = torch.tensor([0.0]).cuda()
        for data in validloader:
            img, label = data[:2]
            img, label = img.cuda(), label.cuda()

            feature = encoder(img)
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


def run(args, encoder, fc, criterion, optimizer, scheduler, trainloader, validloader, logger):
    # train
    for i_epoch in range(args.max_epoch):
        encoder.train()
        fc.train()
        trainloader.sampler.set_epoch(i_epoch)  # 不可少
        correct = torch.tensor(0.0).cuda()
        total = torch.tensor(0.0).cuda()
        start_time = time.time()

        for i_iter, data in enumerate(trainloader):
            img, label = data[:2]
            img, label = img.cuda(), label.cuda()

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

        valid(args, encoder, fc, validloader, logger)


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
    arch = resnet_nas(args.arch_choose)
    if args.arch_dir is not None:
        arch.load_state_dict(load_normal(args.arch_dir))
        print('load success!')
    fc = models.__dict__['cos'](trainloader.dataset.n_classes, arch.emb_size, 100.0)
    criterion = models.__dict__['cross_entropy'](1.0)
    arch = arch.cuda()
    fc = fc.cuda()
    criterion = criterion.cuda()

    # optimizer
    optimizer = optimizers.__dict__['sgd']((arch, fc), args.optim_lr)
    scheduler = optimizers.__dict__['warm_cos'](optimizer, args.warmup_epoch,
                                                args.max_epoch, len(trainloader))

    # ddp
    arch = torch.nn.SyncBatchNorm.convert_sync_batchnorm(arch)
    arch = ddp(arch, device_ids=[args.local_rank], find_unused_parameters=True)
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

    # optimizer
    parser.add_argument('--optim_lr', type=float, default=1e-3,
                        help='The init learning rate.')
    parser.add_argument('--warmup_epoch', type=int, default=3,
                        help='warmup epoch.')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='max epoch.')

    # GPU
    parser.add_argument("--local_rank", default=0, type=int,
                        help='master rank')

    # info
    parser.add_argument('--log_fre', type=int, default=25,
                        help='log show fre.')

    args = parser.parse_args()

    rand_seed(0)
    # ----------------------修改此处根据步骤2获得结构-------------------------------
    # args.arch_choose = [[3, [[1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)]]],
    #                     [4, [[1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)]]],
    #                     [6, [[1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)]]],
    #                     [3, [[1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)]]]]
    args.arch_choose = [[3, [[1.0, (1, 3)], [1.0, (1, 3)], [0.75, (1, 3)]]],
                        [4, [[1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)]]],
                        [6, [[1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)]]],
                        [3, [[1.0, (1, 3)], [1.0, (1, 3)], [1.0, (1, 3)]]]]
    # ----------------------------------------------------------------------------
    main(args)
