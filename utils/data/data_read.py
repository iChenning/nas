from utils.data.augment import augment, normal_transforms, stronger_transforms
from utils.data.dataset import MyDataset, DatasetExtend, ConDataset, CLSADataset
from torch.utils.data import DataLoader
import os
import random
from PIL import Image
import torch


def data_read(opt):
    train_trans, test_trans = augment(opt)

    trainloader = None
    if 'train' in opt.data.keys() and os.path.exists(opt.data.train.file_path):
        trainset = MyDataset(opt.data.train.file_path, transform=train_trans)
        trainloader = DataLoader(trainset, batch_size=opt.data.train.batch_size, shuffle=opt.data.train.shuffle,
                                 num_workers=8, pin_memory=True)

    validloader = None
    if 'valid' in opt.data.keys() and os.path.exists(opt.data.valid.file_path):
        validset = MyDataset(opt.data.valid.file_path, transform=test_trans)
        validloader = DataLoader(validset, batch_size=opt.data.valid.batch_size, shuffle=opt.data.valid.shuffle,
                                 num_workers=8, pin_memory=True)

    testloader = None
    if 'test' in opt.data.keys() and os.path.exists(opt.data.test.file_path):
        testset = MyDataset(opt.data.test.file_path, transform=test_trans)
        testloader = DataLoader(testset, batch_size=opt.data.test.batch_size, shuffle=opt.data.test.shuffle,
                                num_workers=8, pin_memory=True)

    return (trainloader, validloader, testloader)


def data_copy(opt, imgs, batch_size, shuffle=False):
    train_trans, _ = augment(opt)
    trainset = DatasetExtend(imgs, transform=train_trans)
    new_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    return new_loader


def fewshot_read(opt):
    assert os.path.exists(opt.data.test.file_path), "不存在" + opt.data.test.file_path
    f = open(opt.data.test.file_path, 'r')
    imgs_path = []
    imgs_label = []
    for line in f:
        line = line.rstrip()
        words = line.split()
        imgs_path.append((words[0], words[1]))
        imgs_label.append(words[1])
    f.close()
    imgs_label = list(set(imgs_label))
    return (imgs_path, imgs_label)


def imgs_load(opt, imgs_path, label_set):
    train_trans, test_trans = normal_transforms(opt)

    random.shuffle(label_set)
    sel_label = label_set[: opt.n_way]

    imgs_support = []
    labels_support = []
    imgs_query = []
    labels_query = []

    random.shuffle(imgs_path)
    k_sel = opt.k_support + opt.k_query
    sel_label_count = [0] * opt.n_way
    len_ = len(imgs_path)
    for i in range(len_):
        path = imgs_path[i][0]
        label = imgs_path[i][1]
        if label in sel_label:
            if sel_label_count[sel_label.index(label)] < opt.k_support:
                img = Image.open(path).convert('RGB')
                img = train_trans(img)
                imgs_support.append(img)
                labels_support.append(sel_label.index(label))
                sel_label_count[sel_label.index(label)] += 1
                # print('support', label, sel_label.index(label))
            elif sel_label_count[sel_label.index(label)] < k_sel:
                img = Image.open(path).convert('RGB')
                img = test_trans(img)
                imgs_query.append(img)
                labels_query.append(sel_label.index(label))
                sel_label_count[sel_label.index(label)] += 1
                # print('query', label, sel_label.index(label))

        if sum(sel_label_count) >= k_sel * opt.n_way:
            break

    tmp = imgs_support[0]
    imgs_sup_tensor = tmp.view(1, tmp.shape[0], tmp.shape[1], tmp.shape[2])
    for i in range(1, len(imgs_support)):
        tmp = imgs_support[i]
        tmp = tmp.view(1, tmp.shape[0], tmp.shape[1], tmp.shape[2])
        imgs_sup_tensor = torch.cat((imgs_sup_tensor, tmp), 0)

    tmp = imgs_query[0]
    imgs_que_tensor = tmp.view(1, tmp.shape[0], tmp.shape[1], tmp.shape[2])
    for i in range(1, len(imgs_query)):
        tmp = imgs_query[i]
        tmp = tmp.view(1, tmp.shape[0], tmp.shape[1], tmp.shape[2])
        imgs_que_tensor = torch.cat((imgs_que_tensor, tmp), 0)

    return (imgs_sup_tensor, labels_support, imgs_que_tensor, labels_query)


def contrastive_data_read(opt):
    train_trans, test_trans = augment(opt)

    trainloader = None
    if 'train' in opt.data.keys() and os.path.exists(opt.data.train.file_path):
        trainset = ConDataset(opt.data.train.file_path, transform=train_trans)
        trainloader = DataLoader(trainset, batch_size=opt.data.train.batch_size, shuffle=opt.data.train.shuffle,
                                 num_workers=8, pin_memory=True)

    validloader = None
    if 'valid' in opt.data.keys() and os.path.exists(opt.data.valid.file_path):
        validset = ConDataset(opt.data.valid.file_path, transform=test_trans)
        validloader = DataLoader(validset, batch_size=opt.data.valid.batch_size, shuffle=opt.data.valid.shuffle,
                                 num_workers=8, pin_memory=True)

    testloader = None
    if 'test' in opt.data.keys() and os.path.exists(opt.data.test.file_path):
        testset = ConDataset(opt.data.test.file_path, transform=test_trans)
        testloader = DataLoader(testset, batch_size=opt.data.test.batch_size, shuffle=opt.data.test.shuffle,
                                num_workers=8, pin_memory=True)

    return (trainloader, validloader, testloader)


def clsa_data_read(opt):
    train_trans, test_trans = augment(opt)

    trainset = CLSADataset(opt.data.train.file_path, easy_transform=train_trans, strong_transform=stronger_transforms())
    trainloader = DataLoader(trainset, batch_size=opt.data.train.batch_size, shuffle=opt.data.train.shuffle,
                             num_workers=8, pin_memory=True)

    if 'valid' in opt.data.keys() and os.path.exists(opt.data.valid.file_path):
        validset = ConDataset(opt.data.valid.file_path, transform=test_trans)
        validloader = DataLoader(validset, batch_size=opt.data.valid.batch_size, shuffle=opt.data.valid.shuffle,
                                 num_workers=8, pin_memory=True)
    else:
        validloader = None

    testset = CLSADataset(opt.data.test.file_path, easy_transform=test_trans, strong_transform=test_trans)
    testloader = DataLoader(testset, batch_size=opt.data.test.batch_size, shuffle=opt.data.test.shuffle, num_workers=8,
                            pin_memory=True)

    return (trainloader, validloader, testloader)
