import numpy as np
from PIL import ImageFilter
import random
import torch
from torchvision import transforms


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def normal_trans(img_size):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    return (train_transforms, test_transforms)


def hard_trans(img_size, n_holes=None, length=None):
    """
    hard增加随机高斯模糊和cutout
    """
    holes = 1 if n_holes is None else n_holes
    le = int(img_size / 3) if length is None else int(length)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=holes, length=le)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    return (train_transforms, test_transforms)


#  针对cifar10、cifar100与food101、cars196采用不同的数据增强，主要是因为尺寸问题导致的
def normal_transforms(opt):
    if opt.data_name in ('cifar10', 'cifar100', 'fc100', 'svhn10'):
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((opt.data.data_size, opt.data.data_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((opt.data.data_size, opt.data.data_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        re_size = int(opt.data.data_size * 1.25)
        train_transforms = transforms.Compose([
            transforms.Resize((re_size, re_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(opt.data.data_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((opt.data.data_size, opt.data.data_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    return (train_transforms, test_transforms)


def hard_transforms(opt):  # 暂时未修改
    """
    hard增加随机高斯模糊和cutout
    """
    if opt.data_name in ('cifar10', 'cifar100', 'fc100', 'svhn10'):
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((opt.data.data_size, opt.data.data_size)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(n_holes=1, length=int(opt.data.data_size / 3))
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((opt.data.data_size, opt.data.data_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        re_size = int(opt.data.data_size * 1.25)
        train_transforms = transforms.Compose([
            transforms.Resize((re_size, re_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(opt.data.data_size),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(n_holes=1, length=int(opt.data.data_size / 3))
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((opt.data.data_size, opt.data.data_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    return (train_transforms, test_transforms)


from utils.data.augment_clsa import CLSAAug
def stronger_transforms():
    stronger_aug = CLSAAug(num_of_times=5)  # num of repetive times for randaug
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    stronger_aug = transforms.Compose([stronger_aug, transforms.ToTensor(), normalize])

    return stronger_aug


def augment(opt):
    if opt.data.augment_level == 'normal':
        return normal_transforms(opt)
    elif opt.data.augment_level == 'hard':
        return hard_transforms(opt)
    else:
        assert False, "augment_level must belong to {'normal' 'hard'}"