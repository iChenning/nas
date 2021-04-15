from torch.utils.data import Dataset
import os
from PIL import Image, ImageFile
import sys
import codecs
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())  # 用于print打印

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        assert os.path.exists(txt_path), "nonexistent:" + txt_path
        f = open(txt_path, 'r', encoding='utf-8')
        imgs = []
        labels = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0].encode('utf-8'), int(words[1])))
            labels.append(int(words[1]))
        f.close()
        self.n_classes = self.labels_check(labels)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        f_path, label = self.imgs[index]
        img = Image.open(f_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return (img, label, f_path, index)

    def __len__(self):
        return len(self.imgs)

    def labels_check(self, labels):
        """
        判断labels是否是连续的，从0开始的，并返回类别数
        """
        labels_set = set(labels)
        labels_continuous = set(range(len(labels_set)))
        labels_diff = labels_continuous - labels_set
        assert len(labels_diff) == 0, print(labels_diff, len(labels))
        return len(labels_set)


class DatasetExtend(Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        f_path, label = self.imgs[index]
        img = Image.open(f_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return (img, label, f_path, index)

    def __len__(self):
        return len(self.imgs)


class ConDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        assert os.path.exists(txt_path), "不存在" + txt_path
        f = open(txt_path, 'r', encoding='utf-8')
        imgs = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0].encode('utf-8'), int(words[1])))
        f.close()
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        f_path, label = self.imgs[index]
        img = Image.open(f_path).convert("RGB")

        assert self.transform is not None, 'contrastive loss must have transform!'
        img1 = self.transform(img)
        img2 = self.transform(img)

        return (img1, img2, label, f_path, index)

    def __len__(self):
        return len(self.imgs)


from torchvision import transforms
class CLSADataset(Dataset):
    def __init__(self, txt_path, easy_transform=None, strong_transform=None, num_res=5):
        assert os.path.exists(txt_path), "不存在" + txt_path
        f = open(txt_path, 'r', encoding='utf-8')
        imgs = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0].encode('utf-8'), int(words[1])))
        f.close()
        self.imgs = imgs
        self.easy_transform = easy_transform
        self.strong_transform = strong_transform

        resolutions = [96, 128, 160, 192, 224]
        self.res = resolutions[:num_res]
        self.resize_crop_ops = [transforms.RandomResizedCrop(res, scale=(0.5, 1.)) for res in self.res]
        self.num_res = num_res

    def __getitem__(self, index):
        f_path, label = self.imgs[index]
        img = Image.open(f_path).convert("RGB")

        assert self.easy_transform is not None, 'contrastive loss must have transform!'
        q = self.easy_transform(img)
        k = self.easy_transform(img)

        q_stronger_augs = []
        for resize_crop_op in self.resize_crop_ops:
            q_s = self.strong_transform(resize_crop_op(img))
            q_stronger_augs.append(q_s)

        return (q, k, q_stronger_augs, label, f_path)

    def __len__(self):
        return len(self.imgs)


class IterLoader():
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)