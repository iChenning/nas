def cifar100():
    return (
        '/opt/code/Data/cifar100/train.txt',
        '/opt/code/Data/cifar100/test.txt'
    )


def cifar100_gz():
    return (
        '/data/cifar100/train.txt',
        '/data/cifar100/test.txt'
    )


def cifar10():
    return (
        '/opt/code/Data/cifar10/train.txt',
        '/opt/code/Data/cifar10/test.txt'
    )


def svhn10():
    return (
        '/opt/code/Data/svhn10/my_path/train.txt',
        '/opt/code/Data/svhn10/my_path/test.txt'
    )


def cars196():
    return (
        '/opt/code/Data/cars196/train.txt',
        '/opt/code/Data/cars196/test.txt'
    )


def cub200():
    return (
        '/opt/code/Data/cub200/my_path/train.txt',
        '/opt/code/Data/cub200/my_path/test.txt'
    )


def pets37():
    return (
        '/opt/code/Data/pets37/my_path/train.txt',
        '/opt/code/Data/pets37/my_path/test.txt'
    )


def flowers102():
    return (
        '/opt/code/Data/flowers102/my_path/train.txt',
        '/opt/code/Data/flowers102/my_path/test.txt'
    )


def food101():
    return (
        '/opt/code/Data/food101/train.txt',
        '/opt/code/Data/food101/test.txt'
    )


def image3403():
    return (
        # '/home/code/train/image_list/3403_train_list.txt',
        '/home/code/pretrained/image_list/3403_train_list.txt',
        # '/home/code/train/image_list/3403_val_list.txt'
        '/home/code/pretrained/image_list/3403_val_list.txt'
    )


def image3410():
    return (
        '/home/code/train/image_list/3410_train_list.txt',
        '/home/code/train/image_list/3410_val_list.txt'
    )


default_dir = {
    'cifar100': cifar100(),
    'cifar100_gz': cifar100_gz(),
    'cifar10': cifar10(),
    'svhn10': svhn10(),
    'cars196': cars196(),
    'cub200': cub200(),
    'pets37': pets37(),
    'flowers102': flowers102(),
    'food101': food101(),
    'image3403': image3403(),
    'image3410': image3410()
}
