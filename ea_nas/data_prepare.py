import numpy as np


def generate_list(imagedirs, train_txt, val_txt, train_frac=0.7, split=False):
    train_list = open(train_txt, 'w', encoding='utf8')
    val_list = open(val_txt, 'w', encoding='utf8')
    val_frac = 1. - train_frac
    imagedirs = imagedirs.encode('utf8')
    if not isinstance(imagedirs, list):
        imagedirs = [imagedirs]
    for n, imagedir in enumerate(imagedirs):
        labelIDs = os.listdir(imagedir)
        labelIDs = sorted(labelIDs)
        labelID_paths = [os.path.join(imagedir, x) for x in labelIDs]
        for m, labelID_path in enumerate(labelID_paths):
            image_paths = [os.path.join(labelID_path, x) for x in os.listdir(labelID_path)]
            train_files = int(np.ceil(train_frac * len(image_paths)))
            val_files = len(image_paths) - train_files
            # val_files = int(val_frac * len(image_paths))
            for i, image_path in enumerate(image_paths):
                if i + 1 <= train_files:
                    train_list.write(' '.join([image_path.decode('utf8'), str(m), str(n)]) + '\n')
                else:
                    val_list.write(' '.join([image_path.decode('utf8'), str(m), str(n)]) + '\n')
                # elif i+1 > train_files and i+1 <= train_files + val_files:
                #     val_list.write(' '.join([image_path.decode('utf8'), str(m), str(n)]) + '\n')
                # else:
                #     break
                print('\r[%d/%d]' % (i, m), end='', flush=True)
        print('\n')
    train_list.close()
    val_list.close()


import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--imagedir", '-i', type=str, help="imagedir", required=True)
parser.add_argument("--train_frac", '-tf', type=float, help="train frac", default=0.8)
parser.add_argument("--image_name", '-in', type=str, help="image_name", required=True)
args = parser.parse_args()
imagedirs = os.path.abspath(args.imagedir)
train_txt = os.path.join('/code', '%s_train_list.txt' % args.image_name)
val_txt = os.path.join('/code', '%s_val_list.txt' % args.image_name)
if args.train_frac == 1.:
    val_txt = os.path.join('/code', 'bad.txt')

generate_list(imagedirs, train_txt, val_txt, args.train_frac)