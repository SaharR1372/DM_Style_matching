#!/usr/bin/env python
"""Turn the downloaded Tiny ImageNet archive into the tensor file the loader expects.

Tiny ImageNet is not in torchvision, and reading 110k JPEGs through a Dataset every run is
slow enough to dominate a short condensation job.  This script does the decode once and
writes a single tensor file:

    <data_path>/tinyimagenet.pt
        classes       list[str]                      200 human-readable class names
        images_train  uint8 (100000, 3, 64, 64)
        labels_train  int64 (100000,)
        images_val    uint8 (10000, 3, 64, 64)
        labels_val    int64 (10000,)

Usage:

    # download and unpack first
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P data
    unzip -q data/tiny-imagenet-200.zip -d data

    python scripts/prepare_tinyimagenet.py --root data/tiny-imagenet-200 --out data/tinyimagenet.pt

Classes are ordered by their wnid as listed in wnids.txt, so labels are stable across runs.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image


def _load(path):
    with Image.open(path) as im:
        arr = np.array(im.convert('RGB'), dtype=np.uint8)   # (64, 64, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)           # (3, 64, 64)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='data/tiny-imagenet-200',
                   help='the unpacked tiny-imagenet-200 directory')
    p.add_argument('--out', default='data/tinyimagenet.pt')
    args = p.parse_args()

    with open(os.path.join(args.root, 'wnids.txt')) as f:
        wnids = [l.strip() for l in f if l.strip()]
    wnid_to_label = {w: i for i, w in enumerate(wnids)}

    # words.txt maps every wnid in WordNet to its name; keep the 200 we need.
    names = {}
    with open(os.path.join(args.root, 'words.txt')) as f:
        for line in f:
            wnid, _, name = line.strip().partition('\t')
            names[wnid] = name.split(',')[0]
    classes = [names.get(w, w) for w in wnids]

    train_imgs, train_labs = [], []
    for wnid in wnids:
        d = os.path.join(args.root, 'train', wnid, 'images')
        for fn in sorted(os.listdir(d)):
            train_imgs.append(_load(os.path.join(d, fn)))
            train_labs.append(wnid_to_label[wnid])
        print(f'\rtrain: {len(train_imgs):6d} images', end='', flush=True)
    print()

    val_imgs, val_labs = [], []
    ann = os.path.join(args.root, 'val', 'val_annotations.txt')
    with open(ann) as f:
        for line in f:
            fields = line.split('\t')
            val_imgs.append(_load(os.path.join(args.root, 'val', 'images', fields[0])))
            val_labs.append(wnid_to_label[fields[1]])
    print(f'val:   {len(val_imgs):6d} images')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'classes': classes,
                'images_train': torch.stack(train_imgs),
                'labels_train': torch.tensor(train_labs, dtype=torch.long),
                'images_val': torch.stack(val_imgs),
                'labels_val': torch.tensor(val_labs, dtype=torch.long)}, args.out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
