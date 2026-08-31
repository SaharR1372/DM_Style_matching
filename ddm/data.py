"""Datasets.

Loaders for every dataset the method is evaluated on, returning the tensors and the
normalisation statistics the rest of the pipeline expects.  ``get_dataset`` is the single
place a new dataset has to be registered.
"""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_img(path):
    """Load one image file as RGB."""
    return Image.open(path).convert('RGB')


class ImageNetDataset(Dataset):
    """ImageNet-1K read from a CSV manifest of (image_id, label) pairs.

    ``root`` is expected to hold ``imagenet_train.csv``, ``imagenet_val.csv``,
    ``class_names.csv`` and the image files the manifests point at:

        <data_path>/imagenet/imagenet_train.csv
        <data_path>/imagenet/imagenet_val.csv
        <data_path>/imagenet/class_names.csv
        <data_path>/imagenet/<image_id as written in the manifest>

    Image ids are stored with a leading separator in the released manifests, so they are
    joined relative to ``root`` after stripping it.
    """

    def __init__(self, root, part='train', im_size=(128, 128),
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.root = root
        self.part = part
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size)])

        import pandas as pd   # only ImageNet needs it, so it is not a hard dependency

        csv_name = 'imagenet_train.csv' if part == 'train' else 'imagenet_val.csv'
        manifest = pd.read_csv(os.path.join(root, csv_name))
        self.images = [str(p).lstrip('/') for p in manifest['image_id']]
        self.labels = [int(v) for v in manifest['label']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.transforms(get_img(os.path.join(self.root, self.images[index])))
        return image, torch.tensor(self.labels[index], dtype=torch.long)


def get_dataset(dataset, data_path):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        dst_train.nclass = 10

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        dst_train.nclass = 100

    elif dataset == 'ImageNet':
        # ImageNet-1K, read from the CSV manifests under <data_path>/imagenet.
        # See ImageNetDataset for the expected layout.
        root = os.path.join(data_path, 'imagenet')
        channel = 3
        im_size = (128, 128)
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        with open(os.path.join(root, 'class_names.csv')) as f:
            class_names = [line.strip() for line in f]
        dst_train = ImageNetDataset(root, part='train', im_size=im_size, mean=mean, std=std)
        dst_test = ImageNetDataset(root, part='val', im_size=im_size, mean=mean, std=std)

    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')
        class_names = data['classes']
        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation
        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation

    else:
        exit('unknown dataset: %s'%dataset)

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
