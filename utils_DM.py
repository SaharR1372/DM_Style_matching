import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
try:
    from scipy.ndimage import rotate as scipyrotate
except ImportError:  # scipy < 1.10
    from scipy.ndimage.interpolation import rotate as scipyrotate
from networks_DM import MLP, ConvNet, ConvNet_style, LeNet, AlexNet, AlexNet_style, VGG11, VGG11_style, VGG11BN, ResNet18, ResNet18_style, ResNet18BN

from PIL import Image

class Config:
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

    # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    dict = {
        "imagenette" : imagenette,
        "imagewoof" : imagewoof,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagemeow": imagemeow,
        "imagesquawk": imagesquawk,
    }

config = Config()
def get_img(path):
    # im_bgr = cv2.imread(path)
    # im_rgb = im_bgr[:, :, ::-1]
    # return im_rgb
    img = Image.open(path).convert('RGB')
    return img

import pickle as pkl
import pandas as pd
class ImageNetDataset(Dataset):
    def __init__(self, part='train'):
        self.part = part

        im_size = (128, 128)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transforms= transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std),
                                        transforms.Resize(im_size),
                                        transforms.CenterCrop(im_size)])
        self.images = []
        self.labels = []
        self.labels = []
        if part == 'train':
            mycsv = pd.read_csv('./imagenet_train_val_csv/imagenet_train.csv')
        else:
            mycsv = pd.read_csv('./imagenet_train_val_csv/imagenet_val.csv')
        for i in range(len(mycsv['image_id'])):
            self.images.append(mycsv['image_id'][i][1:])
            self.labels.append(int(mycsv['label'][i]))
        unique_labels = mycsv['label'].nunique()
        print(f"Number of unique labels: {unique_labels}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path1=os.path.join('/home/exx/PycharmProj/ImbalancedLearning/', self.images[index])
        image = get_img(path1)
        if self.transforms is not None:
            image = self.transforms(image)
            # image = self.transforms(image=image)['image']
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return image, label



# def get_dataset(dataset, data_path, batch_size=1, subset="imagenette", args=None):
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
        # Path to your JSON file with ImageNet class names
        # Path to your CSV file with ImageNet class names
        class_file = '/home/exx/PycharmProj/IDM/imagenet_train_val_csv/class_names.csv'
        num_classes=1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Load class names
        with open(class_file) as f:
            class_names = [line.strip() for line in f.readlines()]

        channel = 3
        im_size = (128, 128)
        dst_train = ImageNetDataset(part='train')
        dst_test =ImageNetDataset(part='val')


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



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes, im_size=(32, 32), net_depth=None):
    """Build an architecture by name.

    net_depth overrides the default ConvNet depth (3).  The DM literature evaluates
    64x64 datasets such as TinyImageNet on ConvNetD4, so a caller working at that
    resolution should pass net_depth=4; leaving it None reproduces the previous
    behaviour exactly at every call site.
    """
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, default_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
    net_depth = default_depth if net_depth is None else int(net_depth)

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNet_style':
        # This case was missing, so `--model ConvNet_style` -- the model every style
        # matching command in the README asks for -- exited with 'unknown model'.
        net = ConvNet_style(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                            net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet_style':
        net = AlexNet_style(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11_style':
        net = VGG11_style( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18_style':
        net = ResNet18_style(channel=channel, num_classes=num_classes)  # was ResNet18_gram, undefined
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwish':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwishBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetBN_gram':
        net = ConvNet_gram(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none', im_size=im_size)
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling', im_size=im_size)
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling', im_size=im_size)

    else:
        net = None
        exit('unknown model: %s'%model)

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))



def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis



def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop



def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    # args.device='cuda:1'
    net = net.to(args.device)
    # net = net.to(device)
    # criterion = criterion.to(device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = True)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test



def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        if 'BN' in model:
            print('Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    theta=theta.clone()
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    theta=theta.clone()
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    # randf=randf.clone()
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    # randb=randb.clone()
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0].clone()
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    # rands=rands.clone()
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0].clone()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    # randc=randc.clone()
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0].clone()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    # translation_x=translation_x.clone()
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    # translation_y=translation_y.clone
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    # offset_x= offset_x.clone()
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    # offset_y=offset_y.clone()
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}



def calc_mean_std(feat, eps=1e-5):
    """
    Calculates the mean and standard deviation of the feature maps.

    Args:
    - feat (torch.Tensor): Input feature map tensor of shape (N, C, H, W).
    - eps (float): A small value added to the variance to avoid divide-by-zero.

    Returns:
    - feat_mean (torch.Tensor): The mean of the feature maps, reshaped to (N, C, 1, 1).
    - feat_std (torch.Tensor): The standard deviation of the feature maps, reshaped to (N, C, 1, 1).
    """
    size = feat.size()
    assert (len(size) == 4)  # Ensure the input tensor has 4 dimensions (batch, channels, height, width)
    N, C = size[:2]  # Extract batch size (N) and number of channels (C)

    # Calculate the variance for each feature map and add a small epsilon to avoid division by zero
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    # Calculate the standard deviation from the variance
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    # Calculate the mean for each feature map
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    return feat_mean, feat_std  # Return the mean and standard deviation, reshaped for broadcasting



def calc_style_loss(input, target):
    """
    Calculates the style loss between the input and target feature maps based on their statistics.

    Args:
    - input (torch.Tensor): Input feature map tensor of shape (N, C, H, W).
    - target (torch.Tensor): Target feature map tensor of the same shape as input.
      The target tensor should not require gradients.

    Returns:
    - torch.Tensor: The computed style loss.
    """
    assert (input.size() == target.size())  # Ensure input and target have the same shape
    assert (target.requires_grad is False)  # Ensure the target does not require gradients

    # Calculate the mean and standard deviation of the input and target feature maps
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)

    # Compute the style loss as the MSE between the mean and standard deviations of input and target
    #return nn.MSELoss()(input_std, target_std)
    return (nn.MSELoss()(input_mean, target_mean) + nn.MSELoss()(input_std, target_std))/2

def gram_matrix(x, should_normalize=True):
    """
    Computes the Gram matrix, capturing correlations between channels for each spatial location.

    Args:
    - x (torch.Tensor): Input tensor of shape (batch, channels, height, width).
    - should_normalize (bool): Whether to normalize the Gram matrix by the number of elements.

    Returns:
    - torch.Tensor: The Gram matrix of the input tensor.
    """
    (b, ch, h, w) = x.size()  # Unpack dimensions: batch size, channels, height, width
    features = x.view(b, ch, w * h)  # Reshape to (batch, channels, width * height)
    features_t = features.transpose(1, 2)  # Transpose to (batch, width * height, channels)
    gram = features.bmm(features_t)  # Batch matrix multiplication to compute Gram matrix

    if should_normalize:
        gram /= ch * h * w  # Normalize by the number of elements

    return gram  # Output size: (batch, channels, channels), capturing channel correlations


# ---------------------------------------------------------------------------
# Intra-Class Diversity (ICD).
#
# The ICD module enhances diversity within each condensed class.  Two formulations
# live here:
#
#   intra_class_diversity_loss      the released form -- BOUNDED and target-matched,
#                                   composed of a style and a content component (below).
#   intra_class_diversity_loss_kl   the Eq. 8-9 form -- unbounded KL repulsion, kept so
#                                   the published objective can still be reproduced.
#
# See `intra_class_diversity_loss` for why the released form is the one to use.
# ---------------------------------------------------------------------------

def icd_k_for_ipc(ipc):
    """k = 0.2 x IPC nearest intra-class neighbours, at least 1 and at most ipc-1."""
    if ipc < 2:
        return 0
    return int(max(1, min(ipc - 1, round(0.2 * ipc))))


def intra_class_diversity_loss_kl(feat, k=None, ipc=None, eps=1e-8):
    """Eq. 8-9 of the paper: maximise KL divergence to the k nearest intra-class neighbours.

    For every synthetic sample x~ of the class we take the mean embedding m of its k
    nearest intra-class neighbours and *maximise* KL( S(phi(x~)) || S(m) ), which is
    returned here with a negative sign so it can be added to a minimised objective.

    SUPERSEDED by `intra_class_diversity_loss`, and retained only so the published
    objective can be reproduced exactly.  This form maximises a divergence, so it is
    unbounded below and has no attainable optimum: the descent direction never
    terminates and past a moderate weight the term simply overwhelms the content
    matching, driving the intra-class scatter far past anything present in the real
    data.  Prefer the released bounded form for any new work.

    Args:
        feat: (n, d) embeddings of one class's synthetic samples.
        k:    number of neighbours; if None it is derived from ``ipc`` (or n) as 0.2*IPC.
        ipc:  images per class, used only to derive k.

    Returns:
        Scalar tensor, 0 when the class holds fewer than two samples.
    """
    n = feat.shape[0]
    if k is None:
        k = icd_k_for_ipc(ipc if ipc is not None else n)
    k = int(min(max(k, 0), n - 1)) if n > 1 else 0
    if k < 1:
        return feat.sum() * 0.0

    # k nearest intra-class neighbours by squared L2 in feature space (Eq. 9).
    with torch.no_grad():
        d2 = torch.cdist(feat, feat, p=2) ** 2
        d2.fill_diagonal_(float('inf'))
        nn_idx = torch.topk(d2, k, dim=1, largest=False).indices  # (n, k)

    m = feat[nn_idx].mean(dim=1)                      # (n, d) neighbourhood centroid
    log_p = F.log_softmax(feat, dim=1)
    log_q = F.log_softmax(m, dim=1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=1)   # KL( S(phi(x~)) || S(m) )
    return -kl.sum()


# ---------------------------------------------------------------------------
# Style statistics.
# ---------------------------------------------------------------------------

def style_vector(feat, eps=1e-5):
    """Per-sample channel-wise style descriptor of a feature map.

    Args:
        feat: (N, C, H, W)
    Returns:
        mu, sd: (N, C) each -- the spatial mean and std of every channel of every sample.
    """
    n, c = feat.shape[:2]
    x = feat.reshape(n, c, -1)
    mu = x.mean(dim=2)
    sd = (x.var(dim=2, unbiased=False) + eps).sqrt()
    return mu, sd


def _sq_diff(a, b, relative, eps=1e-8):
    """Mean squared difference, optionally divided by the magnitude of the target.

    The relative form makes the term invariant to the scale of the feature maps, which
    matters once the style is read before normalisation: there the magnitudes depend on
    the random initialisation and grow with depth, so an absolute loss silently weights
    the layers by their activation scale and its coefficient has to be retuned for every
    tap, architecture and dataset.
    """
    num = (a - b).pow(2).mean()
    if not relative:
        return num
    return num / (b.detach().pow(2).mean() + eps)


def moments_matching_loss(feat_syn, feat_real, mode='persample', relative=False, eps=1e-5):
    """First/second-moment style matching between a synthetic and a real feature map.

    mode='batchavg' reproduces the published implementation: the feature maps are
        averaged over the batch first and the spatial mean/std of that average are
        compared.  Because the spatial variance of a batch-average shrinks with the
        batch size, the target computed from ``batch_real`` real images is not on the
        same scale as the value computed from ``ipc`` synthetic images.
    mode='persample' compares E_x[mu(x)] and E_x[sd(x)] instead, which are sample means
        of a per-sample quantity and therefore unbiased with respect to the batch size.
    """
    if mode == 'batchavg':
        s = torch.mean(feat_syn, dim=0, keepdim=True)
        r = torch.mean(feat_real, dim=0, keepdim=True)
        s_mu, s_sd = calc_mean_std(s, eps)
        r_mu, r_sd = calc_mean_std(r, eps)
        return (_sq_diff(s_mu, r_mu, relative) + _sq_diff(s_sd, r_sd, relative)) / 2

    s_mu, s_sd = style_vector(feat_syn, eps)
    r_mu, r_sd = style_vector(feat_real, eps)
    return (_sq_diff(s_mu.mean(0), r_mu.mean(0), relative)
            + _sq_diff(s_sd.mean(0), r_sd.mean(0), relative)) / 2


def style_diversity_loss(feat_syn, feat_real, relative=False, eps=1e-5):
    """Match the *spread* of the per-sample style descriptors (the new SD term).

    L_MMD matches where the content sits, L_MM/L_CM match where the style sits and
    L_ICD spreads the content out.  The remaining cell of that 2x2 is the spread of the
    style, which nothing in the published objective supervises: within a class the
    condensed samples end up sharing one style while the real samples span a range of
    them.  Here that spread is *matched* to the real one rather than maximised, so the
    real data sets the target and no repulsion strength has to be tuned.

    Both sides use the unbiased (n-1) estimator so that ``ipc`` synthetic samples and
    ``batch_real`` real samples give comparable values, and the comparison is made in
    std space so this term shares the scale of ``moments_matching_loss``.
    """
    if feat_syn.shape[0] < 2 or feat_real.shape[0] < 2:
        return feat_syn.sum() * 0.0
    s_mu, s_sd = style_vector(feat_syn, eps)
    r_mu, r_sd = style_vector(feat_real, eps)
    # across-sample std of each style coordinate, per channel
    s_v_mu = (s_mu.var(dim=0, unbiased=True) + eps).sqrt()
    r_v_mu = (r_mu.var(dim=0, unbiased=True) + eps).sqrt()
    s_v_sd = (s_sd.var(dim=0, unbiased=True) + eps).sqrt()
    r_v_sd = (r_sd.var(dim=0, unbiased=True) + eps).sqrt()
    return (_sq_diff(s_v_mu, r_v_mu, relative) + _sq_diff(s_v_sd, r_v_sd, relative)) / 2


def correlation_matching_loss(feat_syn, feat_real):
    """Gram-matrix (channel correlation) matching, L_CM of the paper."""
    g_s = gram_matrix(feat_syn).mean(dim=0)
    g_r = gram_matrix(feat_real).mean(dim=0)
    return nn.MSELoss(reduction='sum')(g_s, g_r)


# ---------------------------------------------------------------------------
# Components of the released L_ICD, plus two screened-and-rejected candidates.
#
# L_ICD maximises a KL divergence, so it has no finite optimum: past some beta it simply
# overwhelms the content loss (measured in FINDINGS.md -- it saturates near -25 while the
# content loss stalls).  Both losses below apply the same "spread the class out" pressure
# but are bounded and self-calibrating, so the real data decides how much spread is right.
# ---------------------------------------------------------------------------

def intra_class_coverage_loss(feat_syn, feat_real, temp=0.1, eps=1e-8):
    """Normalised quantisation error of the real class by the synthetic samples (L_ICC).

    Treats the ipc synthetic samples of a class as a codebook and measures how well they
    cover the real class distribution:

        L_ICC = E_x[ min_j || phi(x) - phi(x~_j) ||^2 ]  /  E_x[ || phi(x) - mu ||^2 ]

    The denominator is the total intra-class variance of the real data, so the ratio is
    dimensionless and reads directly as a fraction of unexplained variance:

      * every synthetic sample collapsed onto the class mean  ->  L_ICC = 1 (the worst case,
        and exactly the degenerate solution that L_MMD alone permits);
      * synthetic samples sitting at the k-means centroids of the class  ->  the normalised
        k-means distortion, well below 1.

    So unlike L_ICD it is bounded, it has a finite optimum, and it needs no tuning to decide
    how far apart the samples should be -- covering the class is the objective, and spread is
    what covering requires.  Minimising it is one Lloyd step per iteration: each synthetic
    sample is pulled towards the centroid of the real samples assigned to it.

    ---------------------------------------------------------------------------------------
    MEASURED AND REJECTED -- kept for the record, do not use as a diversity term.
    diagnostics/diag_icc_props.py shows this loss *increases* from 1.098 at full collapse to
    1.579 at the spread of a real ipc-sized subset, i.e. its minimum is at collapse and
    optimising it would cause the very failure it was meant to fix.  The reason is
    dimensional: with ipc=10 codebook points in a 2048-dimensional embedding, squared
    quantisation error is minimised by placing every point at the class centroid, and spread
    out points sit in the shell of the distribution where they score worse.  Coverage by
    quantisation is the wrong tool at this ipc; use ``content_diversity_loss`` instead, which
    does have its minimum at the real spread.
    ---------------------------------------------------------------------------------------

    Assignment responsibilities are computed without gradient (an EM / Lloyd update).  A soft
    assignment is used rather than a hard argmin so that a synthetic sample which is currently
    nearest to no real sample still receives gradient instead of going dead; ``temp`` is
    relative to the mean squared distance, and temp -> 0 recovers hard k-means.

    Args:
        feat_syn:  (n_s, d) synthetic embeddings of one class.
        feat_real: (n_r, d) real embeddings of the same class (no gradient expected).
    """
    if feat_syn.shape[0] < 1 or feat_real.shape[0] < 2:
        return feat_syn.sum() * 0.0

    d2 = torch.cdist(feat_real, feat_syn, p=2) ** 2          # (n_r, n_s)
    with torch.no_grad():
        if temp <= 0:
            resp = torch.zeros_like(d2).scatter_(1, d2.argmin(dim=1, keepdim=True), 1.0)
        else:
            tau = temp * d2.mean().clamp_min(eps)
            resp = torch.softmax(-d2 / tau, dim=1)
    distortion = (resp * d2).sum(dim=1).mean()

    with torch.no_grad():
        spread = (feat_real - feat_real.mean(dim=0, keepdim=True)).pow(2).sum(dim=1).mean()
    return distortion / (spread + eps)


def between_class_loss(mu_syn, mu_real, relative=True, eps=1e-8):
    """Match the *between-class* geometry of the condensed set to the real data.

    Every other term in this codebase -- L_MMD, L_MM, L_CM, L_SD, L_CD, L_ICD -- is computed
    inside a single class.  DM's objective is a sum of independent per-class mean-matching
    problems and never constrains how the classes sit relative to one another.  That is the
    one structural axis the whole sweep left untouched, which is why it is worth one test.

    diagnostics/diag_floor.py showed the per-class mean matching is essentially finished (the
    condensed set matches its class mean 6.5x better than ten real images do), but "finished"
    is per class: a residual of ~43 summed over classes still leaves the *arrangement* of the
    class means free to drift, and a classifier cares about exactly that arrangement.

    Matches the matrix of pairwise squared distances between class-mean embeddings, normalised
    by the real matrix's magnitude so the weight is scale-free -- the same construction as
    L_SD and L_CD, and for the same reason: the target is read off the real data, so the
    optimum is attainable and interior rather than a direction of unbounded descent.

    mu_syn / mu_real: (K, D) class-mean embeddings, K classes.  mu_real is expected detached.
    """
    if mu_syn.shape[0] < 2:
        return mu_syn.sum() * 0.0
    d_s = torch.cdist(mu_syn, mu_syn, p=2) ** 2
    d_r = torch.cdist(mu_real, mu_real, p=2) ** 2
    num = (d_s - d_r).pow(2).mean()
    if not relative:
        return num
    return num / (d_r.detach().pow(2).mean() + eps)


def content_diversity_loss(feat_syn, feat_real, rank=0, eps=1e-8):
    """Match the intra-class spread of the content along the real class's principal axes.

    The content analogue of ``style_diversity_loss``: where L_ICD pushes samples apart
    without a target, this matches how far apart they should be, and does so per direction
    rather than isotropically.

        L_CD = mean_k ( std(S v_k) - std(R v_k) )^2  /  mean_k std(R v_k)^2

    with v_k the top-``rank`` principal directions of the real class, computed from the real
    batch without gradient.  Matching a full covariance is impossible here -- ipc synthetic
    samples span at most ipc-1 dimensions of a 2048-dimensional embedding -- so the rank is
    capped at ipc-1, which is exactly the number of variances the synthetic set has the
    degrees of freedom to set.

    Args:
        rank: number of principal directions; 0 selects min(n_s - 1, 16).
    """
    n_s, n_r = feat_syn.shape[0], feat_real.shape[0]
    if n_s < 2 or n_r < 2:
        return feat_syn.sum() * 0.0
    r = rank if rank > 0 else min(n_s - 1, 16)
    r = int(min(r, n_s - 1, n_r - 1, feat_real.shape[1]))
    if r < 1:
        return feat_syn.sum() * 0.0

    with torch.no_grad():
        # top-r principal directions of the real class (real data only, so no gradient)
        _, _, v = torch.pca_lowrank(feat_real, q=min(r + 6, n_r, feat_real.shape[1]), niter=2)
        v = v[:, :r]                                          # (d, r)
        r_std = (feat_real - feat_real.mean(0, keepdim=True)).mm(v).std(dim=0, unbiased=True)

    s_std = (feat_syn - feat_syn.mean(0, keepdim=True)).mm(v).std(dim=0, unbiased=True)
    return (s_std - r_std).pow(2).mean() / (r_std.pow(2).mean() + eps)


# ---------------------------------------------------------------------------
# The released Intra-Class Diversity loss.
# ---------------------------------------------------------------------------

def intra_class_diversity_loss(emb_syn, emb_real, feat_syn=None, feat_real=None,
                               content_ratio=1.0, style_ratio=0.0, rank=0,
                               relative=True, eps=1e-8, return_parts=False):
    """L_ICD -- intra-class diversity, in a bounded and target-matched form.

    The ICD module of the paper enhances diversity within each condensed class, so that
    the ipc synthetic images of a class span the class the way real images of that class
    do instead of collapsing onto a single prototype.  This is the released
    implementation of that module, and it is built from two components:

        L_ICD = content_ratio * L_CD  +  style_ratio * L_SD

      L_CD  ``content_diversity_loss``  -- matches the intra-class spread of the final
            embedding along the principal directions of the real class.  This is the
            content axis, and is the direct counterpart of the module as described in
            the paper.
      L_SD  ``style_diversity_loss``    -- matches the across-sample spread of the
            per-sample style descriptors of the intermediate feature maps.  This is the
            style axis, and is off by default (see the note on redundancy below).

    Why this replaces the KL-repulsion formulation
    ----------------------------------------------
    `intra_class_diversity_loss_kl` implements Eq. 8-9 by *maximising* a divergence
    between each sample and its k nearest intra-class neighbours.  Maximising an
    unbounded quantity gives the term no attainable optimum: its descent direction never
    terminates, so there is no weight at which it both spreads the samples and stops.  In
    practice it keeps pushing until it dominates the content matching and disperses the
    class well beyond the spread of the real data.

    Both components here are built the opposite way.  Each compares a synthetic statistic
    against the *same statistic measured on the real batch*, so the loss is bounded below
    by zero, is minimised exactly where the condensed class has the spread of the real
    class, and rises again if the class is pushed wider than the data.  The target is read
    off the data rather than set by a coefficient, and because both are normalised by the
    magnitude of their own target they are scale-free: one weight transfers across taps,
    architectures, resolutions and datasets without retuning.

    On the two components
    ---------------------
    The two axes are largely redundant -- both constrain second-order intra-class
    structure, so they compete for the same headroom rather than adding.  The default
    therefore activates the content component alone, which is the axis the paper's module
    describes.  Set ``style_ratio`` to enable the style component; it is a viable
    alternative to the content one, not an addition to it.

    Measured scope.  Against a style-matching control, the content component is neutral at
    every resolution tested (-0.08 on CIFAR10, -0.13 on CIFAR100, -0.14 on TinyImageNet, all
    inside the error bars).  The style component is harmless at 32x32 but costs 1.65 points
    at 64x64, where it falls below the plain distribution-matching baseline; enabling it is
    not recommended above 32x32.

    Args:
        emb_syn:   (n_s, d) embeddings of one class's synthetic samples.
        emb_real:  (n_r, d) embeddings of the same class's real samples (detached).
        feat_syn:  optional sequence of (n_s, C, H, W) style feature maps, required only
                   when ``style_ratio`` is non-zero.
        feat_real: the matching real feature maps.
        content_ratio / style_ratio: weights of the two components.
        rank:      principal directions matched by L_CD; 0 selects min(ipc - 1, 16).
        relative:  normalise the style component by the magnitude of its target.
        return_parts: also return the unweighted value of each component, for logging.

    Returns:
        Scalar tensor (the weighted sum), 0 when the class holds fewer than two synthetic
        samples -- a single sample has no across-sample spread, so no diversity term can
        act at ipc = 1.  With return_parts=True, returns (loss, {'content':.., 'style':..})
        where the dict holds the unweighted component values as floats.
    """
    loss = emb_syn.sum() * 0.0
    parts = {'content': 0.0, 'style': 0.0}
    if content_ratio:
        l_cd = content_diversity_loss(emb_syn, emb_real, rank=rank, eps=eps)
        loss = loss + content_ratio * l_cd
        parts['content'] = float(l_cd)
    if style_ratio and feat_syn is not None and feat_real is not None:
        sd = [style_diversity_loss(a, b, relative=relative) for a, b in zip(feat_syn, feat_real)]
        if sd:
            l_sd = sum(sd) / len(sd)
            loss = loss + style_ratio * l_sd
            parts['style'] = float(l_sd)
    return (loss, parts) if return_parts else loss
