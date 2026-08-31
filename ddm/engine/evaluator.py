"""The evaluation protocol.

Every method in this repository -- distribution matching and coreset selection alike --
is scored the same way: train a freshly initialised network from scratch on the small set,
test it on the real test set, and repeat ``eval.num_eval`` times per architecture.  Keeping
that protocol in one place is what makes the numbers comparable across methods, so both
train.py and evaluate.py route through ``evaluate_set``.
"""
import logging
import time

import numpy as np
import torch
import torch.nn as nn

from ddm.augment import DiffAugment, ParamDiffAug, augment
from ddm.data import TensorDataset
from ddm.models import get_network
from ddm.utils import get_time


def run_epoch(mode, loader, net, optimizer, criterion, device,
              dsa_strategy=None, dsa_param=None, dc_aug_param=None):
    """One pass over ``loader``; returns (mean loss, accuracy).

    Augmentation is applied only when a strategy is given, which is how the training pass
    is distinguished from the test pass.
    """
    loss_avg, acc_avg, num_exp = 0.0, 0.0, 0
    net = net.to(device)
    criterion = criterion.to(device)
    net.train() if mode == 'train' else net.eval()

    for datum in loader:
        img = datum[0].float().to(device)
        if dsa_strategy:
            img = DiffAugment(img, dsa_strategy, param=dsa_param)
        elif dc_aug_param is not None:
            img = augment(img, dc_aug_param, device=device)
        lab = datum[1].long().to(device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1),
                              lab.cpu().data.numpy()))

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_avg / num_exp, acc_avg / num_exp


def train_and_test(it_eval, net, images_train, labels_train, testloader, device, *,
                   epochs, lr, batch_size, dsa_strategy=None, dsa_param=None):
    """Train ``net`` from scratch on the small set and report its test accuracy.

    SGD with momentum 0.9 and weight decay 5e-4, the learning rate divided by ten halfway
    through -- the protocol used throughout the distribution-matching literature, kept
    unchanged so the numbers here sit on the same scale as published ones.
    """
    net = net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)
    lr = float(lr)
    epochs = int(epochs)
    lr_schedule = [epochs // 2 + 1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(device)

    trainloader = torch.utils.data.DataLoader(
        TensorDataset(images_train, labels_train), batch_size=batch_size, shuffle=True,
        num_workers=0)

    start = time.time()
    loss_train = acc_train = 0.0
    for ep in range(epochs + 1):
        loss_train, acc_train = run_epoch('train', trainloader, net, optimizer, criterion,
                                          device, dsa_strategy=dsa_strategy,
                                          dsa_param=dsa_param)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)

    _, acc_test = run_epoch('test', testloader, net, optimizer, criterion, device)
    logging.info('%s eval %02d: epochs=%d train_time=%ds train_loss=%.6f train_acc=%.4f '
                 'test_acc=%.4f', get_time(), it_eval, epochs, int(time.time() - start),
                 loss_train, acc_train, acc_test)
    return net, acc_train, acc_test


def evaluate_set(images, labels, cfg, channel, num_classes, im_size, testloader, device,
                 label=''):
    """Score one small set under the protocol in ``cfg.eval``.

    Args:
        images, labels: the condensed or selected set.
        cfg: the full config; ``cfg.eval`` supplies the protocol and ``cfg.model.net_depth``
            the ConvNet depth to evaluate at.
        label: a string prefix for the log lines, e.g. 'it=4000'.

    Returns:
        ``{arch: [accuracy, ...]}`` -- the raw list, so a caller can pool it across
        repetitions before reducing.
    """
    ev = cfg.eval
    dsa_strategy = ev.get('dsa_strategy') or None
    if dsa_strategy in ('none', 'None'):
        dsa_strategy = None
    dsa_param = ParamDiffAug()
    net_depth = cfg.model.get('net_depth') or None

    out = {}
    for arch in ev.models:
        accs = []
        for it_eval in range(int(ev.num_eval)):
            net = get_network(arch, channel, num_classes, im_size, net_depth=net_depth)
            _, _, acc = train_and_test(
                it_eval, net, images.detach().clone(), labels.detach().clone(), testloader,
                device, epochs=ev.epochs, lr=ev.lr_net, batch_size=ev.batch_train,
                dsa_strategy=dsa_strategy, dsa_param=dsa_param)
            accs.append(acc)
        logging.info('EVAL %s %s: mean=%.4f std=%.4f (n=%d)', label, arch,
                     float(np.mean(accs)), float(np.std(accs)), len(accs))
        out[arch] = accs
    return out


def summarise(accs_by_arch):
    """Reduce ``{arch: [acc, ...]}`` to the percentage mean/std recorded in results.json."""
    return {arch: {'mean': float(np.mean(a) * 100), 'std': float(np.std(a) * 100),
                   'n': len(a), 'accs': [float(x) for x in a]}
            for arch, a in accs_by_arch.items()}
