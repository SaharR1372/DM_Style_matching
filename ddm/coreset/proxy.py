"""The proxy network that supplies the statistics score-based selectors read.

Herding, k-center, k-means, the uncertainty scores, forgetting, EL2N and GraNd all need a
model's view of the training set.  Rather than each method training its own, one proxy
network is trained on the full training set and every statistic is harvested from it, so a
sweep over selectors at the same budget shares one training run's worth of compute.

The proxy is deliberately cheap -- a handful of epochs of the evaluation architecture.
Coreset baselines in the condensation literature are reported this way; docs/coreset.md
records what each method does with the result.
"""
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddm.augment import DiffAugment, ParamDiffAug
from ddm.coreset.base import ProxyStats
from ddm.models import get_network
from ddm.utils import format_time


class _Indexed(torch.utils.data.Dataset):
    """Wrap a dataset so each item carries its position, which forgetting needs."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, lab = self.base[i]
        return img, lab, i


def _embed(net, x):
    """Penultimate features, tolerating the *_style networks' (embedding, maps) return."""
    out = net.module.embed(x) if hasattr(net, 'module') else net.embed(x)
    return out[0] if isinstance(out, tuple) else out


def collect_stats(cfg, requires, dst_train, channel, num_classes, im_size, device):
    """Train the proxy and return the ProxyStats the requested selector needs.

    Args:
        requires: the selector's ``requires`` tuple.  An empty tuple skips training
            entirely and returns labels only.

    Returns:
        ProxyStats over the full training set, in training-set order.
    """
    labels = torch.tensor([int(dst_train[i][1]) for i in range(len(dst_train))],
                          dtype=torch.long)
    if not requires:
        logging.info('selector needs no proxy statistics; skipping proxy training')
        return ProxyStats(labels)

    p = cfg.coreset.get('proxy', {})
    epochs = int(p.get('epochs', 20))
    batch_size = int(p.get('batch_size', 256))
    num_models = int(p.get('num_models', 1))
    lr = float(p.get('lr', 0.01))
    arch = p.get('arch') or cfg.model.arch
    net_depth = cfg.model.get('net_depth') or None
    dsa_strategy = p.get('dsa_strategy') or None
    if dsa_strategy in ('none', 'None'):
        dsa_strategy = None
    dsa_param = ParamDiffAug()

    n = len(dst_train)
    loader = torch.utils.data.DataLoader(_Indexed(dst_train), batch_size=batch_size,
                                         shuffle=True, num_workers=int(p.get('workers', 4)),
                                         drop_last=False)
    eval_loader = torch.utils.data.DataLoader(_Indexed(dst_train), batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=int(p.get('workers', 4)))

    el2n_sum = torch.zeros(n) if 'el2n' in requires else None
    grand_sum = torch.zeros(n) if 'grand' in requires else None
    forgetting = None
    features = logits_out = None

    start = time.time()
    for model_i in range(num_models):
        net = get_network(arch, channel, num_classes, im_size, net_depth=net_depth).to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=float(p.get('momentum', 0.9)),
                                    weight_decay=float(p.get('weight_decay', 5e-4)))
        criterion = nn.CrossEntropyLoss().to(device)

        # Forgetting bookkeeping (Toneva et al., 2019): an event is a transition from
        # correctly to incorrectly classified between two consecutive presentations of the
        # same example.  Examples never classified correctly have no transition to record,
        # so they are given the highest count at the end -- they are, in the method's own
        # terms, the least forgettable-because-never-learned and are always kept.
        track_forget = 'forgetting' in requires
        prev_acc = torch.zeros(n, dtype=torch.bool)
        ever_correct = torch.zeros(n, dtype=torch.bool)
        counts = torch.zeros(n, dtype=torch.long)

        net.train()
        for ep in range(epochs):
            seen, correct, loss_sum = 0, 0, 0.0
            for img, lab, idx in loader:
                img = img.float().to(device, non_blocking=True)
                lab = lab.long().to(device, non_blocking=True)
                if dsa_strategy:
                    img = DiffAugment(img, dsa_strategy, param=dsa_param)
                out = net(img)
                loss = criterion(out, lab)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    ok = (out.argmax(1) == lab).cpu()
                if track_forget:
                    i = idx.cpu()
                    counts[i] += (prev_acc[i] & ~ok).long()
                    prev_acc[i] = ok
                    ever_correct[i] |= ok
                seen += lab.numel()
                correct += int(ok.sum())
                loss_sum += float(loss.detach()) * lab.numel()
            logging.info('proxy %d/%d epoch %02d/%d: loss=%.4f acc=%.4f elapsed=%s',
                         model_i + 1, num_models, ep + 1, epochs, loss_sum / seen,
                         correct / seen, format_time(time.time() - start))

        if track_forget:
            # Never-learned examples: hardest, so they sort above every real count.
            counts[~ever_correct] = int(counts.max()) + 1
            forgetting = counts

        # One clean pass, no augmentation, in dataset order, for the statistics that
        # describe the trained model's final view of each example.
        net.eval()
        feats, logits = [], []
        with torch.no_grad():
            for img, lab, idx in eval_loader:
                img = img.float().to(device, non_blocking=True)
                h = _embed(net, img)
                z = net(img)
                if 'features' in requires:
                    feats.append(h.flatten(1).cpu())
                logits.append(z.cpu())
        logits_out = torch.cat(logits)
        if 'features' in requires:
            features = torch.cat(feats)

        if el2n_sum is not None or grand_sum is not None:
            probs = F.softmax(logits_out.float(), dim=1)
            onehot = F.one_hot(labels, num_classes).float()
            err = (probs - onehot).norm(dim=1)
            if el2n_sum is not None:
                el2n_sum += err
            if grand_sum is not None:
                # Last-layer approximation of the gradient norm: for cross entropy the
                # gradient w.r.t. the final linear weights is (p - y) h^T, whose Frobenius
                # norm factorises as ||p - y||_2 * ||h||_2.
                if features is None:
                    with torch.no_grad():
                        hs = [ _embed(net, img.float().to(device)).flatten(1).norm(dim=1).cpu()
                               for img, lab, idx in eval_loader ]
                    hnorm = torch.cat(hs)
                else:
                    hnorm = features.norm(dim=1)
                grand_sum += err * hnorm

    return ProxyStats(
        labels=labels,
        features=features,
        logits=logits_out if 'logits' in requires else None,
        forgetting=forgetting,
        el2n=(el2n_sum / num_models) if el2n_sum is not None else None,
        grand=(grand_sum / num_models) if grand_sum is not None else None)
