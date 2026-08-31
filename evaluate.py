#!/usr/bin/env python
"""Evaluate: score a small set that has already been built.

    python evaluate.py --config configs/eval/cross_arch.yaml \
        --checkpoint runs/ours_cifar10_ipc10/condensed_CIFAR10_ConvNet_style_10ipc.pt

Reads the images out of a checkpoint written by train.py and re-runs the evaluation
protocol from ``cfg.eval``.  That is what cross-architecture tables are made of: condense
once on ConvNet, then evaluate the same set on AlexNet, VGG11 and ResNet18 by pointing this
script at an eval config that lists them.

Falls back to the config stored inside the checkpoint for the dataset and the architecture,
so an eval config only has to state the protocol.
"""
import argparse
import logging
import os

import numpy as np
import torch

from ddm.config import Config, load_config
from ddm.data import get_dataset
from ddm.engine.condense import resolve_net_depth
from ddm.engine.evaluator import evaluate_set, summarise
from ddm.utils import append_result, set_seed, setup_logging


def build_parser():
    p = argparse.ArgumentParser(
        description='Evaluate a saved condensed set or coreset.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--config', '-c', required=True, help='path to the YAML eval config')
    p.add_argument('--checkpoint', '-k', required=True,
                   help='.pt written by train.py (condensed_*.pt or coreset_*.pt)')
    p.add_argument('--exp', type=int, default=-1,
                   help='which repetition inside the checkpoint to evaluate; '
                        '-1 (default) evaluates every one and pools the accuracies')
    p.add_argument('--set', dest='overrides', nargs='*', default=[], metavar='KEY=VALUE',
                   help='override config entries')
    return p


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config, args.overrides)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    stored = Config(ckpt.get('config', {}))
    # An eval config states the protocol; the dataset and the architecture come from the
    # run that produced the checkpoint unless the eval config overrides them explicitly.
    for section, key in (('data', 'dataset'), ('data', 'data_path'), ('data', 'ipc'),
                         ('model', 'arch'), ('model', 'net_depth')):
        if key not in cfg.get(section, {}) and key in stored.get(section, {}):
            cfg[section][key] = stored[section][key]

    setup_logging(cfg.output.save_path, 'evaluate.log')
    logging.info('checkpoint: %s', args.checkpoint)
    logging.info('evaluating on: %s', ', '.join(cfg.eval.models))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    (channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader) = get_dataset(cfg.data.dataset, cfg.data.data_path)
    resolve_net_depth(cfg, im_size)
    set_seed(int(cfg.eval.get('seed', 0)))

    entries = ckpt['data']
    if args.exp >= 0:
        entries = [entries[args.exp]]

    accs_all = {arch: [] for arch in cfg.eval.models}
    for i, entry in enumerate(entries):
        images, labels = entry[0], entry[1]
        logging.info('set %d/%d: %s', i + 1, len(entries), tuple(images.shape))
        accs = evaluate_set(images.to(device), labels.to(device), cfg, channel, num_classes,
                            im_size, testloader, device, label='exp=%d' % i)
        for arch, a in accs.items():
            accs_all[arch] += a

    summary = {'name': cfg.get('name', 'evaluate'), 'method': 'evaluate',
               'checkpoint': os.path.abspath(args.checkpoint),
               'dataset': cfg.data.dataset, 'ipc': cfg.data.get('ipc'),
               'num_eval': int(cfg.eval.num_eval), 'sets': len(entries),
               'results': summarise(accs_all)}
    for arch, r in summary['results'].items():
        logging.info('%s: mean = %.2f%%  std = %.2f%%  (n = %d)', arch, r['mean'],
                     r['std'], r['n'])
    logging.info('appended summary -> %s',
                 append_result(cfg.output.save_path, summary, 'eval_results.json'))


if __name__ == '__main__':
    main()
