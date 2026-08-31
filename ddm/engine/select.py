"""Coreset selection -- the run loop for ``method: coreset``.

Mirrors ddm.engine.condense: build the small set, evaluate it under the same protocol,
append the summary to the same results.json.  The only structural difference is that the
set is chosen from real images rather than optimised, so there is no iteration loop --
the cost is one proxy training run (none at all for ``random``) plus the evaluation.
"""
import json
import logging
import os
import time

import torch

from ddm.coreset import build_selector
from ddm.coreset.proxy import collect_stats
from ddm.data import get_dataset
from ddm.engine.condense import resolve_net_depth
from ddm.engine.evaluator import evaluate_set, summarise
from ddm.utils import append_result, get_time, save_image_grid, set_seed


def select(cfg):
    """Select a coreset under ``cfg``, evaluate it, and return the run summary."""
    save_path = cfg.output.save_path
    os.makedirs(cfg.data.data_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opts, ev = cfg.coreset, cfg.eval
    ipc = int(cfg.data.ipc)

    (channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader) = get_dataset(cfg.data.dataset, cfg.data.data_path)
    resolve_net_depth(cfg, im_size)
    logging.info('config: %s', json.dumps(cfg.to_dict(), default=str))

    accs_all_exps = {arch: [] for arch in ev.models}
    start_time = time.time()
    saved = []

    for exp in range(int(opts.get('num_exp', 1))):
        seed = int(opts.get('seed', 0)) + exp
        set_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        logging.info('\n================== Exp %d (%s, seed %d) ==================\n',
                     exp, opts.selector, seed)

        selector = build_selector(cfg, rng)
        stats = collect_stats(cfg, selector.requires, dst_train, channel, num_classes,
                              im_size, device)
        logging.info('%s selecting %d images/class with %s', get_time(), ipc, selector.name)
        idx = selector.select(num_classes, stats.labels, ipc, stats)

        images = torch.stack([dst_train[int(i)][0] for i in idx]).to(device)
        labels = stats.labels[idx].to(device)
        logging.info('selected %d images (%d classes x %d)', len(idx), num_classes, ipc)

        accs = evaluate_set(images, labels, cfg, channel, num_classes, im_size, testloader,
                            device, label='%s exp=%d' % (opts.selector, exp))
        for arch, a in accs.items():
            accs_all_exps[arch] += a

        if cfg.output.get('save_images', True):
            save_image_grid(images, mean, std,
                            os.path.join(save_path, 'vis_%s_%s_%dipc_exp%d.png'
                                         % (cfg.data.dataset, opts.selector, ipc, exp)),
                            nrow=ipc)
        saved.append([images.detach().cpu(), labels.detach().cpu(), idx.cpu()])
        torch.save({'data': saved, 'accs_all_exps': accs_all_exps, 'config': cfg.to_dict()},
                   os.path.join(save_path, 'coreset_%s_%s_%dipc.pt'
                                % (cfg.data.dataset, opts.selector, ipc)))

    logging.info('\n==================== Final Results ====================')
    summary = {'name': cfg.get('name', opts.selector), 'method': 'coreset',
               'selector': opts.selector, 'dataset': cfg.data.dataset, 'ipc': ipc,
               'model': cfg.model.arch, 'net_depth': cfg.model.get('net_depth'),
               'num_exp': int(opts.get('num_exp', 1)), 'num_eval': int(ev.num_eval),
               'coreset': opts.to_dict(), 'seed': int(opts.get('seed', 0)),
               'hours': round((time.time() - start_time) / 3600, 2),
               'results': summarise(accs_all_exps)}
    for arch, r in summary['results'].items():
        logging.info('%s, eval %d x %s: mean = %.2f%%  std = %.2f%%',
                     opts.selector, r['n'], arch, r['mean'], r['std'])
    logging.info('appended summary -> %s', append_result(save_path, summary))
    return summary
