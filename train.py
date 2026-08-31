#!/usr/bin/env python
"""Train: build a small set from a config, then score it.

    python train.py --config configs/ours/cifar10_ipc10.yaml
    python train.py --config configs/coreset/cifar10_ipc10_herding.yaml
    python train.py --config configs/dm/cifar10_ipc10.yaml --set condense.iterations=2000

The config's ``method`` field decides what runs -- ``ddm`` synthesises a condensed set by
distribution matching, ``coreset`` selects real images -- and both are evaluated under the
same protocol, so their results.json rows are comparable.  Anything in the config can be
overridden from the command line with ``--set section.key=value``, which is meant for
sweeps; a run you intend to report should have its own config file.
"""
import argparse
import logging
import os

from ddm.config import load_config
from ddm.utils import setup_logging


def build_parser():
    p = argparse.ArgumentParser(
        description='Train a condensed set or select a coreset from a YAML config.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--config', '-c', required=True, help='path to the YAML config')
    p.add_argument('--set', dest='overrides', nargs='*', default=[], metavar='KEY=VALUE',
                   help='override config entries, e.g. --set data.ipc=1 '
                        'output.save_path=runs/tmp')
    return p


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config, args.overrides)

    setup_logging(cfg.output.save_path, 'train.log')
    logging.info('config: %s', cfg.config_path)
    if args.overrides:
        logging.info('overrides: %s', ' '.join(args.overrides))
    cfg.dump(os.path.join(cfg.output.save_path, 'config.resolved.yaml'))

    if cfg.method == 'ddm':
        from ddm.engine.condense import condense
        condense(cfg)
    else:
        from ddm.engine.select import select
        select(cfg)


if __name__ == '__main__':
    main()
