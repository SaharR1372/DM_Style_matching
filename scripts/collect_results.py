#!/usr/bin/env python
"""Collect every results.json under a directory into one markdown table.

    python scripts/collect_results.py runs
    python scripts/collect_results.py runs --dataset CIFAR10 --ipc 10 --out results_table.md

Each run appends a summary to ``<save_path>/results.json``; this walks a tree of those
files and prints one row per (run, evaluation architecture), sorted so that a dataset's
rows sit together and the baseline comes first.
"""
import argparse
import glob
import json
import os


def describe(rec):
    """A short, readable description of what a run's objective or selector was."""
    if rec.get('method') == 'coreset':
        sel = rec.get('selector', '?')
        opts = rec.get('coreset', {})
        extra = []
        for key in ('forgetting', 'uncertainty', 'el2n', 'grand'):
            o = opts.get(key) or {}
            if key == sel and o.get('order') == 'ascending':
                extra.append('ascending')
            if key == sel and o.get('metric'):
                extra.append(o['metric'])
        n = (opts.get('proxy') or {}).get('num_models', 1)
        if n and int(n) > 1:
            extra.append(f'{n} proxies')
        return sel + (' (' + ', '.join(extra) + ')' if extra else '')

    loss = rec.get('loss', {}) or {}
    icd = loss.get('icd', {}) or {}
    terms = ['L_MMD']
    if loss.get('mm_ratio'):
        terms.append('L_MM(%g)' % loss['mm_ratio'])
    if loss.get('cm_ratio'):
        terms.append('L_CM(%g)' % loss['cm_ratio'])
    if icd.get('content_ratio'):
        terms.append('L_CD(%g)' % icd['content_ratio'])
    if icd.get('style_ratio'):
        terms.append('L_SD(%g)' % icd['style_ratio'])
    return ' + '.join(terms)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', nargs='?', default='runs', help='directory to walk')
    p.add_argument('--dataset', default=None, help='keep only this dataset')
    p.add_argument('--ipc', type=int, default=None, help='keep only this budget')
    p.add_argument('--out', default=None, help='write the table here as well as printing it')
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.root, '**', 'results.json'),
                                 recursive=True)):
        with open(path) as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                print(f'# skipped unreadable {path}')
                continue
        for rec in records:
            if args.dataset and rec.get('dataset') != args.dataset:
                continue
            if args.ipc is not None and rec.get('ipc') != args.ipc:
                continue
            for arch, r in (rec.get('results') or {}).items():
                rows.append((rec.get('dataset', '?'), rec.get('ipc', 0),
                             rec.get('method', '?'), rec.get('name', '?'), describe(rec),
                             arch, r['mean'], r['std'], r['n']))

    if not rows:
        print(f'no results found under {args.root}')
        return

    # baseline first inside each dataset/budget block, then by accuracy
    rows.sort(key=lambda r: (r[0], r[1], r[2] != 'ddm', -r[6]))

    lines = ['| dataset | ipc | run | objective / selector | eval | accuracy (%) | n |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for ds, ipc, method, name, desc, arch, mean, std, n in rows:
        lines.append(f'| {ds} | {ipc} | {name} | {desc} | {arch} | '
                     f'{mean:.2f} ± {std:.2f} | {n} |')
    table = '\n'.join(lines)
    print(table)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(table + '\n')
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
