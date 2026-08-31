"""Decomposed Distribution Matching -- unified trainer.

One entry point for every term of the objective, so that each row of an ablation is a
flag combination rather than a separate script:

    L = L_MMD  +  mm_ratio * L_MM  +  cm_ratio * L_CM  +  L_ICD

  L_MMD  content matching, the original DM objective (Zhao & Bilen).
  L_MM   style moments matching        (Style Matching module, DM_MeanStd_Matching.py)
  L_CM   style correlation matching    (Style Matching module, DM_GramMatching.py)
  L_ICD  intra-class diversity         (ICD module, DM_KNearest.py), where

             L_ICD = icd_ratio * L_content  +  icd_style_ratio * L_style

         Both components match an intra-class spread against the same statistic measured
         on the real batch, so each is bounded, is minimised where the condensed class has
         the spread of the real class, and needs no repulsion strength to be tuned.  The
         content component is the default; see utils_DM.intra_class_diversity_loss.
         --icd_form kl selects the published Eq. 8-9 formulation instead, for reproduction.

Presets:
  --preset dm       plain DM, no style and no diversity term.
  --preset legacy   the published objective bit-for-bit, including the un-reset style
                    accumulator of the released scripts.
  --preset paper    the published objective with that accumulator bug fixed.
  --preset ours     the released method: style read before normalisation with the
                    per-sample estimator, plus the bounded L_ICD.

Results are appended to <save_path>/results.json for aggregation.
"""
import argparse
import copy
import json
import logging
import os
import time

import numpy as np
import torch
from torchvision.utils import save_image

from utils_DM import (get_dataset, get_network, get_eval_pool, evaluate_synset, get_time,
                      DiffAugment, ParamDiffAug, icd_k_for_ipc, intra_class_diversity_loss,
                      intra_class_diversity_loss_kl, moments_matching_loss,
                      correlation_matching_loss, between_class_loss)


def format_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


PRESETS = {
    # Plain DM (Zhao & Bilen): content matching only, no style and no diversity term.
    'dm': dict(style_tap='norm', style_mode='batchavg', legacy_style_accum=False,
               relative_style=False),
    # The published objective, reproduced bit-for-bit including the un-reset style
    # accumulator of the released scripts.
    'legacy': dict(style_tap='norm', style_mode='batchavg', mm_ratio=1e4,
                   icd_ratio=10.0, icd_form='kl', legacy_style_accum=True,
                   relative_style=False),
    # The published objective with the accumulator bug fixed.
    'paper': dict(style_tap='norm', style_mode='batchavg', mm_ratio=1e4,
                  icd_ratio=10.0, icd_form='kl', legacy_style_accum=False,
                  relative_style=False),
    # The released method: style matching read before normalisation with the per-sample
    # estimator, plus the bounded intra-class diversity term.  Coefficients calibrated in
    # private/diagnostics/diag_loss_scale.py.
    'ours': dict(style_tap='conv', style_mode='persample', mm_ratio=180.0,
                 cm_ratio=1e4, icd_ratio=30.0, icd_form='bounded',
                 legacy_style_accum=False, relative_style=True),
}


def build_args():
    p = argparse.ArgumentParser(description='Decomposed Distribution Matching (unified)')
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--model', type=str, default='ConvNet_style')
    p.add_argument('--ipc', type=int, default=10)
    p.add_argument('--eval_mode', type=str, default='SS')
    p.add_argument('--eval_model', type=str, default='ConvNet',
                   help='comma-separated architectures to evaluate the condensed set on. '
                        'ConvNet_style shares its forward pass with ConvNet, so the default '
                        'keeps the numbers comparable with the DM literature; pass a list '
                        'such as ConvNet,AlexNet,VGG11,ResNet18 for cross-architecture.')
    p.add_argument('--num_exp', type=int, default=3)
    p.add_argument('--num_eval', type=int, default=5)
    p.add_argument('--epoch_eval_train', type=int, default=1000)
    p.add_argument('--Iteration', type=int, default=20000)
    p.add_argument('--eval_every', type=int, default=4000)
    p.add_argument('--lr_img', type=float, default=1.0)
    p.add_argument('--lr_net', type=float, default=0.01)
    p.add_argument('--batch_real', type=int, default=256)
    p.add_argument('--batch_train', type=int, default=256)
    p.add_argument('--init', type=str, default='real')
    p.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')
    p.add_argument('--data_path', type=str, default='data')
    p.add_argument('--save_path', type=str, default='result')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--class_chunk', type=int, default=10,
                   help='classes fused into one forward pass; lower it if memory is tight')
    p.add_argument('--net_depth', type=int, default=0,
                   help='ConvNet depth; 0 = auto (4 for im_size >= 64 as the DM literature '
                        'uses for TinyImageNet, 3 otherwise)')

    p.add_argument('--preset', type=str, default=None, choices=sorted(PRESETS))
    p.add_argument('--style_tap', type=str, default='norm', choices=['norm', 'conv', 'act', 'pool'])
    p.add_argument('--style_mode', type=str, default='batchavg', choices=['batchavg', 'persample'])
    p.add_argument('--mm_ratio', type=float, default=0.0, help='weight of L_MM')
    p.add_argument('--cm_ratio', type=float, default=0.0, help='weight of L_CM')
    p.add_argument('--icd_ratio', type=float, default=0.0, help='weight of L_ICD')
    p.add_argument('--icd_form', type=str, default='bounded', choices=['bounded', 'kl'],
                   help="'bounded' (default) matches the intra-class spread of the real "
                        "class and has an attainable optimum; 'kl' is the unbounded "
                        "Eq. 8-9 formulation, kept only to reproduce the published objective")
    p.add_argument('--icd_style_ratio', type=float, default=0.0,
                   help='weight of the style component inside the bounded L_ICD; the '
                        'content component alone is the default')
    p.add_argument('--icd_rank', type=int, default=0,
                   help="principal directions matched by L_ICD's content component; "
                        '0 = min(ipc-1, 16)')
    # Low-level aliases, used by the ablation scripts to drive one component at a time.
    p.add_argument('--sd_ratio', type=float, default=0.0,
                   help="alias for --icd_style_ratio (L_ICD's style component, alone)")
    p.add_argument('--cd_ratio', type=float, default=0.0,
                   help="alias for --icd_ratio with --icd_form bounded (L_ICD's content "
                        'component, alone)')
    p.add_argument('--cd_rank', type=int, default=0, help='alias for --icd_rank')
    p.add_argument('--bc_ratio', type=float, default=0.0,
                   help='weight of L_BC, the between-class geometry term (new)')
    p.add_argument('--icd_k', type=int, default=-1, help='-1 = 0.2*ipc as in the paper')
    p.add_argument('--legacy_style_accum', action='store_true',
                   help='reproduce the published scripts, where the style accumulator is '
                        'never reset between classes')
    p.add_argument('--relative_style', action='store_true',
                   help='normalise the style terms by the magnitude of the real target, '
                        'making them invariant to activation scale')
    p.add_argument('--tag', type=str, default='', help='label for this run in results.json')

    args = p.parse_args()
    if args.preset is not None:
        for k, v in PRESETS[args.preset].items():
            setattr(args, k, v)
        if not args.tag:
            args.tag = args.preset

    # Fold the low-level aliases onto the public L_ICD knobs.  --cd_ratio drives the
    # content component and --sd_ratio the style one, so an ablation can run either alone.
    if args.cd_ratio:
        args.icd_ratio = args.cd_ratio
        args.icd_form = 'bounded'
    if args.cd_rank:
        args.icd_rank = args.cd_rank
    if args.sd_ratio:
        args.icd_style_ratio = args.sd_ratio
    args.method = 'DDM'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = args.dsa_strategy not in ('none', 'None')
    args.dc_aug_param = None
    if args.icd_k < 0:
        args.icd_k = icd_k_for_ipc(args.ipc)
    args.needs_style = (args.mm_ratio != 0) or (args.cm_ratio != 0) or (args.icd_style_ratio != 0)
    return args


def main():
    args = build_args()
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(message)s',
        handlers=[logging.FileHandler(os.path.join(args.save_path, 'training.log')),
                  logging.StreamHandler()])

    eval_it_pool = list(range(args.eval_every, args.Iteration + 1, args.eval_every))
    if args.Iteration not in eval_it_pool:
        eval_it_pool.append(args.Iteration)

    (channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader) = get_dataset(args.dataset, args.data_path)

    # 0 = auto: the DM literature evaluates 64x64 datasets (TinyImageNet) on ConvNetD4 and
    # 32x32 ones on ConvNetD3, so resolve the depth from the resolution unless it is given.
    if args.net_depth == 0:
        args.net_depth = 4 if min(im_size) >= 64 else None
    logging.info('net_depth = %s', args.net_depth)
    model_eval_pool = ([m.strip() for m in args.eval_model.split(',') if m.strip()]
                       if args.eval_model else get_eval_pool(args.eval_mode, args.model, args.model))

    logging.info('run config: %s', json.dumps(
        {k: v for k, v in vars(args).items() if k not in ('dsa_param',)}, default=str))

    accs_all_exps = {k: [] for k in model_eval_pool}
    data_save = []

    images_all = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(args.device)
    labels_all = torch.tensor([dst_train[i][1] for i in range(len(dst_train))],
                              dtype=torch.long, device=args.device)
    indices_class = [(labels_all == c).nonzero(as_tuple=True)[0] for c in range(num_classes)]
    logging.info('real images: %d, per class min=%d max=%d', images_all.shape[0],
                 min(len(i) for i in indices_class), max(len(i) for i in indices_class))

    def get_images(c, n):
        idx = indices_class[c][torch.randperm(len(indices_class[c]), device=args.device)[:n]]
        return images_all[idx]

    start_time = time.time()
    for exp in range(args.num_exp):
        torch.manual_seed(args.seed + exp)
        np.random.seed(args.seed + exp)
        logging.info('\n================== Exp %d ==================\n', exp)

        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
                                dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.arange(num_classes, device=args.device).repeat_interleave(args.ipc)
        if args.init == 'real':
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data

        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        logging.info('%s training begins', get_time())

        for it in range(args.Iteration + 1):
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size,
                                               net_depth=args.net_depth).to(args.device)
                        _, _, acc_test = evaluate_synset(
                            it_eval, net_eval, copy.deepcopy(image_syn.detach()),
                            copy.deepcopy(label_syn.detach()), testloader, args)
                        accs.append(acc_test)
                    logging.info('EVAL it=%d %s: mean=%.4f std=%.4f (n=%d)',
                                 it, model_eval, np.mean(accs), np.std(accs), len(accs))
                    if it == args.Iteration:
                        accs_all_exps[model_eval] += accs

                vis = image_syn.detach().cpu().clone()
                for ch in range(channel):
                    vis[:, ch] = vis[:, ch] * std[ch] + mean[ch]
                save_image(vis.clamp(0, 1),
                           os.path.join(args.save_path, 'vis_%s_%s_%dipc_exp%d_iter%d.png'
                                        % (args.dataset, args.model, args.ipc, exp, it)),
                           nrow=args.ipc)

            if it == args.Iteration:
                break

            net = get_network(args.model, channel, num_classes, im_size,
                              net_depth=args.net_depth).to(args.device)
            if args.needs_style and hasattr(net, 'set_style_tap'):
                net.set_style_tap(args.style_tap)
            net.train()
            for param in net.parameters():
                param.requires_grad = False
            embed = net.module.embed if torch.cuda.device_count() > 1 and hasattr(net, 'module') else net.embed

            loss = torch.zeros((), device=args.device)
            parts = {'mmd': 0.0, 'mm': 0.0, 'cm': 0.0, 'icd': 0.0, 'sd': 0.0, 'cd': 0.0,
                     'bc': 0.0}
            style_accum = torch.zeros((), device=args.device)  # only used by --legacy_style_accum
            # L_BC is the only term that spans classes, so its inputs are collected across
            # the chunk loop and consumed after it.
            mu_syn_all, mu_real_all = [], []

            for c0 in range(0, num_classes, args.class_chunk):
                classes = range(c0, min(c0 + args.class_chunk, num_classes))
                reals, syns = [], []
                for c in classes:
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc]
                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    reals.append(img_real)
                    syns.append(img_syn)

                # InstanceNorm couples nothing across samples, so fusing the classes into
                # a single forward pass is numerically identical and much faster.
                with torch.no_grad():
                    out_real = embed(torch.cat(reals, 0))
                out_syn = embed(torch.cat(syns, 0))
                # *_style networks return (embedding, per-layer feature maps); the plain
                # ones return just the embedding.
                out_real, feats_real = out_real if isinstance(out_real, tuple) else (out_real, ())
                out_syn, feats_syn = out_syn if isinstance(out_syn, tuple) else (out_syn, ())
                if not args.needs_style:
                    feats_real = feats_syn = ()

                nr, ns = args.batch_real, args.ipc
                for j, c in enumerate(classes):
                    r = slice(j * nr, (j + 1) * nr)
                    s = slice(j * ns, (j + 1) * ns)

                    mu_r, mu_s = out_real[r].mean(0), out_syn[s].mean(0)
                    l_mmd = torch.sum((mu_r - mu_s) ** 2)
                    loss = loss + l_mmd
                    parts['mmd'] += float(l_mmd)
                    if args.bc_ratio:
                        mu_syn_all.append(mu_s)
                        mu_real_all.append(mu_r)

                    if args.needs_style:
                        style_c = torch.zeros((), device=args.device)
                        for f_r, f_s in zip(feats_real, feats_syn):
                            if args.mm_ratio:
                                l = moments_matching_loss(f_s[s], f_r[r], mode=args.style_mode,
                                                          relative=args.relative_style)
                                style_c = style_c + args.mm_ratio * l
                                parts['mm'] += float(l)
                            if args.cm_ratio:
                                l = correlation_matching_loss(f_s[s], f_r[r])
                                style_c = style_c + args.cm_ratio * l
                                parts['cm'] += float(l)
                        style_c = style_c / max(len(feats_syn), 1)
                        if args.legacy_style_accum:
                            # The published scripts never reset the accumulator between
                            # classes and divide it by the layer count each time, so an
                            # early class's style loss is re-added, geometrically damped,
                            # for every later class.  Kept here only to quantify the effect.
                            style_accum = (style_accum + style_c * len(feats_syn)) / max(len(feats_syn), 1)
                            loss = loss + style_accum
                        else:
                            loss = loss + style_c

                    if args.icd_form == 'kl':
                        # Published Eq. 8-9, retained for reproduction only.
                        if args.icd_ratio and args.icd_k >= 1:
                            l_icd = intra_class_diversity_loss_kl(out_syn[s], k=args.icd_k)
                            loss = loss + args.icd_ratio * l_icd
                            parts['icd'] += float(l_icd)
                    elif args.icd_ratio or args.icd_style_ratio:
                        use_style = args.icd_style_ratio and feats_syn
                        l_icd, icd_parts = intra_class_diversity_loss(
                            out_syn[s], out_real[r],
                            feat_syn=[f[s] for f in feats_syn] if use_style else None,
                            feat_real=[f[r] for f in feats_real] if use_style else None,
                            content_ratio=args.icd_ratio, style_ratio=args.icd_style_ratio,
                            rank=args.icd_rank, relative=args.relative_style,
                            return_parts=True)
                        loss = loss + l_icd
                        parts['cd'] += icd_parts['content']
                        parts['sd'] += icd_parts['style']
                        parts['icd'] += float(l_icd)

            # L_BC needs every class at once, so it is applied after the chunk loop.
            if args.bc_ratio and len(mu_syn_all) >= 2:
                l_bc = between_class_loss(torch.stack(mu_syn_all), torch.stack(mu_real_all),
                                          relative=args.relative_style)
                loss = loss + args.bc_ratio * l_bc
                parts['bc'] = float(l_bc)

            # A diverged run is worse than a crashed one: once image_syn goes non-finite it
            # stays that way, every later evaluation returns chance, and the summary reports
            # that chance number as though it were a measurement (c4_best_c10_ipc1 produced a
            # confident-looking 10.00 +/- 0.00 this way).  Fail loudly instead.
            if not torch.isfinite(loss):
                logging.error('non-finite loss at it=%05d (exp %d) -- aborting; the '
                              'coefficients are unsafe for this configuration', it, exp)
                raise FloatingPointError(
                    f'non-finite loss at iteration {it}: mmd={parts["mmd"]:.4g} '
                    f'mm={parts["mm"]:.4g} cm={parts["cm"]:.4g} sd={parts["sd"]:.4g}')

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            if it % 100 == 0:
                el = time.time() - start_time
                eta = el / max(it, 1) * (args.Iteration - it)
                logging.info('it=%05d loss=%.4f | mmd=%.4f mm=%.6f cm=%.6f icd=%.4f sd=%.6f '
                             'cd=%.6f bc=%.6f | elapsed=%s eta=%s', it, float(loss),
                             parts['mmd'], parts['mm'], parts['cm'], parts['icd'], parts['sd'],
                             parts['cd'], parts['bc'], format_time(el), format_time(eta))

        data_save.append([image_syn.detach().cpu(), label_syn.detach().cpu()])
        torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, 'args': vars(args)},
                   os.path.join(args.save_path, 'res_%s_%s_%dipc.pt'
                                % (args.dataset, args.model, args.ipc)))

    logging.info('\n==================== Final Results ====================')
    summary = {'tag': args.tag or args.preset or 'custom',
               'dataset': args.dataset, 'ipc': args.ipc, 'model': args.model,
               'iteration': args.Iteration, 'num_exp': args.num_exp, 'num_eval': args.num_eval,
               'style_tap': args.style_tap, 'style_mode': args.style_mode,
               'mm_ratio': args.mm_ratio, 'cm_ratio': args.cm_ratio,
               'icd_ratio': args.icd_ratio, 'icd_form': args.icd_form,
               'icd_style_ratio': args.icd_style_ratio, 'icd_rank': args.icd_rank,
               'sd_ratio': args.sd_ratio, 'cd_ratio': args.cd_ratio,
               'cd_rank': args.cd_rank, 'net_depth': args.net_depth,
               'legacy_style_accum': args.legacy_style_accum,
               'relative_style': args.relative_style, 'seed': args.seed,
               'hours': round((time.time() - start_time) / 3600, 2), 'results': {}}
    for key, accs in accs_all_exps.items():
        logging.info('%d exps, train %s, eval %d x %s: mean = %.2f%%  std = %.2f%%',
                     args.num_exp, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100)
        summary['results'][key] = {'mean': float(np.mean(accs) * 100),
                                   'std': float(np.std(accs) * 100), 'n': len(accs),
                                   'accs': [float(a) for a in accs]}

    res_file = os.path.join(args.save_path, 'results.json')
    allres = []
    if os.path.exists(res_file):
        with open(res_file) as f:
            allres = json.load(f)
    allres.append(summary)
    with open(res_file, 'w') as f:
        json.dump(allres, f, indent=2)
    logging.info('appended summary -> %s', res_file)


if __name__ == '__main__':
    main()
