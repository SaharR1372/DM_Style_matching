"""Distribution-matching condensation -- the training loop for ``method: ddm``.

One loop covers every term of the objective, so each row of an ablation is a config
difference rather than a separate script:

    L = L_MMD  +  mm_ratio * L_MM  +  cm_ratio * L_CM  +  L_ICD

  L_MMD  content matching, the original DM objective (Zhao & Bilen).
  L_MM   style moments matching       -- Style Matching module, ddm.losses.style
  L_CM   style correlation matching   -- Style Matching module, ddm.losses.style
  L_ICD  intra-class diversity        -- ICD module, ddm.losses.diversity, where

             L_ICD = icd.content_ratio * L_CD  +  icd.style_ratio * L_SD

         Both components match an intra-class spread against the same statistic measured on
         the real batch, so each is bounded, is minimised where the condensed class has the
         spread of the real class, and needs no repulsion strength to be tuned.  Setting
         ``loss.icd.form: kl`` selects the published Eq. 8-9 formulation instead.

Setting a weight to zero removes its term: ``configs/dm`` leaves all of them at zero and so
reproduces plain distribution matching.  See docs/method.md.
"""
import json
import logging
import os
import time

import torch

from ddm.augment import DiffAugment, ParamDiffAug
from ddm.data import get_dataset
from ddm.engine.evaluator import evaluate_set, summarise
from ddm.losses import (correlation_matching_loss, icd_k_for_ipc,
                        intra_class_diversity_loss, intra_class_diversity_loss_kl,
                        moments_matching_loss)
from ddm.models import get_network
from ddm.utils import append_result, format_time, get_time, save_image_grid, set_seed


def resolve_net_depth(cfg, im_size):
    """0 (or absent) means: pick the depth the DM literature uses for this resolution.

    64x64 datasets such as TinyImageNet are condensed and evaluated on ConvNetD4, 32x32
    ones on ConvNetD3, which is the ConvNet default.
    """
    depth = cfg.model.get('net_depth') or 0
    if int(depth) == 0:
        depth = 4 if min(im_size) >= 64 else None
    cfg.model.net_depth = depth
    return depth


def _needs_style(loss_cfg):
    """Whether any active term reads the intermediate feature maps."""
    icd = loss_cfg.get('icd', {})
    return bool(loss_cfg.get('mm_ratio') or loss_cfg.get('cm_ratio')
                or icd.get('style_ratio'))


def condense(cfg):
    """Synthesise a condensed set under ``cfg`` and return the run summary."""
    save_path = cfg.output.save_path
    os.makedirs(cfg.data.data_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cond, loss_cfg, ev = cfg.condense, cfg.loss, cfg.eval
    icd_cfg = loss_cfg.get('icd', {})
    icd_form = icd_cfg.get('form', 'bounded')
    icd_content = float(icd_cfg.get('content_ratio', 0.0))
    icd_style = float(icd_cfg.get('style_ratio', 0.0))
    icd_rank = int(icd_cfg.get('rank', 0))
    icd_k = int(icd_cfg.get('k', -1))
    if icd_k < 0:
        icd_k = icd_k_for_ipc(int(cfg.data.ipc))
    needs_style = _needs_style(loss_cfg)

    dsa_strategy = cond.get('dsa_strategy') or None
    if dsa_strategy in ('none', 'None'):
        dsa_strategy = None
    dsa_param = ParamDiffAug()

    (channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader) = get_dataset(cfg.data.dataset, cfg.data.data_path)
    net_depth = resolve_net_depth(cfg, im_size)
    logging.info('net_depth = %s', net_depth)
    logging.info('config: %s', json.dumps(cfg.to_dict(), default=str))

    ipc = int(cfg.data.ipc)
    iterations = int(cond.iterations)
    eval_every = int(ev.get('every', 0) or iterations)
    eval_it_pool = sorted(set(list(range(eval_every, iterations + 1, eval_every))
                              + [iterations]))

    images_all = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
    labels_all = torch.tensor([dst_train[i][1] for i in range(len(dst_train))],
                              dtype=torch.long, device=device)
    indices_class = [(labels_all == c).nonzero(as_tuple=True)[0] for c in range(num_classes)]
    logging.info('real images: %d, per class min=%d max=%d', images_all.shape[0],
                 min(len(i) for i in indices_class), max(len(i) for i in indices_class))

    def get_images(c, n):
        idx = indices_class[c][torch.randperm(len(indices_class[c]), device=device)[:n]]
        return images_all[idx]

    accs_all_exps = {arch: [] for arch in ev.models}
    data_save = []
    start_time = time.time()

    for exp in range(int(cond.num_exp)):
        set_seed(int(cond.seed) + exp)
        logging.info('\n================== Exp %d ==================\n', exp)

        image_syn = torch.randn(size=(num_classes * ipc, channel, im_size[0], im_size[1]),
                                dtype=torch.float, requires_grad=True, device=device)
        label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc)
        if cond.get('init', 'real') == 'real':
            for c in range(num_classes):
                image_syn.data[c * ipc:(c + 1) * ipc] = get_images(c, ipc).detach().data

        optimizer_img = torch.optim.SGD([image_syn], lr=float(cond.lr_img),
                                        momentum=float(cond.get('momentum', 0.5)))
        optimizer_img.zero_grad()
        logging.info('%s training begins', get_time())

        for it in range(iterations + 1):
            if it in eval_it_pool:
                accs = evaluate_set(image_syn.detach(), label_syn.detach(), cfg, channel,
                                    num_classes, im_size, testloader, device,
                                    label='it=%d exp=%d' % (it, exp))
                if it == iterations:
                    for arch, a in accs.items():
                        accs_all_exps[arch] += a
                if cfg.output.get('save_images', True):
                    save_image_grid(
                        image_syn, mean, std,
                        os.path.join(save_path, 'vis_%s_%s_%dipc_exp%d_iter%d.png'
                                     % (cfg.data.dataset, cfg.model.arch, ipc, exp, it)),
                        nrow=ipc)

            if it == iterations:
                break

            net = get_network(cfg.model.arch, channel, num_classes, im_size,
                              net_depth=net_depth).to(device)
            if needs_style and hasattr(net, 'set_style_tap'):
                net.set_style_tap(loss_cfg.get('style_tap', 'norm'))
            net.train()
            for param in net.parameters():
                param.requires_grad = False
            embed = (net.module.embed
                     if torch.cuda.device_count() > 1 and hasattr(net, 'module') else net.embed)

            loss = torch.zeros((), device=device)
            parts = {'mmd': 0.0, 'mm': 0.0, 'cm': 0.0, 'icd': 0.0, 'sd': 0.0, 'cd': 0.0}
            style_accum = torch.zeros((), device=device)  # only used by legacy_style_accum

            chunk = int(cond.get('class_chunk', 10)) or num_classes
            for c0 in range(0, num_classes, chunk):
                classes = range(c0, min(c0 + chunk, num_classes))
                reals, syns = [], []
                for c in classes:
                    img_real = get_images(c, int(cond.batch_real))
                    img_syn = image_syn[c * ipc:(c + 1) * ipc]
                    if dsa_strategy:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, dsa_strategy, seed=seed, param=dsa_param)
                        img_syn = DiffAugment(img_syn, dsa_strategy, seed=seed, param=dsa_param)
                    reals.append(img_real)
                    syns.append(img_syn)

                # InstanceNorm couples nothing across samples, so fusing the classes into a
                # single forward pass is numerically identical and much faster.
                with torch.no_grad():
                    out_real = embed(torch.cat(reals, 0))
                out_syn = embed(torch.cat(syns, 0))
                # *_style networks return (embedding, per-layer feature maps); the plain
                # ones return just the embedding.
                out_real, feats_real = out_real if isinstance(out_real, tuple) else (out_real, ())
                out_syn, feats_syn = out_syn if isinstance(out_syn, tuple) else (out_syn, ())
                if not needs_style:
                    feats_real = feats_syn = ()

                nr, ns = int(cond.batch_real), ipc
                for j, c in enumerate(classes):
                    r = slice(j * nr, (j + 1) * nr)
                    s = slice(j * ns, (j + 1) * ns)

                    mu_r, mu_s = out_real[r].mean(0), out_syn[s].mean(0)
                    l_mmd = torch.sum((mu_r - mu_s) ** 2)
                    loss = loss + l_mmd
                    parts['mmd'] += float(l_mmd.detach())

                    if needs_style:
                        style_c = torch.zeros((), device=device)
                        for f_r, f_s in zip(feats_real, feats_syn):
                            if loss_cfg.get('mm_ratio'):
                                l = moments_matching_loss(
                                    f_s[s], f_r[r], mode=loss_cfg.get('style_mode', 'batchavg'),
                                    relative=loss_cfg.get('relative_style', False))
                                style_c = style_c + float(loss_cfg.mm_ratio) * l
                                parts['mm'] += float(l.detach())
                            if loss_cfg.get('cm_ratio'):
                                l = correlation_matching_loss(f_s[s], f_r[r])
                                style_c = style_c + float(loss_cfg.cm_ratio) * l
                                parts['cm'] += float(l.detach())
                        style_c = style_c / max(len(feats_syn), 1)
                        if loss_cfg.get('legacy_style_accum', False):
                            # The published scripts never reset the accumulator between
                            # classes and divide it by the layer count each time, so an
                            # early class's style loss is re-added, geometrically damped,
                            # for every later class.  Kept only to quantify the effect.
                            style_accum = ((style_accum + style_c * len(feats_syn))
                                           / max(len(feats_syn), 1))
                            loss = loss + style_accum
                        else:
                            loss = loss + style_c

                    if icd_form == 'kl':
                        # Published Eq. 8-9, retained for reproduction only.
                        if icd_content and icd_k >= 1:
                            l_icd = intra_class_diversity_loss_kl(out_syn[s], k=icd_k)
                            loss = loss + icd_content * l_icd
                            parts['icd'] += float(l_icd.detach())
                    elif icd_content or icd_style:
                        use_style = icd_style and feats_syn
                        l_icd, icd_parts = intra_class_diversity_loss(
                            out_syn[s], out_real[r],
                            feat_syn=[f[s] for f in feats_syn] if use_style else None,
                            feat_real=[f[r] for f in feats_real] if use_style else None,
                            content_ratio=icd_content, style_ratio=icd_style,
                            rank=icd_rank, relative=loss_cfg.get('relative_style', True),
                            return_parts=True)
                        loss = loss + l_icd
                        parts['cd'] += icd_parts['content']
                        parts['sd'] += icd_parts['style']
                        parts['icd'] += float(l_icd.detach())

            # A diverged run is worse than a crashed one: once image_syn goes non-finite it
            # stays that way, every later evaluation returns chance, and the summary reports
            # that chance number as though it were a measurement.  Fail loudly instead.
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
                eta = el / max(it, 1) * (iterations - it)
                logging.info('it=%05d loss=%.4f | mmd=%.4f mm=%.6f cm=%.6f icd=%.4f '
                             'sd=%.6f cd=%.6f | elapsed=%s eta=%s', it, float(loss),
                             parts['mmd'], parts['mm'], parts['cm'], parts['icd'],
                             parts['sd'], parts['cd'], format_time(el), format_time(eta))

        data_save.append([image_syn.detach().cpu(), label_syn.detach().cpu()])
        torch.save({'data': data_save, 'accs_all_exps': accs_all_exps,
                    'config': cfg.to_dict()},
                   os.path.join(save_path, 'condensed_%s_%s_%dipc.pt'
                                % (cfg.data.dataset, cfg.model.arch, ipc)))

    logging.info('\n==================== Final Results ====================')
    summary = {'name': cfg.get('name', 'ddm'), 'method': 'ddm',
               'dataset': cfg.data.dataset, 'ipc': ipc, 'model': cfg.model.arch,
               'net_depth': net_depth, 'iterations': iterations,
               'num_exp': int(cond.num_exp), 'num_eval': int(ev.num_eval),
               'loss': loss_cfg.to_dict(), 'seed': int(cond.seed),
               'hours': round((time.time() - start_time) / 3600, 2),
               'results': summarise(accs_all_exps)}
    for arch, r in summary['results'].items():
        logging.info('%d exps, train %s, eval %d x %s: mean = %.2f%%  std = %.2f%%',
                     cond.num_exp, cfg.model.arch, r['n'], arch, r['mean'], r['std'])
    logging.info('appended summary -> %s', append_result(save_path, summary))
    return summary
