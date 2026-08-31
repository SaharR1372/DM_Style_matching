"""Intra-Class Diversity (ICD) module -- L_ICD.

Two formulations live here:

  ``intra_class_diversity_loss``     the released form -- bounded and target-matched,
                                     built from a content and a style component.
  ``intra_class_diversity_loss_kl``  the Eq. 8-9 form -- unbounded KL repulsion, kept so
                                     the published objective can still be reproduced.

See ``intra_class_diversity_loss`` for why the released form is the one to use, and
docs/method.md for the measured comparison between them.
"""
import torch
import torch.nn.functional as F

from ddm.losses.style import _sq_diff, style_vector


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
