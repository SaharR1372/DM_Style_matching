# Decomposed Distribution Matching

The method of *Decomposed Distribution Matching in Dataset Condensation* (WACV 2025),
as implemented in `ddm/engine/condense.py` and `ddm/losses/`.

## The starting point

Distribution Matching (Zhao & Bilen, WACV 2023) condenses a dataset by making the class-mean
embedding of the synthetic images match that of the real images, under a randomly
initialised network resampled at every iteration:

```
L_MMD = Σ_c ‖ E_{x ∈ c}[ φ(x) ] − E_{x̃ ∈ c}[ φ(x̃) ] ‖²
```

It is fast, because it avoids the bi-level optimisation that gradient-matching methods need,
and that speed is paid for in accuracy. This work decomposes the gap into two parts.

## The decomposition

Split what a feature map carries into **content** -- where the class sits in embedding space
-- and **style** -- the channel statistics that describe how it is rendered. Measured
against real data, a DM-condensed set is deficient in both:

1. **Style discrepancy.** L_MMD constrains only the final pooled embedding. The channel
   moments and channel correlations of the intermediate feature maps are left free, and they
   drift away from the real data's.
2. **Limited intra-class diversity.** L_MMD constrains only the class *mean*. Every
   arrangement of `ipc` images with the right mean is an equally good solution, including
   the degenerate one where all `ipc` images are the same prototype.

So the objective gains one module per deficiency:

```
L = L_MMD  +  mm_ratio · L_MM  +  cm_ratio · L_CM  +  L_ICD
    └ content ┘  └──── Style Matching (SM) ────┘   └ Intra-Class Diversity ┘
```

Every weight defaults to zero, so `configs/dm` -- with all of them left alone -- is exactly
plain distribution matching, and each ablation row is a config diff.

---

## Style Matching (SM)

`ddm/losses/style.py`. Two statistics of the intermediate feature maps, both computed per
class per layer.

### L_MM — moments matching

The channel-wise first and second moments, the standard style descriptor:

```
L_MM = ½ [ d( μ_syn, μ_real ) + d( σ_syn, σ_real ) ]
```

`loss.style_mode` chooses the estimator:

- **`batchavg`** reproduces the published implementation: average the feature maps over the
  batch first, then take the spatial mean and std of that average. The spatial variance of a
  batch-average shrinks with batch size, so the target computed from `batch_real` real images
  is not on the same scale as the value computed from `ipc` synthetic images.
- **`persample`** compares `E_x[μ(x)]` and `E_x[σ(x)]` -- sample means of a per-sample
  quantity, and therefore unbiased with respect to batch size. This is the released default.

### L_CM — correlation matching

The Gram matrix, capturing correlations between channels:

```
L_CM = ‖ G(F_syn) − G(F_real) ‖²_F ,   G(F) = F Fᵀ / (C·H·W)
```

### Where the style is read: `loss.style_tap`

`ConvNet_style` can expose its feature maps at four points -- `conv`, `norm`, `act`, `pool`.
This matters more than it looks. The network uses **InstanceNorm**, which normalises away
exactly the per-sample channel statistics that L_MM is trying to match. Reading at `norm`
therefore measures style *after* it has largely been removed. The released configuration
reads at `conv`, before normalisation, where the style still exists.

### `loss.relative_style`

Reading before normalisation costs something: pre-norm magnitudes depend on the random
initialisation and grow with depth, so an absolute loss silently weights the layers by their
activation scale, and its coefficient has to be retuned for every tap, architecture and
dataset. Setting `relative_style: true` divides each term by the magnitude of its own real
target, making it scale-free. This is why `mm_ratio: 180` transfers unchanged from CIFAR10
to TinyImageNet.

---

## Intra-Class Diversity (ICD)

`ddm/losses/diversity.py`. The module that keeps the `ipc` images of a class from collapsing
onto one prototype. Two formulations ship, selected by `loss.icd.form`.

### `form: kl` — the published Eq. 8-9

For each synthetic sample, take the mean embedding *m* of its *k* nearest intra-class
neighbours (`k = 0.2 · ipc`) and **maximise** `KL( S(φ(x̃)) ‖ S(m) )`.

This is what the paper specifies, and it is kept so the published objective can be
reproduced. It has a structural problem: maximising an unbounded divergence gives the term
no attainable optimum. There is no weight at which it both spreads the samples and then
stops -- its descent direction never terminates, and past a moderate weight it simply
overwhelms the content matching and disperses the class far past anything present in the
real data. Its weight consequently has to be retuned per dataset and per budget.

### `form: bounded` — the released implementation

Same role, built the opposite way. Each component compares a synthetic statistic against the
**same statistic measured on the real batch**:

```
L_ICD = content_ratio · L_CD  +  style_ratio · L_SD
```

- **L_CD (content).** Take the top-*r* principal directions of the real class (computed from
  the real batch, no gradient), and match the synthetic set's standard deviation along each
  one to the real set's:
  `L_CD = mean_k ( std(S·v_k) − std(R·v_k) )² / mean_k std(R·v_k)²`.
  A full covariance cannot be matched -- `ipc` synthetic samples span at most `ipc − 1`
  dimensions of a 2048-dimensional embedding -- so `loss.icd.rank` is capped at `ipc − 1`,
  exactly the number of variances the synthetic set has the freedom to set.
- **L_SD (style).** The same idea one level down: match the across-sample spread of the
  per-sample style descriptors of the intermediate feature maps.

Both are bounded below by zero, are minimised exactly where the condensed class has the
spread of the real class, and rise again if the class is pushed *wider* than the data. The
target is read off the data rather than set by a coefficient, and both are normalised by the
magnitude of their own target, so one weight transfers across taps, architectures,
resolutions and datasets without retuning.

### The two components are alternatives, not additions

They constrain the same second-order intra-class structure, so they compete for the same
headroom rather than adding. The default activates the **content** component alone --
the axis the paper's module describes. Enabling both measured slightly *below* either alone
(`configs/ablation/icd_both_cifar10_ipc10.yaml`; numbers in [results.md](results.md)).

`style_ratio` is harmless at 32×32 but costs accuracy at 64×64, where it falls below the
plain distribution-matching baseline. It is not recommended above 32×32.

---

## The released configuration

```yaml
loss:
  style_tap: conv          # read style before InstanceNorm removes it
  style_mode: persample    # unbiased with respect to batch size
  relative_style: true     # scale-free coefficients
  mm_ratio: 180.0
  cm_ratio: 10000.0
  icd:
    form: bounded
    content_ratio: 30.0
    style_ratio: 0.0
```

`configs/ours/_base.yaml`. The three coefficients were calibrated once on CIFAR10 and are
used unchanged on CIFAR100 and TinyImageNet.

## Implementation note on the ICD module

This release implements the ICD module in a **bounded, target-matched** form: the
intra-class spread of the condensed class is matched to the same statistic measured on the
real batch, rather than obtained by maximising the KL divergence to the nearest intra-class
neighbours as written in Eq. 8-9 of the paper.

The matched form is considerably more stable. It has an attainable optimum, so its weight
does not have to be tuned per dataset or budget, and it is therefore the default in every
shipped config. **The module's role in the pipeline is unchanged** -- it is still the
component that keeps the `ipc` images of a class from collapsing onto one prototype; only
the way that pressure is expressed differs.

Set `loss.icd.form: kl` to recover the Eq. 8-9 formulation exactly. `configs/paper/` does
this, so the published objective remains one command away:

```bash
python train.py --config configs/paper/cifar10_ipc10.yaml
```

The measured comparison between the two formulations is in [results.md](results.md).

## A note on scope

`configs/paper` reproduces the objective as published; `configs/ours` is the configuration
to build on. Where the two differ, [results.md](results.md) records the measured difference
rather than asserting one. In particular, the intra-class diversity module is neutral within
error bars on top of Style Matching at every resolution tested -- the measured gain in this
implementation comes from the Style Matching module. That is stated here because a reader
reproducing the ablation will find it, and should find it documented rather than surprising.

## References

- Bo Zhao, Hakan Bilen. *Dataset Condensation with Distribution Matching.* WACV 2023.
  <https://arxiv.org/abs/2110.04181> -- the L_MMD objective this method extends.
- Bo Zhao, Hakan Bilen. *Dataset Condensation with Differentiable Siamese Augmentation.*
  ICML 2021. <https://arxiv.org/abs/2102.08259> -- the DSA augmentation used throughout,
  in both the condensation loop and the evaluation protocol.
- Leon A. Gatys, Alexander S. Ecker, Matthias Bethge. *Image Style Transfer Using
  Convolutional Neural Networks.* CVPR 2016. <https://arxiv.org/abs/1508.06576> -- the Gram
  matrix as a style descriptor, which L_CM matches.
- Xun Huang, Serge Belongie. *Arbitrary Style Transfer in Real-time with Adaptive Instance
  Normalization.* ICCV 2017. <https://arxiv.org/abs/1703.06868> -- channel-wise moments as a
  style descriptor, which L_MM matches, and the reason reading style after InstanceNorm
  measures almost nothing.
