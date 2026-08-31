# Measured results

Every number here was produced by this code, with the config named in its row. Accuracy is
the mean and standard deviation over `num_exp × num_eval` networks trained from scratch on
the condensed set and tested on the real test set, all evaluated on `ConvNet`.

Protocols differ by dataset, and are stated per table. They are the protocol each shipped
config declares, so a row can be reproduced with one command.

---

## CIFAR10, 10 images/class

20 000 iterations, 3 condensation runs × 10 evaluation networks = **30 networks per row**.

| config | objective | accuracy (%) |
| --- | --- | --- |
| `configs/dm/cifar10_ipc10.yaml` | L_MMD only (baseline) | 48.79 ± 0.87 |
| `configs/paper/cifar10_ipc10.yaml` | L_MMD + L_MM + L_ICD(kl) | 45.13 ± 0.98 |
| `configs/paper/legacy_cifar10_ipc10.yaml` | as above, un-reset accumulator | 45.26 ± 0.98 |
| `configs/ablation/sm_only_cifar10_ipc10.yaml` | L_MMD + L_MM + L_CM | **51.13 ± 0.66** |
| `configs/ablation/icd_style_cifar10_ipc10.yaml` | + L_SD | 51.13 ± 0.50 |
| `configs/ours/cifar10_ipc10.yaml` | + L_CD *(released)* | 51.05 ± 0.43 |
| `configs/ablation/icd_both_cifar10_ipc10.yaml` | + L_CD + L_SD | 50.56 ± 0.51 |

## CIFAR100, 10 images/class

10 000 iterations, 1 condensation run × 10 evaluation networks = **10 networks per row**.

| config | objective | accuracy (%) |
| --- | --- | --- |
| `configs/dm/cifar100_ipc10.yaml` | L_MMD only | 29.32 ± 0.29 |
| `configs/paper` (CIFAR100) | L_MMD + L_MM + L_ICD(kl) | 26.23 ± 0.36 |
| `sm_only` (CIFAR100) | L_MMD + L_MM + L_CM | **30.57 ± 0.37** |
| `configs/ours/cifar100_ipc10.yaml` | + L_CD *(released)* | 30.44 ± 0.17 |
| `icd_style` (CIFAR100) | + L_SD | 30.33 ± 0.38 |

## TinyImageNet, 10 images/class

3 000 iterations, 2 condensation runs × 5 evaluation networks = **10 networks per row**.
ConvNetD4, resolved automatically from the 64×64 resolution.

| config | objective | accuracy (%) |
| --- | --- | --- |
| `configs/dm/tinyimagenet_ipc10.yaml` | L_MMD only | 14.19 ± 0.23 |
| `sm_only` (TinyImageNet) | L_MMD + L_MM + L_CM | **14.87 ± 0.30** |
| `configs/ours/tinyimagenet_ipc10.yaml` | + L_CD *(released)* | 14.73 ± 0.40 |
| `icd_style` (TinyImageNet) | + L_SD | 13.22 ± 0.34 |

## Other CIFAR10 budgets

20 000 iterations, 30 networks per row.

| ipc | `configs/dm` | `configs/paper` |
| --- | --- | --- |
| 1 | 26.14 ± 0.87 | 25.67 ± 0.97 |
| 50 | 62.87 ± 0.37 | 43.48 ± 0.48 |

`configs/ours` has not been measured at ipc = 1 or ipc = 50; the shipped configs for those
budgets are the released settings applied unchanged, not a measured result.

---

## What the tables say

**1. The gain comes from the Style Matching module.** Adding L_MM + L_CM to the baseline is
worth +2.34 on CIFAR10, +1.25 on CIFAR100 and +0.68 on TinyImageNet. Every row that beats
the baseline contains it, and no row without it does.

**2. The Intra-Class Diversity module is neutral on top of it.** Against the `sm_only`
control, the content component measures −0.08 (CIFAR10), −0.13 (CIFAR100) and −0.14
(TinyImageNet) -- inside the error bar at every resolution. The style component is level at
32×32 and costs 1.65 at 64×64, which is why `style_ratio` defaults to 0. Enabling both
components together is slightly worse than either alone: they constrain the same
second-order intra-class structure and compete for the same headroom rather than adding.

This is documented rather than hidden because anyone reproducing the ablation will find it.
The module is retained, at the weight measured to be safest, because it is part of the
published method and because it is the right place to build on -- see the discussion of
bounded versus unbounded formulations in [method.md](method.md).

**3. The gain shrinks as the data gets harder.** +2.34 → +1.25 → +0.68 across CIFAR10,
CIFAR100 and TinyImageNet. Second-order feature statistics buy less where the gap to real
data is largest, which is worth knowing before extending the method to harder datasets.

**4. On reproducing the published objective.** `configs/paper` implements Eq. 8-9 as
written, and in this implementation it measures below the plain DM baseline at every budget
tested, with the deficit widening at ipc = 50. The cause is structural rather than a bug:
the published L_ICD **maximises** an unbounded KL divergence, so it has no attainable
optimum and no weight at which it both spreads the samples and stops. Its coefficient has to
be retuned for each dataset and budget, and a value calibrated at ipc = 10 is far too strong
at ipc = 50. The released implementation replaces it with the bounded, target-matched form
(`loss.icd.form: bounded`), whose optimum is read off the real data and whose weight
therefore transfers unchanged. `form: kl` is kept so the published objective can still be
run.

---

## Coreset baselines

The eight selectors in [coreset.md](coreset.md) ship with configs but without a results
table -- they are provided so a reader can measure them under this repository's exact
protocol rather than importing numbers from another paper's setup. One command each:

```bash
for s in random herding kcenter kmeans forgetting uncertainty el2n grand; do
    python train.py --config configs/coreset/cifar10_ipc10_$s.yaml
done
python scripts/collect_results.py runs --out results_table.md
```

## Reproducing

```bash
python train.py --config configs/ours/cifar10_ipc10.yaml       # the released method
python train.py --config configs/dm/cifar10_ipc10.yaml         # the baseline
python train.py --config configs/paper/cifar10_ipc10.yaml      # the published objective
```

Each writes its summary to `<save_path>/results.json`. Point several runs at one
`save_path` and the file accumulates the rows of the table.
