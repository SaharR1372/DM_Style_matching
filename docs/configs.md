# Config reference

Every run is defined by exactly one YAML file, so a result is reproducible from that file
alone. The resolved config -- after inheritance and command-line overrides -- is written to
`<output.save_path>/config.resolved.yaml` at the start of each run, which is the file to
keep if you want to repeat it.

```bash
python train.py --config configs/ours/cifar10_ipc10.yaml
python train.py --config configs/ours/cifar10_ipc10.yaml --set data.ipc=1 condense.iterations=5000
```

## Inheritance

A config may name one parent with `inherit:`, resolved relative to the file's own directory.
The child's keys are merged over the parent's, recursively, one leaf at a time -- so a child
that sets `loss.mm_ratio` keeps every other key under `loss`. Chains are allowed and
`configs/ours/cifar10_ipc10.yaml` uses one:

```
configs/base.yaml           shared defaults: data, model, eval, output
  └─ configs/dm/_base.yaml       method: ddm, condense + loss, all weights zero
       └─ configs/ours/_base.yaml    the released loss weights
            └─ configs/ours/cifar10_ipc10.yaml   dataset, budget, protocol
```

Files whose name begins with `_` are bases and are not meant to be run directly.

## Overrides

`--set key.path=value` applies after inheritance. Values are parsed as YAML, so types work
as expected:

```bash
--set data.ipc=50                       # int
--set loss.mm_ratio=180.0               # float
--set loss.relative_style=false         # bool
--set eval.models=[ConvNet,AlexNet]     # list
--set coreset.proxy.arch=null           # null
```

Overrides are for sweeps. A run you intend to report should get its own config file.

---

## Schema

### `method` *(required)*

`ddm` -- synthesise a condensed set by distribution matching.
`coreset` -- select real images. See [coreset.md](coreset.md).

### `name`

Label recorded in `results.json`. Defaults to the method name.

### `data` *(required)*

| key | meaning |
| --- | --- |
| `dataset` | `MNIST` \| `FashionMNIST` \| `SVHN` \| `CIFAR10` \| `CIFAR100` \| `TinyImageNet` \| `ImageNet` |
| `data_path` | where datasets live; torchvision downloads here. See [datasets.md](datasets.md) |
| `ipc` | images per class -- the budget. Shared by both methods |

### `model` *(required)*

| key | meaning |
| --- | --- |
| `arch` | architecture the small set is built with. `ConvNet_style` is `ConvNet` plus the ability to expose feature maps |
| `net_depth` | ConvNet depth. `0` = auto: 4 at 64×64 and above (TinyImageNet), 3 below, matching the DM literature |

### `condense` — `method: ddm` only *(required)*

| key | default | meaning |
| --- | --- | --- |
| `iterations` | 20000 | optimisation steps on the synthetic images |
| `lr_img` | 1.0 | SGD learning rate on the pixels |
| `momentum` | 0.5 | SGD momentum on the pixels |
| `batch_real` | 256 | real images sampled per class per iteration |
| `init` | `real` | `real` \| `noise` -- how the synthetic set is initialised |
| `num_exp` | 3 | independent condensation runs, seeded `seed`, `seed+1`, ... |
| `seed` | 0 | base seed |
| `class_chunk` | 10 | classes fused into one forward pass. Numerically identical for InstanceNorm networks; lower it if memory is tight |
| `dsa_strategy` | `color_crop_cutout_flip_scale_rotate` | DSA augmentation; `none` disables |

### `loss` — `method: ddm` only *(required)*

| key | default | meaning |
| --- | --- | --- |
| `style_tap` | `norm` | where the style is read: `conv` \| `norm` \| `act` \| `pool` |
| `style_mode` | `batchavg` | moments estimator: `batchavg` (published) \| `persample` (released) |
| `relative_style` | false | normalise style terms by the magnitude of the real target |
| `legacy_style_accum` | false | reproduce the released scripts' un-reset style accumulator |
| `mm_ratio` | 0.0 | weight of L_MM |
| `cm_ratio` | 0.0 | weight of L_CM |
| `icd.form` | `bounded` | `bounded` (released) \| `kl` (published Eq. 8-9) |
| `icd.content_ratio` | 0.0 | weight of L_CD, the content component |
| `icd.style_ratio` | 0.0 | weight of L_SD, the style component |
| `icd.rank` | 0 | principal directions matched by L_CD; 0 = `min(ipc-1, 16)` |
| `icd.k` | −1 | `kl` form only: neighbours; −1 = `0.2 · ipc` |

A weight of zero removes its term entirely, including the cost of computing it. See
[method.md](method.md).

### `coreset` — `method: coreset` only *(required)*

| key | default | meaning |
| --- | --- | --- |
| `selector` | — | `random` \| `herding` \| `kcenter` \| `kmeans` \| `forgetting` \| `uncertainty` \| `el2n` \| `grand` |
| `num_exp` | 1 | independent selections, each with a freshly trained proxy |
| `seed` | 0 | base seed |
| `proxy.arch` | null | proxy architecture; null = `model.arch` |
| `proxy.epochs` | 20 | proxy training epochs |
| `proxy.batch_size` | 256 | |
| `proxy.lr` / `momentum` / `weight_decay` | 0.01 / 0.9 / 5e-4 | proxy SGD |
| `proxy.num_models` | 1 | proxies to average; raise for `el2n` and `grand` |
| `proxy.workers` | 4 | dataloader workers |
| `proxy.dsa_strategy` | DSA | augmentation during proxy training |
| `forgetting.order` | `descending` | `descending` keeps the most forgotten |
| `uncertainty.metric` | `entropy` | `least_confidence` \| `entropy` \| `margin` |
| `uncertainty.order` | `descending` | |
| `el2n.order` / `grand.order` | `descending` | |
| `kmeans.iterations` | 50 | Lloyd steps |

### `eval` *(required)*

The protocol every method is scored by: train `num_eval` freshly initialised networks from
scratch on the small set and test each on the real test set.

| key | default | meaning |
| --- | --- | --- |
| `models` | `[ConvNet]` | architectures to evaluate on. More than one gives a cross-architecture row |
| `num_eval` | 5 | networks per architecture per repetition |
| `epochs` | 1000 | training epochs per evaluation network |
| `lr_net` | 0.01 | SGD; momentum 0.9, weight decay 5e-4, ÷10 halfway |
| `batch_train` | 256 | |
| `dsa_strategy` | DSA | augmentation during evaluation training |
| `every` | 0 | `ddm` only: also evaluate every N iterations. 0 = only at the end |

Reported accuracy pools `num_exp × num_eval` networks.

### `output` *(required)*

| key | default | meaning |
| --- | --- | --- |
| `save_path` | — | directory for the checkpoint, log, PNG grid and `results.json` |
| `save_images` | true | write a PNG grid of the small set |

---

## What a run writes

```
<save_path>/
    config.resolved.yaml                   the exact config, after inheritance and --set
    train.log                              full log
    results.json                           list of run summaries, appended to
    condensed_<dataset>_<arch>_<ipc>ipc.pt   ddm: images, labels, accuracies, config
    coreset_<dataset>_<selector>_<ipc>ipc.pt coreset: the same, plus the chosen indices
    vis_*.png                              the small set as an image grid
```

`results.json` is a list, so pointing several runs at one `save_path` accumulates the rows
of an ablation table in one file.

---

## Mapping from the previous layout

The `dev` branch keeps the earlier flat scripts. Each is now a config:

| previous script | equivalent config |
| --- | --- |
| `DM_DDM.py --preset dm` | `configs/dm/cifar10_ipc10.yaml` |
| `DM_DDM.py --preset ours` | `configs/ours/cifar10_ipc10.yaml` |
| `DM_DDM.py --preset paper` | `configs/paper/cifar10_ipc10.yaml` |
| `DM_DDM.py --preset legacy` | `configs/paper/legacy_cifar10_ipc10.yaml` |
| `DM_MeanStd_Matching.py` | `configs/ablation/mm_only_cifar10_ipc10.yaml` |
| `DM_GramMatching.py` | `configs/ablation/cm_only_cifar10_ipc10.yaml` |
| `DM_KNearest.py` | `configs/ablation/icd_only_cifar10_ipc10.yaml` |

Command-line flags map onto config keys directly: `--ipc` → `data.ipc`, `--Iteration` →
`condense.iterations`, `--mm_ratio` → `loss.mm_ratio`, `--icd_ratio` →
`loss.icd.content_ratio`, `--icd_style_ratio` → `loss.icd.style_ratio`, `--icd_form` →
`loss.icd.form`, `--save_path` → `output.save_path`.
