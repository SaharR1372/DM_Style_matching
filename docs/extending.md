# Extending the code

The four things most likely to be added, and the one place each of them goes.

## A new loss term

1. Write it in `ddm/losses/style.py` or `ddm/losses/diversity.py` (or a new module there),
   taking tensors and returning a scalar. Export it from `ddm/losses/__init__.py`.
2. Add its weight to `configs/dm/_base.yaml` under `loss:`, defaulting to `0.0`, so every
   existing config keeps its current behaviour.
3. Consume it in `ddm/engine/condense.py` inside the per-class loop, guarded by
   `if loss_cfg.get('your_ratio'):` and accumulated into `parts` for the log line.
4. If it reads the intermediate feature maps rather than the final embedding, add it to
   `_needs_style` so the feature maps are actually collected.

**One design note worth reading first.** Every term added in this work is *bounded and
target-matched*: it compares a synthetic statistic against the same statistic measured on
the real batch, and is normalised by the magnitude of that target. That gives it an
attainable optimum, makes it scale-free, and means its weight transfers across datasets
without retuning. A term that instead *maximises* something unbounded has no weight at which
it stops -- see the L_ICD discussion in [method.md](method.md) for what that costs in
practice.

## A new coreset selector

1. Add a module under `ddm/coreset/` with a `Selector` subclass:

   ```python
   class MySelector(Selector):
       name = 'mine'
       requires = ('features',)          # what the proxy must compute

       def select_class(self, class_idx, budget, stats):
           feats = stats.require('features', self.name)[class_idx]
           ...
           return class_idx[chosen]      # (budget,) indices into the training set
   ```

2. Register it in the `SELECTORS` dict in `ddm/coreset/__init__.py`. Config validation and
   `--set coreset.selector=mine` pick it up automatically.
3. If it needs a statistic the proxy does not yet compute, add it to `ProxyStats` and to
   `collect_stats` in `ddm/coreset/proxy.py`, gated on its name appearing in `requires` so
   other selectors do not pay for it.
4. Add a config under `configs/coreset/` and a section to [coreset.md](coreset.md).

Per-class budgeting, the "class has fewer examples than the budget" case, and the assertion
that a selector returns the right number of indices are all handled by the base class.

## A new dataset

One `elif` branch in `get_dataset` (`ddm/data.py`) returning

```python
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader
```

Nothing else changes: the ConvNet depth heuristic keys off `im_size`, and both the
condensation loop and every selector are written against this tuple. See
[datasets.md](datasets.md).

## A new architecture

Define it in `ddm/networks.py` and add a branch to `get_network` in `ddm/models.py`. Two
requirements:

- **`embed(x)`** must return the penultimate feature, since that is what L_MMD matches and
  what the feature-based selectors read.
- To be usable as a *condensation* architecture with a style term, `embed` should return
  `(embedding, feature_maps)` and the class should expose `set_style_tap(tap)`. See
  `ConvNet_style` for the pattern. An architecture without those can still be used for
  evaluation (`eval.models`) and as a coreset proxy.

## Running a sweep

`--set` exists for this. Anything in the config can be overridden, so a sweep is a shell
loop and needs no new script:

```bash
for w in 30 100 300; do
    python train.py --config configs/ours/cifar10_ipc10.yaml \
        --set loss.icd.content_ratio=$w output.save_path=runs/sweep name=icd_$w
done
python scripts/collect_results.py runs/sweep
```

Pointing every run at one `save_path` accumulates the rows in a single `results.json`, which
`scripts/collect_results.py` renders as a markdown table.
