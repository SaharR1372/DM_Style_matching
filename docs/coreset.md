# Coreset selection methods

A coreset method keeps a subset of the **real** training images. A condensation method
synthesises new ones. Both are given the same budget -- `data.ipc` images per class -- and
are scored by the same protocol, so the two families sit in one table and the comparison is
meaningful.

Every method here is run the same way:

```bash
python train.py --config configs/coreset/cifar10_ipc10_herding.yaml
```

and every one of them is a config away from any other:

```bash
python train.py --config configs/coreset/cifar10_ipc10_random.yaml \
    --set coreset.selector=kcenter data.ipc=50 output.save_path=runs/kcenter_ipc50
```

---

## How selection works here

Selection is **class-balanced**: each class independently contributes `ipc` images. This is
the convention in the condensation literature and it is what makes the budget comparable to
a condensed set, which is balanced by construction. A class with fewer than `ipc` examples
contributes all of them and logs a warning.

Six of the eight methods need a model's opinion about the training set. Rather than each
training its own, one **proxy network** is trained on the full training set and every
statistic is harvested from it (`ddm/coreset/proxy.py`):

| statistic | what it is | used by |
| --- | --- | --- |
| `features` | penultimate embedding of every training image | herding, kcenter, kmeans, grand |
| `logits` | final-layer output | uncertainty |
| `forgetting` | count of correct → incorrect transitions during training | forgetting |
| `el2n` | ‖softmax(f(x)) − onehot(y)‖₂ | el2n |
| `grand` | last-layer gradient-norm approximation | grand |

Only what the chosen selector declares in `requires` is computed, so `random` trains
nothing at all and `herding` does not pay for the forgetting bookkeeping.

The proxy is configured under `coreset.proxy` and is deliberately cheap -- 20 epochs of the
evaluation architecture by default. It is **re-trained for every repetition**
(`coreset.num_exp`). That is intentional: `herding`, `kcenter`, `forgetting`, `uncertainty`,
`el2n` and `grand` are deterministic *given* a proxy, so the spread across repetitions is
exactly the sensitivity of the method to which proxy it happened to get. That is a real
source of variance and reporting it is more honest than hiding it behind one fixed proxy.

---

## The methods

### `random` — the budget baseline

Draw `ipc` examples per class uniformly without replacement.

Costs nothing -- no proxy, no scoring -- and it is the baseline everything else has to beat.
It is also a much better baseline than it appears: an i.i.d. sample preserves the class
distribution in expectation, so its only weakness is the variance of a small sample. Several
of the informed methods below are *worse* than random at condensation-sized budgets, because
they are optimising a criterion that stops being the right one when the budget gets this
small.

**Config:** nothing beyond `selector: random`.

**Reference:** none -- it is the trivial baseline.

---

### `herding` — match the class mean in feature space

Greedily keep the running mean of the selected set as close as possible to the true class
mean in feature space. At step *k* the method adds

```
argmin_x  ‖ μ − (1/k)( Σ_{chosen} φ(x_j) + φ(x) ) ‖
```

which, after dropping the terms that do not depend on *x*, is the example whose feature is
most aligned with the residual `k·μ − Σ_chosen`. One matrix-vector product per step.

**Why it matters here.** Herding is the coreset counterpart of distribution matching: both
minimise a discrepancy between the class mean of the real data and of the small set. The
difference is the search space -- herding may only pick existing images, condensation may
synthesise them. The gap between a herding row and a DM row in the same table is therefore a
fairly clean measurement of what synthesis buys over selection.

**Failure mode.** Matching one moment says nothing about spread. Herding tends to pick
near-modal images, so the selected set is tight and unrepresentative of the class's tails --
the same degeneracy that motivates the Intra-Class Diversity module on the condensation side
(see [method.md](method.md)).

**Config:** nothing beyond `selector: herding`. Sensitive to `coreset.proxy.epochs`, since
the features are the proxy's.

**References:**
- Max Welling. *Herding Dynamical Weights to Learn.* ICML 2009.
  [[pdf]](https://icml.cc/Conferences/2009/papers/447.pdf)
- Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert.
  *iCaRL: Incremental Classifier and Representation Learning.* CVPR 2017.
  [[arXiv:1611.07725]](https://arxiv.org/abs/1611.07725) -- the paper that made herding
  the standard exemplar-selection baseline.

---

### `kcenter` — cover the class's support

Solve the minimax facility-location problem: choose *S* so that no example is far from every
selected example.

```
min_S  max_x  min_{s ∈ S}  ‖ φ(x) − φ(s) ‖
```

The greedy 2-approximation implemented here seeds at the example nearest the class mean --
deterministic, and representative rather than extreme -- then repeatedly adds whichever
example is currently *furthest* from everything already selected.

**Where herding matches the first moment, this matches the support.** The two fail in
opposite directions, which is what makes running both informative: herding concentrates on
the mode and misses the tails; k-center walks the tails and is therefore sensitive to
outliers, because an unrepresentative or mislabelled image is by construction the point
furthest from the rest. At `ipc = 10` almost the entire budget can go to the boundary of the
class.

**Config:** nothing beyond `selector: kcenter`.

**Reference:** Ozan Sener, Silvio Savarese. *Active Learning for Convolutional Neural
Networks: A Core-Set Approach.* ICLR 2018.
[[arXiv:1708.00489]](https://arxiv.org/abs/1708.00489)

---

### `kmeans` — one representative per mode

Run Lloyd's algorithm on the class's features with `ipc` clusters (k-means++ initialisation,
`coreset.kmeans.iterations` steps), then keep the example nearest each centroid. Centroids
are matched to examples greedily, tightest cluster first, so no image is used twice.

Sits between the previous two: herding matches the class mean, k-center covers the support,
k-means covers the **modes** with the density weighting neither of the others has. It is the
selection analogue of what the diversity term does on the condensation side -- spread the
budget over the class instead of piling it on the mode -- and it is the informed method that
most often beats random at small budgets.

**Config:** `coreset.kmeans.iterations` (default 50). Stochastic through k-means++, so it
genuinely varies across repetitions.

**References:**
- Stuart P. Lloyd. *Least Squares Quantization in PCM.* IEEE Transactions on Information
  Theory, 28(2):129-137, 1982. [[doi]](https://doi.org/10.1109/TIT.1982.1056489) -- the
  algorithm itself.
- David Arthur, Sergei Vassilvitskii. *k-means++: The Advantages of Careful Seeding.*
  SODA 2007. [[pdf]](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) -- the
  initialisation used here.

---

### `forgetting` — keep what the model keeps losing

While the proxy trains, watch every example. A **forgetting event** is a transition from
correctly to incorrectly classified between two consecutive presentations. Examples with
many events sit near the decision boundary and carry information the model struggles to
retain; examples with none are learned early and never revisited, and the original paper
shows a large fraction of them can be discarded with no loss of accuracy.

Examples never classified correctly at all record no transition, so they are assigned a
count above every observed one and are always kept -- the convention of the original work.

**Read this before using it at condensation budgets.** The method was designed for keeping
30–70% of a dataset. At `ipc = 10` the budget is 0.2% of CIFAR10, and "most forgotten" then
selects almost exclusively boundary and mislabelled images. Those are the examples that are
most informative *to a model that has already seen the rest of the data*, and close to the
worst possible set to train a network from scratch on. Set

```yaml
coreset:
  forgetting:
    order: ascending
```

to keep the *least* forgotten instead, which is the better choice at very small budgets and
is worth reporting alongside the default.

**Config:** `coreset.forgetting.order` (`descending` | `ascending`).

**Reference:** Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler,
Yoshua Bengio, Geoffrey J. Gordon. *An Empirical Study of Example Forgetting during Deep
Neural Network Learning.* ICLR 2019.
[[arXiv:1812.05159]](https://arxiv.org/abs/1812.05159)

---

### `uncertainty` — keep what the model is least sure about

Three classical scores over the proxy's softmax output *p*:

| `coreset.uncertainty.metric` | score | keeps |
| --- | --- | --- |
| `least_confidence` | 1 − max_k p_k | highest |
| `entropy` | −Σ_k p_k log p_k | highest |
| `margin` | p₍₁₎ − p₍₂₎ | **lowest** (a small margin means uncertain) |

The sign for `margin` is handled internally, so switching metric needs no other change.
`coreset.uncertainty.order: ascending` inverts the ranking for all three.

The same caveat as `forgetting` applies, and for the same reason: uncertainty ranks
ambiguous images highest, which is what an active-learning loop wants -- it already has a
model and wants the next label -- but not obviously what a 10-image-per-class budget wants,
since there is no model yet.

**Config:** `coreset.uncertainty.metric`, `coreset.uncertainty.order`.

**References:**
- David D. Lewis, William A. Gale. *A Sequential Algorithm for Training Text Classifiers.*
  SIGIR 1994. [[arXiv:cmp-lg/9407020]](https://arxiv.org/abs/cmp-lg/9407020) -- the origin
  of least-confidence uncertainty sampling.
- Burr Settles. *Active Learning Literature Survey.* Technical Report 1648, University of
  Wisconsin-Madison, 2009.
  [[pdf]](https://burrsettles.com/pub/settles.activelearning.pdf) -- Section 3.1 defines
  all three metrics used here, including margin sampling.

---

### `el2n` — the size of the error vector

Score each example by the norm of its error vector early in training:

```
EL2N(x, y) = ‖ softmax(f(x)) − onehot(y) ‖₂
```

Cheap, and a good approximation of the gradient norm once training has begun. Highest score
is kept by default: those are the examples still generating a learning signal.

**Averaging matters.** A single network's early scores are dominated by its own
initialisation. Set `coreset.proxy.num_models` to 3–10 to average over independently
initialised proxies; the shipped config uses 3. This is the main cost driver of the method.

**Config:** `coreset.el2n.order`, `coreset.proxy.num_models`, `coreset.proxy.epochs`
(the paper computes the score early -- around epoch 20 -- not at convergence).

**Reference:** Mansheej Paul, Surya Ganguli, Gintare Karolina Dziugaite. *Deep Learning on a
Data Diet: Finding Important Examples Early in Training.* NeurIPS 2021.
[[arXiv:2107.07075]](https://arxiv.org/abs/2107.07075)

---

### `grand` — the size of the gradient

The expected gradient-norm score, `E ‖ ∇_θ L(x, y) ‖₂`, computed here with the standard
last-layer approximation. For cross entropy the gradient with respect to the final linear
weights is `(p − y) hᵀ`, whose Frobenius norm factorises **exactly** as

```
‖ p − y ‖₂ · ‖ h ‖₂
```

with *h* the penultimate feature. So GraNd as implemented is EL2N reweighted by feature
magnitude, and the two rank differently only where representation norms vary a lot across
examples. The approximation is stated here because it is the difference between this
implementation and a full-backward one, which costs a backward pass per example.

**Config:** `coreset.grand.order`, `coreset.proxy.num_models`, `coreset.proxy.epochs`.

**Reference:** Mansheej Paul, Surya Ganguli, Gintare Karolina Dziugaite. *Deep Learning on a
Data Diet: Finding Important Examples Early in Training.* NeurIPS 2021.
[[arXiv:2107.07075]](https://arxiv.org/abs/2107.07075) -- same paper as EL2N; GraNd is its
gradient-norm sibling.

---

## Measured, on this repository's protocol

CIFAR10, 10 images/class, ConvNet. 3 independent selections × 5 evaluation networks =
**15 networks per row**; proxy trained for 20 epochs (3 proxies averaged for `el2n` and
`grand`). Produced by `configs/coreset/cifar10_ipc10_<selector>.yaml`, unmodified.

| selector | accuracy (%) | vs. random |
| --- | --- | --- |
| `kmeans` | **39.31 ± 0.62** | +6.16 |
| `herding` | 38.48 ± 0.64 | +5.33 |
| `random` | 33.15 ± 1.67 | — |
| `kcenter` | 19.84 ± 0.66 | −13.31 |
| `uncertainty` (entropy) | 11.80 ± 0.66 | −21.35 |
| `grand` | 11.20 ± 1.57 | −21.95 |
| `el2n` | 11.15 ± 0.39 | −22.00 |
| `forgetting` | 9.55 ± 0.65 | −23.60 |

For scale, condensation on the same protocol reaches 48.79 (plain DM) and 51.05 (the
released method) -- see [results.md](results.md).

Three things this table shows, all of them predicted by the discussions above:

1. **The representativeness methods work; the difficulty methods do not.** `kmeans` and
   `herding` -- the two that try to *represent* the class -- beat random by 5-6 points.
   Every difficulty-ranked method (`forgetting`, `el2n`, `grand`, `uncertainty`) lands near
   the 10% chance level, because at 10 images per class "keep the hardest examples" selects
   ten boundary or mislabelled images and nothing else. They are being used far outside the
   30-70% pruning regime they were designed for; try `order: ascending` before concluding
   anything about the methods themselves.
2. **`kcenter` sits in between, and for the reason given above:** matching the support means
   walking the tails, and at this budget most of the budget goes to the tails.
3. **Random is a real baseline.** Five of the seven informed methods lose to it.

Rerun any row with one command, or all of them:

```bash
for s in random herding kcenter kmeans forgetting uncertainty el2n grand; do
    python train.py --config configs/coreset/cifar10_ipc10_$s.yaml
done
python scripts/collect_results.py runs
```

---

## Choosing between them

| if you want | use |
| --- | --- |
| the honest floor for any table | `random` |
| the closest selection analogue of distribution matching | `herding` |
| coverage of the class, outliers included | `kcenter` |
| coverage of the class's modes | `kmeans` |
| the published difficulty rankings | `forgetting`, `el2n`, `grand`, `uncertainty` |

Two things are worth knowing before reading too much into a difficulty ranking at
condensation budgets:

1. **Difficulty-ranked methods were designed for large budgets.** Their published regime is
   pruning 30–70% of a dataset, where "drop the easy examples" is safe because thousands of
   easy examples remain. At `ipc = 10` there are ten, and they carry the whole class. Try
   `order: ascending` before concluding a method is bad.
2. **Everything downstream of the proxy inherits the proxy's blind spots.** `coreset.proxy`
   controls how good that model is; a 1-epoch proxy makes every informed method approximately
   random. If an informed method ties with `random`, check `coreset.proxy.epochs` first.

## Not implemented

Deliberately, to keep the baselines classical and cheap: submodular selection (CRAIG,
GLISTER), contextual diversity, bilevel and gradient-matching selection, and anything that
requires unrolled training. Those are closer to condensation than to coreset selection, and
this repository's condensation side is the place to compare against them.

## References

- Max Welling. *Herding Dynamical Weights to Learn.* ICML 2009.
  <https://icml.cc/Conferences/2009/papers/447.pdf>
- Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert.
  *iCaRL: Incremental Classifier and Representation Learning.* CVPR 2017.
  <https://arxiv.org/abs/1611.07725>
- Ozan Sener, Silvio Savarese. *Active Learning for Convolutional Neural Networks: A
  Core-Set Approach.* ICLR 2018. <https://arxiv.org/abs/1708.00489>
- Stuart P. Lloyd. *Least Squares Quantization in PCM.* IEEE Trans. Information Theory,
  28(2):129-137, 1982. <https://doi.org/10.1109/TIT.1982.1056489>
- David Arthur, Sergei Vassilvitskii. *k-means++: The Advantages of Careful Seeding.*
  SODA 2007. <https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf>
- Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio,
  Geoffrey J. Gordon. *An Empirical Study of Example Forgetting during Deep Neural Network
  Learning.* ICLR 2019. <https://arxiv.org/abs/1812.05159>
- David D. Lewis, William A. Gale. *A Sequential Algorithm for Training Text Classifiers.*
  SIGIR 1994. <https://arxiv.org/abs/cmp-lg/9407020>
- Burr Settles. *Active Learning Literature Survey.* TR 1648, University of
  Wisconsin-Madison, 2009. <https://burrsettles.com/pub/settles.activelearning.pdf>
- Mansheej Paul, Surya Ganguli, Gintare Karolina Dziugaite. *Deep Learning on a Data Diet:
  Finding Important Examples Early in Training.* NeurIPS 2021.
  <https://arxiv.org/abs/2107.07075>
