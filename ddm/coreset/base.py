"""Common scaffolding for coreset selection.

A selector answers one question: given the real training set and a budget of ``ipc``
images per class, which images do we keep?  Every method here answers it the same way --
score or greedily order the candidates of a class, then take the top ``ipc`` -- so the
budget arithmetic, the class bookkeeping and the statistics the methods share live here,
and a method only implements ``select_class``.
"""
import logging

import torch


class ProxyStats:
    """Per-example statistics of the real training set, measured with a proxy network.

    Only the fields a selector declares in ``requires`` are computed, so ``random`` costs
    nothing and ``herding`` does not pay for the forgetting bookkeeping.

    Attributes:
        labels:     (N,) int64 -- the training label of every example.
        features:   (N, D) float32 -- penultimate embedding, for herding/k-center/k-means.
        logits:     (N, K) float32 -- final-layer output, for the uncertainty scores.
        forgetting: (N,) int64 -- number of times an example went from correctly to
                    incorrectly classified during proxy training (Toneva et al., 2019).
        el2n:       (N,) float32 -- ||softmax(logits) - onehot||_2 (Paul et al., 2021).
        grand:      (N,) float32 -- last-layer gradient-norm approximation of GraNd.
    """

    def __init__(self, labels, features=None, logits=None, forgetting=None, el2n=None,
                 grand=None):
        self.labels = labels
        self.features = features
        self.logits = logits
        self.forgetting = forgetting
        self.el2n = el2n
        self.grand = grand

    def require(self, field, method):
        value = getattr(self, field, None)
        if value is None:
            raise RuntimeError(
                f"selector '{method}' needs proxy statistic '{field}', which was not "
                f"computed; this is a bug in the selector's `requires` declaration")
        return value


class Selector:
    """Base class: subclasses implement ``select_class``.

    Attributes:
        name:     the string used in ``coreset.selector``.
        requires: proxy statistics this method needs.  An empty tuple means no proxy
                  network is trained at all.
    """

    name = None
    requires = ()

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.opts = cfg.coreset
        self.rng = rng

    def select_class(self, class_idx, budget, stats):
        """Choose ``budget`` examples for one class.

        Args:
            class_idx: (n_c,) int64 tensor of the class's indices into the training set.
            budget:    number of examples to keep.
            stats:     the ProxyStats for the whole training set.

        Returns:
            (budget,) int64 tensor of indices into the training set, a subset of class_idx.
        """
        raise NotImplementedError

    def select(self, num_classes, labels, ipc, stats):
        """Run ``select_class`` over every class and concatenate the result.

        A class with fewer than ``ipc`` examples contributes all of them, with a warning,
        rather than failing the run.
        """
        chosen = []
        for c in range(num_classes):
            class_idx = (labels == c).nonzero(as_tuple=True)[0]
            budget = min(int(ipc), len(class_idx))
            if budget < int(ipc):
                logging.warning('class %d holds only %d examples, budget was %d',
                                c, len(class_idx), ipc)
            if budget == 0:
                continue
            picked = self.select_class(class_idx, budget, stats)
            picked = torch.as_tensor(picked, dtype=torch.long).reshape(-1)
            assert len(picked) == budget, (
                f'{self.name} returned {len(picked)} indices for class {c}, expected {budget}')
            chosen.append(picked)
        return torch.cat(chosen)


def top_k(class_idx, scores, budget, largest=True):
    """Take the ``budget`` examples of a class with the highest (or lowest) score."""
    s = torch.as_tensor(scores)[class_idx].float()
    order = torch.argsort(s, descending=largest)[:budget]
    return class_idx[order]


def pairwise_sq_dists(a, b):
    """Squared euclidean distances between the rows of ``a`` and of ``b``."""
    return torch.cdist(a.float(), b.float(), p=2) ** 2
