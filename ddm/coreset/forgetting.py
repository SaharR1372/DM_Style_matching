"""Forgetting events (Toneva et al., 2019) -- keep the examples the model keeps losing.

While a network trains on the full dataset, each example is watched: a *forgetting event*
is a transition from correctly to incorrectly classified between two consecutive
presentations.  Examples with many events sit near the decision boundary and carry the
information the model has trouble retaining; examples with none are learned early and never
revisited, and the original paper shows a large fraction of them can be discarded with no
loss of accuracy.

Examples that are never classified correctly at all record no transition, so they are
assigned a count above every observed one and are always kept -- they are the hardest, and
this is the convention of the original work.

Caveat at condensation budgets.  The method is designed for keeping 30-70% of a dataset.
At ``ipc = 10`` the budget is 0.2% of CIFAR10, and "most forgotten" then selects almost
exclusively boundary and mislabelled images, which train a network from scratch poorly.
Setting ``coreset.forgetting.order: ascending`` selects the *least* forgotten instead, which
is the better choice at very small budgets; see docs/coreset.md.
"""
from ddm.coreset.base import Selector, top_k


class ForgettingSelector(Selector):
    name = 'forgetting'
    requires = ('forgetting',)

    def select_class(self, class_idx, budget, stats):
        counts = stats.require('forgetting', self.name)
        order = self.opts.get('forgetting', {}).get('order', 'descending')
        return top_k(class_idx, counts, budget, largest=(order == 'descending'))
