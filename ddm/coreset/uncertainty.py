"""Uncertainty selection -- keep the examples the proxy is least sure about.

Three classical scores over the proxy's softmax output p:

    least_confidence   1 - max_k p_k          how much probability the top class lacks
    entropy            -sum_k p_k log p_k     the spread of the whole distribution
    margin             p_(1) - p_(2)          the gap to the runner-up class

Higher uncertainty is kept for the first two, *lower* margin for the third -- the code
handles that sign for you, so ``coreset.uncertainty.metric: margin`` needs no other change.

The same caveat as forgetting applies: uncertainty ranks boundary and ambiguous images
highest, which is what an active-learning loop wants but not necessarily what a 10-image
budget wants.  ``coreset.uncertainty.order: ascending`` inverts the ranking.
"""
import torch
import torch.nn.functional as F

from ddm.coreset.base import Selector, top_k


class UncertaintySelector(Selector):
    name = 'uncertainty'
    requires = ('logits',)

    def select_class(self, class_idx, budget, stats):
        opts = self.opts.get('uncertainty', {})
        metric = opts.get('metric', 'entropy')
        probs = F.softmax(stats.require('logits', self.name).float(), dim=1)

        if metric == 'least_confidence':
            score, largest = 1.0 - probs.max(dim=1).values, True
        elif metric == 'entropy':
            score = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
            largest = True
        elif metric == 'margin':
            top2 = probs.topk(2, dim=1).values
            score, largest = top2[:, 0] - top2[:, 1], False   # small margin = uncertain
        else:
            raise ValueError(f"coreset.uncertainty.metric must be one of "
                             f"least_confidence/entropy/margin, got {metric!r}")

        if opts.get('order', 'descending') == 'ascending':
            largest = not largest
        return top_k(class_idx, score, budget, largest=largest)
