"""K-Center greedy (Sener & Savarese, 2018) -- cover the class, do not average it.

Solves the minimax facility-location problem: choose S so that every example is close to
some selected example,

    min_S  max_x  min_{s in S} || phi(x) - phi(s) ||

The greedy 2-approximation starts from the example nearest the class mean and then
repeatedly adds whichever example is currently *furthest* from everything already selected.

Where herding matches the first moment, this one matches the support.  The two fail in
opposite directions: herding concentrates on the mode and misses the tails, k-center walks
the tails and is therefore sensitive to outliers, since an unrepresentative image is by
construction the point furthest from the rest.
"""
import torch

from ddm.coreset.base import Selector, pairwise_sq_dists


class KCenterGreedySelector(Selector):
    name = 'kcenter'
    requires = ('features',)

    def select_class(self, class_idx, budget, stats):
        feats = stats.require('features', self.name)[class_idx].float()
        # Seed at the medoid rather than at random, so the run is deterministic and the
        # first centre is representative instead of extreme.
        mu = feats.mean(0, keepdim=True)
        first = int(torch.argmin(pairwise_sq_dists(feats, mu).squeeze(1)))
        chosen = [first]
        min_d = pairwise_sq_dists(feats, feats[first:first + 1]).squeeze(1)
        for _ in range(budget - 1):
            min_d[torch.tensor(chosen, dtype=torch.long)] = -1.0
            j = int(torch.argmax(min_d))
            chosen.append(j)
            min_d = torch.minimum(min_d,
                                  pairwise_sq_dists(feats, feats[j:j + 1]).squeeze(1))
        return class_idx[torch.tensor(chosen, dtype=torch.long)]
