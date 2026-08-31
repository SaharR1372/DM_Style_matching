"""Herding (Welling, 2009; as used by iCaRL) -- match the class mean in feature space.

Greedy: keep the running mean of the examples chosen so far as close as possible to the
true class mean.  At step k the method adds

    argmin_x  || mu  -  (1/k) ( sum_{chosen} phi(x_j)  +  phi(x) ) ||

which reduces to picking the example whose feature is most aligned with the residual
``k * mu - sum_chosen``, so each step is one matrix-vector product.

It is the natural coreset counterpart of distribution matching: both minimise a discrepancy
between the class mean of the real data and of the small set.  The difference is the search
space -- herding may only choose existing images, while condensation may synthesise them --
which is exactly the comparison the tables in the paper are making.
"""
import torch

from ddm.coreset.base import Selector


class HerdingSelector(Selector):
    name = 'herding'
    requires = ('features',)

    def select_class(self, class_idx, budget, stats):
        feats = stats.require('features', self.name)[class_idx].float()
        mu = feats.mean(0)
        chosen, running = [], torch.zeros_like(mu)
        mask = torch.zeros(len(class_idx), dtype=torch.bool)
        for k in range(1, budget + 1):
            # Maximising <phi(x), k*mu - running> is equivalent to minimising the norm of
            # the residual after adding x, with the terms not depending on x dropped.
            score = feats @ (k * mu - running)
            score[mask] = float('-inf')
            j = int(torch.argmax(score))
            mask[j] = True
            running = running + feats[j]
            chosen.append(j)
        return class_idx[torch.tensor(chosen, dtype=torch.long)]
