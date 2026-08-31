"""Random selection -- the budget baseline every other method has to beat.

Draws ``ipc`` examples per class uniformly without replacement.  It needs no model, so it
is free, and it is unbiased: the selected set is an i.i.d. sample of the class, which means
it preserves the class distribution in expectation and only suffers from the variance of a
small sample.  That is a stronger baseline than it looks, and at small budgets it beats
several of the informed methods -- see docs/coreset.md.
"""
import torch

from ddm.coreset.base import Selector


class RandomSelector(Selector):
    name = 'random'
    requires = ()

    def select_class(self, class_idx, budget, stats):
        perm = torch.randperm(len(class_idx), generator=self.rng)[:budget]
        return class_idx[perm]
