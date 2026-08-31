"""K-means selection -- one representative per mode of the class.

Runs Lloyd's algorithm on the class's features with ``ipc`` clusters, then keeps the example
nearest each centroid.  It sits between herding and k-center: herding matches the class
mean, k-center covers the class's support, k-means covers its *modes* with the density
weighting that neither of the other two has.

Centroids are initialised with k-means++ so the result does not depend on an arbitrary first
pick, and the run is deterministic given ``coreset.seed``.
"""
import torch

from ddm.coreset.base import Selector, pairwise_sq_dists


class KMeansSelector(Selector):
    name = 'kmeans'
    requires = ('features',)

    def _kmeanspp_init(self, feats, k):
        n = feats.shape[0]
        first = int(torch.randint(n, (1,), generator=self.rng))
        centres = [feats[first]]
        d2 = pairwise_sq_dists(feats, feats[first:first + 1]).squeeze(1)
        for _ in range(k - 1):
            probs = d2.clamp_min(0)
            probs = probs / probs.sum() if float(probs.sum()) > 0 else None
            j = (int(torch.multinomial(probs, 1, generator=self.rng)) if probs is not None
                 else int(torch.randint(n, (1,), generator=self.rng)))
            centres.append(feats[j])
            d2 = torch.minimum(d2, pairwise_sq_dists(feats, feats[j:j + 1]).squeeze(1))
        return torch.stack(centres)

    def select_class(self, class_idx, budget, stats):
        feats = stats.require('features', self.name)[class_idx].float()
        iters = int(self.opts.get('kmeans', {}).get('iterations', 50))
        centres = self._kmeanspp_init(feats, budget)

        for _ in range(iters):
            assign = pairwise_sq_dists(feats, centres).argmin(dim=1)
            new = centres.clone()
            for j in range(budget):
                members = feats[assign == j]
                if len(members):
                    new[j] = members.mean(0)
            if torch.allclose(new, centres):
                break
            centres = new

        # One example per centroid, without repeats: assign greedily by distance so a
        # centroid whose nearest example is already taken falls back to its next nearest.
        d = pairwise_sq_dists(feats, centres)                 # (n_c, budget)
        taken, chosen = set(), []
        for j in torch.argsort(d.min(dim=0).values):          # tightest cluster first
            for i in torch.argsort(d[:, j]):
                i = int(i)
                if i not in taken:
                    taken.add(i)
                    chosen.append(i)
                    break
        return class_idx[torch.tensor(chosen, dtype=torch.long)]
