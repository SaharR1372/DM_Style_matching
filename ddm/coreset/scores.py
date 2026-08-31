"""EL2N and GraNd (Paul, Ganguli & Dziugaite, 2021) -- prune by how much a sample teaches.

Both score an example by the size of the learning signal it produces early in training, and
both are averaged over ``coreset.proxy.num_models`` independently initialised proxies, which
is what makes them stable -- a single network's early scores are dominated by its own
initialisation.

    EL2N   || softmax(f(x)) - onehot(y) ||_2
           The error vector's norm.  Cheap, and a good approximation of GraNd once training
           has begun.

    GraNd  E || grad_theta L(x, y) ||_2
           Computed here with the standard last-layer approximation: for cross entropy the
           gradient with respect to the final linear weights is (p - y) h^T, whose Frobenius
           norm factorises exactly as ||p - y||_2 * ||h||_2 with h the penultimate feature.
           So GraNd is EL2N weighted by feature magnitude, and the two rank differently only
           where the representation norms differ a lot across examples.

Highest score is kept by default: those are the examples still generating gradient.  As with
the other difficulty-ranked methods, ``order: ascending`` inverts it, which is worth trying
at condensation-sized budgets.
"""
from ddm.coreset.base import Selector, top_k


class _ScoreSelector(Selector):
    field = None

    def select_class(self, class_idx, budget, stats):
        score = stats.require(self.field, self.name)
        order = self.opts.get(self.name, {}).get('order', 'descending')
        return top_k(class_idx, score, budget, largest=(order == 'descending'))


class EL2NSelector(_ScoreSelector):
    name = 'el2n'
    requires = ('el2n',)
    field = 'el2n'


class GraNdSelector(_ScoreSelector):
    name = 'grand'
    requires = ('grand', 'features')
    field = 'grand'
