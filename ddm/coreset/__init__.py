"""Coreset selection baselines.

A coreset method keeps a subset of the *real* images instead of synthesising new ones, and
is the reference every condensation number is read against.  All of them share the budget
(``data.ipc`` images per class) and the evaluation protocol with the condensation methods,
so a coreset row and a DM row in the same table are directly comparable.

Available selectors -- the values ``coreset.selector`` accepts:

    random       uniform sample per class; no model needed
    herding      greedily match the class mean in feature space (Welling, 2009)
    kcenter      greedy minimax cover of the class (Sener & Savarese, 2018)
    kmeans       one example per k-means centroid of the class
    forgetting   most (or least) forgotten during proxy training (Toneva et al., 2019)
    uncertainty  least confident / highest entropy / smallest margin
    el2n         || softmax(f(x)) - onehot(y) ||_2 (Paul et al., 2021)
    grand        gradient-norm score, last-layer approximation (Paul et al., 2021)

Each is documented in full, with when it helps and when it does not, in docs/coreset.md.
"""
from ddm.coreset.base import ProxyStats, Selector
from ddm.coreset.forgetting import ForgettingSelector
from ddm.coreset.herding import HerdingSelector
from ddm.coreset.kcenter import KCenterGreedySelector
from ddm.coreset.kmeans import KMeansSelector
from ddm.coreset.random_selection import RandomSelector
from ddm.coreset.scores import EL2NSelector, GraNdSelector
from ddm.coreset.uncertainty import UncertaintySelector

SELECTORS = {s.name: s for s in (
    RandomSelector, HerdingSelector, KCenterGreedySelector, KMeansSelector,
    ForgettingSelector, UncertaintySelector, EL2NSelector, GraNdSelector)}


def build_selector(cfg, rng):
    """Instantiate the selector named by ``cfg.coreset.selector``."""
    name = cfg.coreset.selector
    if name not in SELECTORS:
        raise ValueError(f'unknown coreset.selector {name!r}; '
                         f'available: {sorted(SELECTORS)}')
    return SELECTORS[name](cfg, rng)


__all__ = ['SELECTORS', 'build_selector', 'Selector', 'ProxyStats']
