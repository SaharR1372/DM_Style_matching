"""Decomposed Distribution Matching in Dataset Condensation (WACV 2025).

The package is organised around three things a user actually runs:

    ddm.engine.condense   synthesise a condensed set by distribution matching
    ddm.coreset           select a coreset from the real data (random, herding, ...)
    ddm.engine.evaluator  train networks on either kind of small set and test them

Everything is driven by a YAML config; see ddm.config and the configs/ directory.
"""

__version__ = '1.0.0'
