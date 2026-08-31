"""YAML configuration: loading, inheritance, command-line overrides and validation.

Every entry point takes exactly one config file, so a run is reproducible from that file
alone.  Configs compose through a single ``inherit:`` key, which names another config
resolved relative to the inheriting file; the child's keys are merged over the parent's,
one nested level at a time.  ``configs/base.yaml`` holds the defaults that every method
shares, so a method config only has to state what it changes.

The schema is documented in docs/configs.md.
"""
import copy
import os

import yaml


class Config(dict):
    """A dict whose keys are also attributes, recursively.

    Nested sections are what a config is mostly made of, so ``cfg.loss.mm_ratio`` reads
    better than ``cfg['loss']['mm_ratio']`` at the twenty-odd places the loss weights are
    consumed.  It stays a dict, so it still serialises with ``yaml.safe_dump``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, Config):
                self[k] = Config(v)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"config has no key '{name}'; the available keys here are "
                f"{sorted(self.keys())}") from None

    def __setattr__(self, name, value):
        self[name] = Config(value) if isinstance(value, dict) else value

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self.items()}

    def dump(self, path):
        """Write the fully resolved config next to the run's outputs."""
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)


def _merge(base, over):
    """Recursively merge ``over`` into ``base``; ``over`` wins on every leaf."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path, _seen=None):
    path = os.path.abspath(path)
    _seen = _seen or []
    if path in _seen:
        raise ValueError('circular inherit: ' + ' -> '.join(_seen + [path]))
    if not os.path.exists(path):
        raise FileNotFoundError(f'config not found: {path}')
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop('inherit', None)
    if parent is None:
        return raw
    parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
    return _merge(_load_raw(parent_path, _seen + [path]), raw)


def _apply_override(cfg, item):
    """Apply one ``a.b.c=value`` override in place, parsing the value as YAML."""
    if '=' not in item:
        raise ValueError(f"--set expects key=value, got '{item}'")
    key, _, value = item.partition('=')
    node, parts = cfg, key.strip().split('.')
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = yaml.safe_load(value)


def load_config(path, overrides=None):
    """Load ``path``, resolve ``inherit``, apply ``--set`` overrides, validate.

    Args:
        path: path to the YAML config.
        overrides: sequence of ``key.path=value`` strings; values are parsed as YAML, so
            ``loss.mm_ratio=180`` gives a float and ``eval.models=[ConvNet,AlexNet]`` a list.
    """
    raw = _load_raw(path)
    for item in overrides or []:
        _apply_override(raw, item)
    cfg = Config(raw)
    cfg.config_path = os.path.abspath(path)
    validate(cfg)
    return cfg


METHODS = ('ddm', 'coreset')


def validate(cfg):
    """Fail on a malformed config before a run spends an hour discovering it.

    Checks only what is cheap and unambiguous -- the keys every code path dereferences,
    and the enumerations a typo would otherwise turn into a silent behaviour change.
    """
    if cfg.get('method') not in METHODS:
        raise ValueError(f"config.method must be one of {METHODS}, got {cfg.get('method')!r}")
    for section in ('data', 'model', 'eval', 'output'):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"config is missing the '{section}' section")
    for key in ('dataset', 'data_path', 'ipc'):
        if key not in cfg.data:
            raise ValueError(f"config.data.{key} is required")
    if int(cfg.data.ipc) < 1:
        raise ValueError('config.data.ipc must be >= 1')
    if 'arch' not in cfg.model:
        raise ValueError('config.model.arch is required')
    if 'save_path' not in cfg.output:
        raise ValueError('config.output.save_path is required')

    if cfg.method == 'ddm':
        if 'condense' not in cfg or 'loss' not in cfg:
            raise ValueError("method 'ddm' needs both a 'condense' and a 'loss' section")
        tap = cfg.loss.get('style_tap', 'norm')
        if tap not in ('norm', 'conv', 'act', 'pool'):
            raise ValueError(f"loss.style_tap must be one of norm/conv/act/pool, got {tap!r}")
        mode = cfg.loss.get('style_mode', 'batchavg')
        if mode not in ('batchavg', 'persample'):
            raise ValueError(f"loss.style_mode must be batchavg or persample, got {mode!r}")
        init = cfg.condense.get('init', 'real')
        if init not in ('real', 'noise'):
            raise ValueError(f"condense.init must be 'real' or 'noise', got {init!r}")
    else:
        if 'coreset' not in cfg:
            raise ValueError("method 'coreset' needs a 'coreset' section")
        from ddm.coreset import SELECTORS
        sel = cfg.coreset.get('selector')
        if sel not in SELECTORS:
            raise ValueError(f"coreset.selector must be one of {sorted(SELECTORS)}, got {sel!r}")

    models = cfg.eval.get('models')
    if not models or not isinstance(models, (list, tuple)):
        raise ValueError('config.eval.models must be a non-empty list of architecture names')
    return cfg
