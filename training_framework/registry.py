"""Name -> adapter lookup. Not a plugin system, just a dict -- adapters are
first-class modules in training_framework/adapters/, registered by decorator."""

_ADAPTERS = {}

# name -> dotted module path, imported lazily so `train --config x` doesn't
# pay for torch unless the config actually asks for a torch-based adapter.
_BUILTINS = {
    "lstm_bias_correction": "training_framework.adapters.lstm_bias_correction",
}


def register(name):
    def deco(obj):
        _ADAPTERS[name] = obj
        return obj
    return deco


def get(name):
    if name not in _ADAPTERS and name in _BUILTINS:
        import importlib
        importlib.import_module(_BUILTINS[name])
    if name not in _ADAPTERS:
        raise KeyError(f"No adapter registered as '{name}'. Known: {sorted(_ADAPTERS) or sorted(_BUILTINS)}")
    return _ADAPTERS[name]
