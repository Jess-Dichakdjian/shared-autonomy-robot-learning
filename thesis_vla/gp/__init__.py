"""Lazy exports for offline GP utilities and inference clients."""
from importlib import import_module

_EXPORTS = {
    "GPDataset": ("data", "GPDataset"),
    "load_gp_dataset": ("data", "load_gp_dataset"),
    "GPInference": ("inference", "GPInference"),
    "measure_cpu_prediction_latency": ("inference", "measure_cpu_prediction_latency"),
    "GPInferenceClient": ("client", "GPInferenceClient"),
    "IndependentGaussianProcess": ("model", "IndependentGaussianProcess"),
    "RidgeBaseline": ("model", "RidgeBaseline"),
    "EpisodeSplit": ("split", "EpisodeSplit"),
    "split_by_episode": ("split", "split_by_episode"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
