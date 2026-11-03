"""NumPy-only-at-call-site inference adapter for trained offline GP artifacts."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
from numpy.typing import ArrayLike
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from .model import IndependentGaussianProcess


class GPInference:
    """Load a fitted GP bundle and predict in the stored checkpoint target units.

    The predictor accepts NumPy-like normalized states only. A single state has
    shape (8,) and returns two (7,) arrays; a batch has shape (N, 8) and returns
    two (N, 7) arrays. Means and standard deviations refer to raw
    checkpoint-normalized pi0.5 horizon-zero Y, not controller commands.
    """

    input_dimensions = 8
    output_dimensions = 7

    def __init__(self, model: IndependentGaussianProcess):
        self._validate_model(model)
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "GPInference":
        """Load a direct GP object or the models.joblib bundle from training."""
        artifact_path = Path(path).expanduser()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"GP model artifact does not exist: {artifact_path}")
        try:
            artifact: Any = joblib.load(artifact_path)
        except Exception as exc:
            raise ValueError(f"Could not load GP model artifact {artifact_path}: {exc}") from exc

        if isinstance(artifact, dict):
            version = artifact.get("schema_version", 1)
            if version != 1:
                raise ValueError(f"Unsupported GP model artifact schema_version: {version}")
            model = artifact.get("gp")
            if model is None:
                raise ValueError("GP model artifact bundle is missing the 'gp' model")
        else:
            model = artifact
        if not isinstance(model, IndependentGaussianProcess):
            raise ValueError("Artifact does not contain a compatible IndependentGaussianProcess")
        return cls(model)

    @classmethod
    def _validate_model(cls, model: IndependentGaussianProcess) -> None:
        models = getattr(model, "models", None)
        x_scaler = getattr(model, "x_scaler", None)
        y_scaler = getattr(model, "y_scaler", None)
        if not isinstance(x_scaler, StandardScaler) or not isinstance(y_scaler, StandardScaler):
            raise ValueError("GP artifact is missing compatible fitted preprocessing scalers")
        if not isinstance(models, list) or len(models) != cls.output_dimensions:
            raise ValueError("Fitted GP artifact must contain exactly seven action-dimension models")
        if getattr(x_scaler, "n_features_in_", None) != cls.input_dimensions:
            raise ValueError("GP artifact was not fitted with eight input state dimensions")
        if getattr(y_scaler, "n_features_in_", None) != cls.output_dimensions:
            raise ValueError("GP artifact was not fitted with seven output action dimensions")
        if np.asarray(getattr(y_scaler, "scale_", [])).shape != (cls.output_dimensions,):
            raise ValueError("GP artifact is missing compatible fitted action scaling")
        for action_dim, scalar_model in enumerate(models):
            if not isinstance(scalar_model, GaussianProcessRegressor) or not hasattr(scalar_model, "kernel_") or getattr(scalar_model, "n_features_in_", None) != cls.input_dimensions:
                raise ValueError(f"GP artifact action model {action_dim} is not fitted for eight inputs")

    def predict(self, state: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean and std for one normalized state or a batch."""
        values = np.asarray(state, dtype=np.float64)
        is_single = values.ndim == 1
        if is_single:
            if values.shape != (self.input_dimensions,):
                raise ValueError(f"Single state must have shape ({self.input_dimensions},); got {values.shape}")
            values = values[None, :]
        elif values.ndim != 2 or values.shape[1] != self.input_dimensions:
            raise ValueError(f"State batch must have shape (N, {self.input_dimensions}); got {values.shape}")
        if values.shape[0] == 0:
            raise ValueError("State batch must contain at least one state")
        if not np.isfinite(values).all():
            raise ValueError("States must contain only finite values")

        mean, std = self.model.predict(values)
        expected = (values.shape[0], self.output_dimensions)
        if mean.shape != expected or std.shape != expected:
            raise RuntimeError(f"Fitted GP returned incompatible shapes: mean={mean.shape}, std={std.shape}")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise FloatingPointError("Fitted GP returned non-finite means or invalid predictive standard deviations")
        if is_single:
            return mean[0], std[0]
        return mean, std


def measure_cpu_prediction_latency(
    predictor: GPInference,
    state: ArrayLike,
    *,
    warmup: int = 3,
    repeats: int = 30,
) -> dict[str, float | int]:
    """Measure local CPU wall latency without imposing a hardware threshold."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        predictor.predict(state)
    elapsed = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        start = time.perf_counter()
        predictor.predict(state)
        elapsed[index] = time.perf_counter() - start
    return {
        "repeats": repeats,
        "mean_seconds": float(np.mean(elapsed)),
        "median_seconds": float(np.median(elapsed)),
        "p95_seconds": float(np.quantile(elapsed, 0.95)),
    }
