"""Action-wise error, predictive spread, and Gaussian interval coverage."""
from __future__ import annotations
from statistics import NormalDist
from typing import Sequence
import numpy as np
from numpy.typing import ArrayLike
def regression_metrics(Y_true: ArrayLike, Y_pred: ArrayLike, predictive_std: ArrayLike | None = None, *, interval_levels: Sequence[float] = (0.90, 0.95)) -> dict:
    truth, prediction = np.asarray(Y_true, dtype=np.float64), np.asarray(Y_pred, dtype=np.float64)
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("Y_true and Y_pred must have the same two-dimensional shape")
    if truth.shape[0] == 0 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Targets and predictions must be non-empty and finite")
    residual = truth - prediction
    result = {"n_samples": int(truth.shape[0]), "n_action_dimensions": int(truth.shape[1]),
        "mae_by_action_dim": np.mean(np.abs(residual), axis=0).tolist(),
        "rmse_by_action_dim": np.sqrt(np.mean(np.square(residual), axis=0)).tolist(),
        "overall_mae": float(np.mean(np.abs(residual))), "overall_rmse": float(np.sqrt(np.mean(np.square(residual))))}
    if predictive_std is None:
        return result
    std = np.asarray(predictive_std, dtype=np.float64)
    if std.shape != truth.shape or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("predictive_std must be positive, finite, and match target shape")
    zr = residual / std
    result["predictive_std_mean_by_action_dim"] = np.mean(std, axis=0).tolist()
    result["predictive_variance_mean_by_action_dim"] = np.mean(np.square(std), axis=0).tolist()
    result["standardized_residual_mean_by_action_dim"] = np.mean(zr, axis=0).tolist()
    result["standardized_residual_std_by_action_dim"] = np.std(zr, axis=0).tolist()
    result["predictive_variance_definition"] = "Future observation variance per action is latent posterior variance plus fixed homoscedastic Gaussian observation-noise variance, transformed to raw checkpoint-normalized Y units."
    intervals = {}
    for level in interval_levels:
        if not (0.0 < level < 1.0):
            raise ValueError("interval levels must be between zero and one")
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        covered = np.abs(residual) <= z * std
        intervals[str(float(level))] = {"z_value": float(z), "coverage_by_action_dim": np.mean(covered, axis=0).tolist(),
            "overall_elementwise_coverage": float(np.mean(covered)), "definition": "Empirical held-out fraction within mean +/- z_level * predictive_std using Gaussian quantiles."}
    result["predictive_intervals"] = intervals
    return result
