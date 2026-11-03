"""Independent scalar GPs with train-only scaling and explicit observation noise."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
@dataclass
class IndependentGaussianProcess:
    """One sklearn GPR per action coordinate.

    noise_variance is fixed homoscedastic Gaussian observation-noise variance
    in training-standardized target units. sklearn's returned standard
    deviation is latent-function posterior spread because alpha is added to
    the training covariance. Predictive standard deviation adds independent
    future-observation noise and transforms back to raw target units.
    """
    kernel_name: str = "Matern"
    matern_nu: float = 2.5
    noise_variance: float = 1e-4
    n_restarts_optimizer: int = 0
    length_scale_bounds: tuple[float, float] = (1e-2, 1e2)
    amplitude_bounds: tuple[float, float] = (1e-2, 1e2)
    seed: int = 17
    def __post_init__(self) -> None:
        if self.kernel_name not in {"Matern", "RBF"}:
            raise ValueError("kernel_name must be 'Matern' or 'RBF'")
        if self.kernel_name == "Matern" and self.matern_nu not in {0.5, 1.5, 2.5, float("inf")}:
            raise ValueError("matern_nu must be one of 0.5, 1.5, 2.5, or infinity")
        if not np.isfinite(self.noise_variance) or self.noise_variance <= 0:
            raise ValueError("noise_variance must be finite and positive")
        if self.n_restarts_optimizer < 0:
            raise ValueError("n_restarts_optimizer must be non-negative")
        self.x_scaler, self.y_scaler, self.models = StandardScaler(), StandardScaler(), []
    def _kernel(self, dimensions: int) -> Any:
        ls = np.ones(dimensions, dtype=np.float64)
        if self.kernel_name == "RBF":
            covariance = RBF(length_scale=ls, length_scale_bounds=self.length_scale_bounds)
        else:
            covariance = Matern(length_scale=ls, length_scale_bounds=self.length_scale_bounds, nu=self.matern_nu)
        return ConstantKernel(1.0, self.amplitude_bounds) * covariance
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "IndependentGaussianProcess":
        X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be 2-D arrays with matching sample counts")
        if X.shape[0] < 2 or X.shape[1] == 0 or Y.shape[1] == 0:
            raise ValueError("At least two samples and non-empty feature/target axes are required")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError("X and Y must be finite")
        self.x_scaler.fit(X)
        self.y_scaler.fit(Y)
        scaled_X, scaled_Y = self.x_scaler.transform(X), self.y_scaler.transform(Y)
        self.models = []
        for dim in range(Y.shape[1]):
            model = GaussianProcessRegressor(kernel=self._kernel(X.shape[1]), alpha=self.noise_variance,
                normalize_y=False, n_restarts_optimizer=self.n_restarts_optimizer, random_state=self.seed + dim)
            model.fit(scaled_X, scaled_Y[:, dim])
            self.models.append(model)
        return self
    def predict(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if not self.models:
            raise RuntimeError("Call fit before predict")
        if X.ndim != 2 or X.shape[1] != self.x_scaler.n_features_in_:
            raise ValueError(f"X must have shape (N, {self.x_scaler.n_features_in_}); got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X must be finite")
        sx = self.x_scaler.transform(X)
        sm, latent_s = np.empty((X.shape[0], len(self.models))), np.empty((X.shape[0], len(self.models)))
        for dim, model in enumerate(self.models):
            mean, std = model.predict(sx, return_std=True)
            sm[:, dim], latent_s[:, dim] = mean, std
        mean = self.y_scaler.inverse_transform(sm)
        scales = self.y_scaler.scale_[None, :]
        latent_s *= scales
        observation_s = np.sqrt(self.noise_variance) * scales
        predictive_s = np.sqrt(np.square(latent_s) + np.square(observation_s))
        if not np.isfinite(mean).all() or not np.isfinite(predictive_s).all():
            raise FloatingPointError("GP prediction produced non-finite outputs")
        return mean, predictive_s
@dataclass
class RidgeBaseline:
    """Simple non-GP baseline using independently fitted train-only scalers."""
    alpha: float = 1.0
    seed: int = 17
    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("Ridge alpha must be finite and non-negative")
        self.x_scaler, self.y_scaler = StandardScaler(), StandardScaler()
        self.model = Ridge(alpha=self.alpha)
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "RidgeBaseline":
        X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be 2-D arrays with matching sample counts")
        if X.shape[0] < 2 or not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError("Ridge training arrays need at least two finite rows")
        self.x_scaler.fit(X)
        self.y_scaler.fit(Y)
        self.model.fit(self.x_scaler.transform(X), self.y_scaler.transform(Y))
        return self
    def predict(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.x_scaler.n_features_in_:
            raise ValueError("Ridge prediction input has the wrong feature shape")
        if not np.isfinite(X).all():
            raise ValueError("X must be finite")
        return self.y_scaler.inverse_transform(self.model.predict(self.x_scaler.transform(X)))
