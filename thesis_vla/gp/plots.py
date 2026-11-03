"""Static action-wise GP diagnostic plots."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
def _axes():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    return fig, axes.ravel()
def predicted_vs_target_plot(Y_true: ArrayLike, Y_pred: ArrayLike, path: str | Path) -> None:
    truth, prediction = np.asarray(Y_true, dtype=np.float64), np.asarray(Y_pred, dtype=np.float64)
    fig, axes = _axes()
    for dim, ax in enumerate(axes[:truth.shape[1]]):
        ax.scatter(truth[:, dim], prediction[:, dim], s=18, alpha=0.75)
        low, high = min(float(truth[:, dim].min()), float(prediction[:, dim].min())), max(float(truth[:, dim].max()), float(prediction[:, dim].max()))
        if low == high:
            low, high = low - 1e-6, high + 1e-6
        ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set(title=f"Action {dim}", xlabel="Target", ylabel="GP predicted mean")
    for ax in axes[truth.shape[1]:]:
        ax.set_visible(False)
    fig.suptitle("Predicted versus target (checkpoint-normalized action units)")
    fig.savefig(path, dpi=160)
    plt.close(fig)
def standardized_residual_plot(Y_true: ArrayLike, Y_pred: ArrayLike, predictive_std: ArrayLike, path: str | Path) -> None:
    truth, prediction, std = (np.asarray(a, dtype=np.float64) for a in (Y_true, Y_pred, predictive_std))
    residual = (truth - prediction) / std
    fig, axes = _axes()
    for dim, ax in enumerate(axes[:truth.shape[1]]):
        ax.scatter(prediction[:, dim], residual[:, dim], s=18, alpha=0.75)
        ax.axhline(0.0, color="black", linewidth=1)
        for level, color, style in ((1.645, "tab:orange", "--"), (1.96, "tab:red", ":")):
            ax.axhline(level, color=color, linestyle=style, linewidth=1)
            ax.axhline(-level, color=color, linestyle=style, linewidth=1)
        ax.set(title=f"Action {dim}", xlabel="GP predicted mean", ylabel="Standardized residual")
    for ax in axes[truth.shape[1]:]:
        ax.set_visible(False)
    fig.suptitle("Standardized residuals (future-observation predictive standard deviation)")
    fig.savefig(path, dpi=160)
    plt.close(fig)
