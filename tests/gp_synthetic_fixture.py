"""Deterministic synthetic data for GP tests; never represents a VLA rollout."""
from __future__ import annotations
from pathlib import Path
import numpy as np

def write_synthetic_gp_npz(path: str | Path, *, seed: int = 20261103,
                           n_episodes: int = 15, samples_per_episode: int = 5) -> Path:
    """Write a numeric X/Y/episode_id fixture with smooth targets and episode clusters."""
    rng = np.random.default_rng(seed)
    episode_ids, states = [], []
    centers = rng.normal(size=(n_episodes, 8))
    for episode in range(n_episodes):
        local = centers[episode] + rng.normal(scale=0.18, size=(samples_per_episode, 8))
        states.append(local)
        episode_ids.extend([episode] * samples_per_episode)
    X = np.vstack(states).astype(np.float32)
    Y = np.column_stack([
        np.sin(X[:, 0]) + 0.25 * X[:, 1],
        np.cos(X[:, 1]) - 0.2 * X[:, 2],
        np.tanh(X[:, 2] + X[:, 3]),
        0.3 * X[:, 0] - 0.4 * X[:, 4],
        np.sin(X[:, 4] * X[:, 5]),
        np.cos(X[:, 6]) + 0.1 * X[:, 7],
        np.tanh(X[:, 7] - X[:, 0]),
    ])
    Y += rng.normal(scale=0.015, size=Y.shape)
    np.savez_compressed(path, X=X, Y=Y.astype(np.float32),
                        episode_id=np.asarray(episode_ids, dtype=np.int64))
    return Path(path)
