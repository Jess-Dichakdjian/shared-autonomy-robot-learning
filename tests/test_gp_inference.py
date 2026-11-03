"""Synthetic-only coverage for the reusable GP inference interface."""
from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from gp_synthetic_fixture import write_synthetic_gp_npz
from thesis_vla.gp.data import load_gp_dataset
from thesis_vla.gp.inference import GPInference, measure_cpu_prediction_latency
from thesis_vla.gp.model import IndependentGaussianProcess
from thesis_vla.gp.split import split_by_episode


def test_saved_gp_round_trip_single_batch_and_cpu_latency(tmp_path, capsys):
    dataset_path = write_synthetic_gp_npz(tmp_path / "synthetic_gp.npz")
    data = load_gp_dataset(dataset_path)
    split = split_by_episode(data.episode_id, seed=17)
    fitted = IndependentGaussianProcess(seed=17).fit(data.X[split.train], data.Y[split.train])
    states = data.X[split.validation][:4]
    expected_mean, expected_std = fitted.predict(states)

    artifact_path = tmp_path / "models.joblib"
    joblib.dump({"gp": fitted, "ridge": None}, artifact_path)
    predictor = GPInference.load(artifact_path)

    single_mean, single_std = predictor.predict(states[0])
    batch_mean, batch_std = predictor.predict(states)
    assert single_mean.shape == single_std.shape == (7,)
    assert batch_mean.shape == batch_std.shape == (4, 7)
    np.testing.assert_allclose(batch_mean, expected_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(batch_std, expected_std, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(single_mean, batch_mean[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(single_std, batch_std[0], rtol=1e-10, atol=1e-10)
    assert np.isfinite(batch_mean).all()
    assert np.isfinite(batch_std).all()
    assert (batch_std > 0).all()

    timing = measure_cpu_prediction_latency(predictor, states[0], warmup=2, repeats=20)
    assert timing["repeats"] == 20
    assert 0 < timing["mean_seconds"] < float("inf")
    assert 0 < timing["median_seconds"] < float("inf")
    assert 0 < timing["p95_seconds"] < float("inf")
    print(f"Synthetic CPU GP prediction latency: {timing}")


def test_inference_rejects_missing_bad_and_incompatible_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GPInference.load(tmp_path / "missing.joblib")

    missing_gp = tmp_path / "missing_gp.joblib"
    joblib.dump({"ridge": None}, missing_gp)
    with pytest.raises(ValueError, match="missing the 'gp'"):
        GPInference.load(missing_gp)

    wrong_type = tmp_path / "wrong_type.joblib"
    joblib.dump({"schema_version": 2, "gp": None}, wrong_type)
    with pytest.raises(ValueError, match="schema_version"):
        GPInference.load(wrong_type)

    incompatible = tmp_path / "incompatible.joblib"
    joblib.dump({"gp": "not a fitted GP"}, incompatible)
    with pytest.raises(ValueError, match="compatible IndependentGaussianProcess"):
        GPInference.load(incompatible)


def test_inference_rejects_bad_state_shapes_empty_batches_and_nonfinite(tmp_path):
    data_path = write_synthetic_gp_npz(tmp_path / "synthetic_gp.npz")
    data = load_gp_dataset(data_path)
    split = split_by_episode(data.episode_id, seed=17)
    model = IndependentGaussianProcess(seed=17).fit(data.X[split.train], data.Y[split.train])
    artifact_path = tmp_path / "gp.joblib"
    joblib.dump(model, artifact_path)
    predictor = GPInference.load(artifact_path)

    for bad in (np.zeros(7), np.zeros((2, 9)), np.zeros((1, 1, 8)), np.zeros((0, 8))):
        with pytest.raises(ValueError):
            predictor.predict(bad)
    nonfinite = np.zeros(8)
    nonfinite[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(nonfinite)

def test_inference_rejects_fitted_model_with_wrong_feature_contract(tmp_path):
    incompatible = IndependentGaussianProcess()
    incompatible.x_scaler = StandardScaler().fit(np.zeros((3, 6)))
    incompatible.y_scaler = StandardScaler().fit(np.zeros((3, 7)))
    incompatible.models = [None] * 7
    artifact = tmp_path / "wrong_dimensions.joblib"
    joblib.dump(incompatible, artifact)
    with pytest.raises(ValueError, match="eight input state dimensions"):
        GPInference.load(artifact)
