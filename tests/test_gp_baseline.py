"""Synthetic-only tests for the offline GP baseline and export workflow."""
from __future__ import annotations
import json
import hashlib
import tarfile
from pathlib import Path
import joblib
import numpy as np
import pytest
from gp_synthetic_fixture import write_synthetic_gp_npz
from thesis_vla.gp.data import load_gp_dataset
from thesis_vla.gp.inference import GPInference
from thesis_vla.gp.metrics import regression_metrics
from thesis_vla.gp.model import IndependentGaussianProcess, RidgeBaseline
from thesis_vla.gp.pipeline import run_gp_experiment
from thesis_vla.gp.split import split_by_episode

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "gp_baseline.json"

def test_loader_schema_and_tar_bundle(tmp_path):
    npz = write_synthetic_gp_npz(tmp_path / "fixture.npz")
    loaded = load_gp_dataset(npz)
    assert loaded.X.shape == (75, 8)
    assert loaded.Y.shape == (75, 7)
    assert loaded.episode_id.shape == (75,)
    # Simulate the documented UCL bundle member without extracting it.
    bundle = tmp_path / "gp_dataset_bundle.tar.gz"
    manifest = tmp_path / "sha256.json"
    manifest.write_text(json.dumps({"gp_chunk_starts.npz": hashlib.sha256(npz.read_bytes()).hexdigest()}))
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(npz, arcname="dataset/gp_chunk_starts.npz")
        archive.add(manifest, arcname="dataset/sha256.json")
    from_bundle = load_gp_dataset(bundle)
    np.testing.assert_array_equal(from_bundle.X, loaded.X)
    np.testing.assert_array_equal(from_bundle.Y, loaded.Y)
    assert not (tmp_path / "dataset").exists()

def test_loader_rejects_invalid_shapes_nonfinite_and_object_ids(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, X=np.zeros((4, 7)), Y=np.zeros((4, 7)), episode_id=np.arange(4))
    with pytest.raises(ValueError, match="X must have shape"):
        load_gp_dataset(path)
    np.savez(path, X=np.zeros((4, 8)), Y=np.zeros((4, 7)), episode_id=np.arange(4, dtype=object))
    with pytest.raises(ValueError):
        load_gp_dataset(path)
    bad = np.zeros((4, 8))
    bad[0, 0] = np.nan
    np.savez(path, X=bad, Y=np.zeros((4, 7)), episode_id=np.arange(4))
    with pytest.raises(ValueError, match="finite"):
        load_gp_dataset(path)

def test_episode_split_is_reproducible_and_leak_free():
    ids = np.repeat(np.arange(20, dtype=np.int64), [3 + i % 4 for i in range(20)])
    first = split_by_episode(ids, seed=41)
    second = split_by_episode(ids, seed=41)
    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.validation, second.validation)
    np.testing.assert_array_equal(first.test, second.test)
    assert set(first.train_episode_ids).isdisjoint(first.validation_episode_ids)
    assert set(first.train_episode_ids).isdisjoint(first.test_episode_ids)
    assert set(first.validation_episode_ids).isdisjoint(first.test_episode_ids)
    for rows, groups in ((first.train, first.train_episode_ids),
                         (first.validation, first.validation_episode_ids),
                         (first.test, first.test_episode_ids)):
        assert set(np.unique(ids[rows])) == set(groups)
    assert len(first.train_episode_ids) == 12
    assert len(first.validation_episode_ids) == 4
    assert len(first.test_episode_ids) == 4

def test_split_rejects_too_few_episodes_and_invalid_fractions():
    with pytest.raises(ValueError, match="three"):
        split_by_episode(np.asarray([1, 1, 2, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="sum"):
        split_by_episode(np.arange(4), validation_fraction=0.5, test_fraction=0.5)

def test_gp_fit_predict_reproducibility_scaling_and_duplicate_inputs(tmp_path):
    path = write_synthetic_gp_npz(tmp_path / "fixture.npz")
    data = load_gp_dataset(path)
    split = split_by_episode(data.episode_id, seed=17)
    train, held = split.train, split.validation
    model_a = IndependentGaussianProcess(seed=17).fit(data.X[train], data.Y[train])
    model_b = IndependentGaussianProcess(seed=17).fit(data.X[train], data.Y[train])
    mean, std = model_a.predict(data.X[held])
    mean_b, std_b = model_b.predict(data.X[held])
    assert len(model_a.models) == 7
    assert mean.shape == std.shape == (len(held), 7)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
    np.testing.assert_allclose(mean, mean_b, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(std, std_b, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model_a.x_scaler.mean_, data.X[train].mean(axis=0))
    assert not np.allclose(model_a.x_scaler.mean_, data.X.mean(axis=0))
    # Positive alpha stabilizes repeated inputs in the training covariance.
    duplicate = np.repeat(data.X[:1], 6, axis=0)
    targets = np.repeat(data.Y[:1], 6, axis=0)
    stable = IndependentGaussianProcess().fit(duplicate, targets)
    dm, ds = stable.predict(duplicate[:2])
    assert np.isfinite(dm).all() and np.isfinite(ds).all() and (ds > 0).all()

def test_ridge_comparison_and_interval_metrics():
    truth = np.zeros((4, 7))
    metrics = regression_metrics(truth, truth, np.ones_like(truth))
    assert metrics["overall_mae"] == 0
    assert metrics["overall_rmse"] == 0
    assert metrics["predictive_intervals"]["0.9"]["overall_elementwise_coverage"] == 1.0
    assert metrics["predictive_intervals"]["0.95"]["overall_elementwise_coverage"] == 1.0
    model = RidgeBaseline().fit(np.arange(40).reshape(5, 8), np.arange(35).reshape(5, 7))
    assert model.predict(np.zeros((3, 8))).shape == (3, 7)

def test_pipeline_exports_validation_without_touching_test(tmp_path):
    data_path = write_synthetic_gp_npz(tmp_path / "synthetic.npz")
    out = tmp_path / "validation"
    result = run_gp_experiment(data_path, out, CONFIG, synthetic_data=True)
    assert "final_test" not in result["metrics"]
    assert result["splits"]["test_evaluated"] is False
    assert result["metrics"]["dataset_label"] == "synthetic_fixture"
    assert result["metrics"]["real_vla_results"] is False
    assert (out / "models.joblib").is_file()
    assert (out / "validation_predicted_vs_target.png").stat().st_size > 1000
    assert (out / "validation_standardized_residuals.png").stat().st_size > 1000
    with np.load(out / "predictions.npz", allow_pickle=False) as pred:
        assert pred["validation_gp_mean"].shape[1] == 7
        assert "test_Y_true" not in pred.files
    saved = joblib.load(out / "models.joblib")
    data = load_gp_dataset(data_path)
    validation_ids = result["splits"]["episode_ids"]["validation"]
    validation_mask = np.isin(data.episode_id, validation_ids)
    restored_mean, restored_std = saved["gp"].predict(data.X[validation_mask])
    pipeline_predictor = GPInference.load(out / "models.joblib")
    pipeline_mean, pipeline_std = pipeline_predictor.predict(data.X[validation_mask])
    np.testing.assert_allclose(pipeline_mean, restored_mean)
    np.testing.assert_allclose(pipeline_std, restored_std)
    with np.load(out / "predictions.npz", allow_pickle=False) as pred:
        np.testing.assert_allclose(restored_mean, pred["validation_gp_mean"])
        np.testing.assert_allclose(restored_std, pred["validation_gp_predictive_std"])
    with pytest.raises(FileExistsError, match="overwrite"):
        run_gp_experiment(data_path, out, CONFIG, synthetic_data=True)

def test_pipeline_final_test_is_explicit_and_exported(tmp_path):
    data_path = write_synthetic_gp_npz(tmp_path / "synthetic.npz")
    out = tmp_path / "final"
    result = run_gp_experiment(data_path, out, CONFIG, evaluate_test=True, synthetic_data=True)
    assert "final_test" in result["metrics"]
    assert result["splits"]["test_evaluated"] is True
    assert result["metrics"]["final_test"]["gp"]["n_samples"] > 0
    assert (out / "final_test_standardized_residuals.png").is_file()
    with np.load(out / "predictions.npz", allow_pickle=False) as pred:
        assert pred["test_gp_mean"].shape[1] == 7

def test_verified_ucl_mode_rejects_unmatched_synthetic_bundle(tmp_path):
    data_path = write_synthetic_gp_npz(tmp_path / "synthetic.npz")
    with pytest.raises(ValueError, match="documented UCL GP dataset SHA-256"):
        run_gp_experiment(data_path, tmp_path / "verified", CONFIG, verified_ucl_dataset=True)
