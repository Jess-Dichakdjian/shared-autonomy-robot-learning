"""Reproducible train/validation workflow with an opt-in untouched test evaluation."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import joblib
import numpy as np
from .data import load_gp_dataset
from .metrics import regression_metrics
from .model import IndependentGaussianProcess, RidgeBaseline
from .plots import predicted_vs_target_plot, standardized_residual_plot
from .split import split_by_episode

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)

def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def run_gp_experiment(data_path: str | Path, output_dir: str | Path, config_path: str | Path,
                      *, evaluate_test: bool = False, synthetic_data: bool = False, verified_ucl_dataset: bool = False) -> dict:
    data_path, output_dir, config_path = Path(data_path).expanduser(), Path(output_dir).expanduser(), Path(config_path).expanduser()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if synthetic_data and verified_ucl_dataset:
        raise ValueError("A dataset cannot be both synthetic and verified UCL data")
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported GP config schema_version")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {output_dir}")
    dataset = load_gp_dataset(data_path)
    seed = int(config["seed"])
    split = split_by_episode(dataset.episode_id, seed=seed,
        validation_fraction=float(config["split"]["validation_fraction_of_episodes"]),
        test_fraction=float(config["split"]["test_fraction_of_episodes"]))
    X, Y = dataset.X, dataset.Y
    data_sha256 = _sha256(data_path)
    if verified_ucl_dataset:
        if data_sha256 != "c4ef96e62cd72bcde473150feffed443fc45dddc37e0333d176c8622268bfeef":
            raise ValueError("The provided tar bundle does not match the documented UCL GP dataset SHA-256")
        if X.shape != (161, 8) or Y.shape != (161, 7) or np.unique(dataset.episode_id).size != 20:
            raise ValueError("The verified UCL dataset must contain 161 X/Y pairs across 20 episodes")
    X_train, Y_train = X[split.train], Y[split.train]
    X_val, Y_val = X[split.validation], Y[split.validation]
    gp_config = config["gp"]
    gp = IndependentGaussianProcess(kernel_name=gp_config["kernel"], matern_nu=float(gp_config["matern_nu"]),
        noise_variance=float(gp_config["observation_noise_variance_standardized_target_units"]),
        n_restarts_optimizer=int(gp_config["optimizer_restarts"]), seed=seed,
        length_scale_bounds=tuple(gp_config["length_scale_bounds"]), amplitude_bounds=tuple(gp_config["amplitude_bounds"]))
    ridge = RidgeBaseline(alpha=float(config["comparison"]["alpha"]), seed=seed)
    gp.fit(X_train, Y_train)
    ridge.fit(X_train, Y_train)
    levels = tuple(float(v) for v in config["interval_levels"])
    val_gp_mean, val_gp_std = gp.predict(X_val)
    val_ridge_mean = ridge.predict(X_val)
    metrics = {
        "dataset_label": "synthetic_fixture" if synthetic_data else ("verified_ucl_pi05" if verified_ucl_dataset else "user_supplied_unverified"),
        "real_vla_results": bool(verified_ucl_dataset),
        "warning": "Metrics are not verified real VLA results unless the supplied input is the validated UCL benchmark dataset.",
        "validation": {
            "gp": regression_metrics(Y_val, val_gp_mean, val_gp_std, interval_levels=levels),
            "ridge": regression_metrics(Y_val, val_ridge_mean),
        },
    }
    predictions = {"validation": {
        "Y_true": Y_val, "gp_mean": val_gp_mean, "gp_predictive_std": val_gp_std,
        "ridge_mean": val_ridge_mean, "episode_id": dataset.episode_id[split.validation]}}
    if evaluate_test:
        X_test, Y_test = X[split.test], Y[split.test]
        test_gp_mean, test_gp_std = gp.predict(X_test)
        test_ridge_mean = ridge.predict(X_test)
        metrics["final_test"] = {
            "gp": regression_metrics(Y_test, test_gp_mean, test_gp_std, interval_levels=levels),
            "ridge": regression_metrics(Y_test, test_ridge_mean),
            "test_partition_evaluated_once": True,
        }
        predictions["test"] = {"Y_true": Y_test, "gp_mean": test_gp_mean,
            "gp_predictive_std": test_gp_std, "ridge_mean": test_ridge_mean,
            "episode_id": dataset.episode_id[split.test]}
    splits = {
        "seed": seed,
        "row_counts": {"train": int(split.train.size), "validation": int(split.validation.size), "test": int(split.test.size)},
        "episode_ids": {"train": split.train_episode_ids.tolist(),
            "validation": split.validation_episode_ids.tolist(), "test": split.test_episode_ids.tolist()},
        "test_evaluated": bool(evaluate_test),
    }
    joblib.dump({"gp": gp, "ridge": ridge}, output_dir / "models.joblib")
    np.savez_compressed(output_dir / "predictions.npz", **{
        f"{partition}_{name}": values for partition, result in predictions.items() for name, values in result.items()})
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "splits.json", splits)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "experiment.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_path": str(data_path.resolve()),
        "dataset_sha256": data_sha256,
        "verified_ucl_dataset": bool(verified_ucl_dataset),
        "dataset_label": metrics["dataset_label"],
        "config_sha256": _sha256(config_path),
        "config_path": str(config_path.resolve()),
        "evaluate_test": bool(evaluate_test),
        "model_formulation": "Seven independent scalar GaussianProcessRegressor models with a shared configuration.",
        "optimized_kernel_per_action_dim": [str(model.kernel_) for model in gp.models],
        "python_version": __import__("platform").python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "pytest", "joblib")},
    })
    predicted_vs_target_plot(Y_val, val_gp_mean, output_dir / "validation_predicted_vs_target.png")
    standardized_residual_plot(Y_val, val_gp_mean, val_gp_std, output_dir / "validation_standardized_residuals.png")
    if evaluate_test:
        predicted_vs_target_plot(Y_test, test_gp_mean, output_dir / "final_test_predicted_vs_target.png")
        standardized_residual_plot(Y_test, test_gp_mean, test_gp_std, output_dir / "final_test_standardized_residuals.png")
    return {"output_dir": str(output_dir), "metrics": metrics, "splits": splits}
