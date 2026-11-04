"""Validation-only tuning and diagnostic plots for GP/action-monitor signals."""
from __future__ import annotations
import argparse
import csv
import json
import warnings
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from thesis_vla.gp.data import load_gp_dataset
from thesis_vla.gp.model import IndependentGaussianProcess, RidgeBaseline
from thesis_vla.gp.split import split_by_episode

Z90 = 1.6448536269514722
Z95 = 1.959963984540054


def read_episode_samples(root: Path, episode_index: int):
    epdir = root / "episodes" / f"episode_{episode_index:03d}"
    episode = json.loads((epdir / "episodes.jsonl").read_text().splitlines()[0])
    rows = [json.loads(line) for line in (epdir / "steps.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["chunk_start"]]
    if not rows:
        raise ValueError(f"No action chunk starts in {epdir}")
    X, Y, gp_mean, gp_std = [], [], [], []
    for row in rows:
        monitor = row.get("passive_gp_prediction")
        if not isinstance(monitor, dict):
            raise ValueError(f"Missing passive GP output at {epdir} step {row['step']}")
        state = np.asarray(row["prediction_state"]["vla_normalized"], dtype=np.float64)
        if not np.array_equal(state, np.asarray(monitor["input_state_vla_normalized"], dtype=np.float64)):
            raise ValueError("GP input is not the exact current VLA-normalized state")
        X.append(state)
        Y.append(np.asarray(row["predicted_action"], dtype=np.float64))
        gp_mean.append(np.asarray(monitor["mean_raw_normalized_action"], dtype=np.float64))
        gp_std.append(np.asarray(monitor["predictive_std_raw_normalized_action"], dtype=np.float64))
    return {"X": np.asarray(X), "Y": np.asarray(Y), "sidecar_mean": np.asarray(gp_mean),
            "sidecar_std": np.asarray(gp_std), "success": episode["success"],
            "task": episode["task"], "episode_key": f"{episode['configuration']['task_id']}:{episode['reset_info']['saved_init_state_index']}",
            "steps": episode["steps"]}


def load_group(root: Path, indices):
    return [read_episode_samples(root, i) for i in indices]


def concatenate(group, key):
    return np.concatenate([item[key] for item in group], axis=0)


def row_rmse(y, mean, scale):
    return np.sqrt(np.mean(np.square((y - mean) / scale), axis=1))


def scores(X, y, mean, std, scale, x_scaler, train_X):
    unc = np.sqrt(np.mean(np.square(std / scale[None, :]), axis=1))
    dis = row_rmse(y, mean, scale)
    sx, st = x_scaler.transform(X), x_scaler.transform(train_X)
    dist = np.linalg.norm(sx[:, None, :] - st[None, :, :], axis=2).min(axis=1)
    return unc, dis, dist


def coverage(y, mean, std, z):
    return float(np.mean(np.abs(y - mean) <= z * std))


def spearman(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--pilot-task1", type=Path, required=True)
    parser.add_argument("--pilot-task4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    dataset = load_gp_dataset(args.dataset)
    split_info = json.loads(args.split_json.read_text())
    train_eps = set(split_info["episode_ids"]["train"])
    base_mask = np.isin(dataset.episode_id, list(train_eps))
    X_base, Y_base = dataset.X[base_mask], dataset.Y[base_mask]

    # Fixed before the held-out rollouts: state 1 augments training; state 0
    # selects the kernel/noise and validation-only signal thresholds.
    task1_train = load_group(args.pilot_task1, [1])
    task4_train = load_group(args.pilot_task4, [1])
    task1_val = load_group(args.pilot_task1, [0])
    task4_val = load_group(args.pilot_task4, [0])
    train_group = task1_train + task4_train
    val_group = task1_val + task4_val
    X_train = np.concatenate([X_base, concatenate(train_group, "X")])
    Y_train = np.concatenate([Y_base, concatenate(train_group, "Y")])
    X_val, Y_val = concatenate(val_group, "X"), concatenate(val_group, "Y")
    task_rows = np.concatenate([[int(item["episode_key"].split(":")[0])] * len(item["X"]) for item in val_group])
    episode_rows = np.concatenate([[item["episode_key"]] * len(item["X"]) for item in val_group])

    candidates = []
    candidate_models = []
    for noise in (1e-4, 1e-3, 1e-2):
        for upper in (100.0, 1000.0):
            for restarts in (0, 2):
                gp = IndependentGaussianProcess(kernel_name="Matern", matern_nu=2.5,
                    noise_variance=noise, n_restarts_optimizer=restarts,
                    length_scale_bounds=(1e-2, upper), seed=17)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    gp.fit(X_train, Y_train)
                mean, std = gp.predict(X_val)
                rmse = float(np.sqrt(np.mean(np.square(Y_val - mean))))
                coverage90, coverage95 = coverage(Y_val, mean, std, Z90), coverage(Y_val, mean, std, Z95)
                rows = {"noise": noise, "upper": upper, "restarts": restarts,
                    "rmse": rmse, "mae": float(np.mean(np.abs(Y_val-mean))),
                    "coverage90": coverage90, "coverage95": coverage95,
                    "warning_count": len(caught),
                    "warnings": " | ".join(sorted(set(str(w.message) for w in caught))),
                    "kernels": [str(model.kernel_) for model in gp.models]}
                candidates.append(rows)
                candidate_models.append(gp)

    best_rmse = min(row["rmse"] for row in candidates)
    eligible = [i for i,row in enumerate(candidates) if row["rmse"] <= best_rmse * 1.02]
    selected_idx = min(eligible, key=lambda i: (abs(candidates[i]["coverage90"]-0.90), candidates[i]["rmse"]))
    selected = candidate_models[selected_idx]
    ridge = RidgeBaseline(alpha=1.0, seed=17).fit(X_train, Y_train)
    mean, std = selected.predict(X_val)
    scale = selected.y_scaler.scale_.copy()
    unc, dis, dist = scores(X_val, Y_val, mean, std, scale, selected.x_scaler, X_train)
    err = row_rmse(Y_val, mean, scale)
    ratio = np.abs(Y_val-mean) / std
    calibration = float(np.quantile(ratio, 0.90, method="linear") / Z90)
    calibrated_std = std * calibration
    val_scores = {
        "gp_uncertainty_90th_percentile": float(np.quantile(unc, 0.90)),
        "gp_vla_disagreement_90th_percentile": float(np.quantile(dis, 0.90)),
        "state_novelty_90th_percentile": float(np.quantile(dist, 0.90)),
    }
    ridge_mean = ridge.predict(X_val)
    report = {
        "dataset_label": "BEAST pilot validation only",
        "train_rows": int(len(X_train)), "validation_rows": int(len(X_val)),
        "training_episode_groups": sorted([item["episode_key"] for item in train_group]),
        "validation_episode_groups": sorted([item["episode_key"] for item in val_group]),
        "selected_candidate_index": selected_idx, "selected_config": candidates[selected_idx],
        "selection_rule": "Within 2% of minimum validation RMSE, select 90% coverage closest to 0.90; otherwise minimum RMSE.",
        "validation": {
            "gp_mae": float(np.mean(np.abs(Y_val-mean))),
            "gp_rmse": float(np.sqrt(np.mean((Y_val-mean)**2))),
            "ridge_mae": float(np.mean(np.abs(Y_val-ridge_mean))),
            "ridge_rmse": float(np.sqrt(np.mean((Y_val-ridge_mean)**2))),
            "coverage90_original": coverage(Y_val,mean,std,Z90),
            "coverage95_original": coverage(Y_val,mean,std,Z95),
            "coverage90_calibrated": coverage(Y_val,mean,calibrated_std,Z90),
            "coverage95_calibrated": coverage(Y_val,mean,calibrated_std,Z95),
            "validation_calibration_multiplier": calibration,
            "spearman_uncertainty_vs_standardized_action_error": spearman(unc,err),
            "spearman_state_novelty_vs_standardized_action_error": spearman(dist,err),
            "spearman_disagreement_vs_standardized_action_error": spearman(dis,err),
            "note": "Chunk-start rows are correlated within episode. Correlations and pooled calibration are descriptive, not independent-sample inference or conformal guarantees.",
        },
        "thresholds_from_validation_only": val_scores,
        "test_partition_opened": False,
    }
    joblib.dump({"gp": selected, "ridge": ridge}, output/"models.joblib")
    (output/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    with (output/"candidate_metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["noise","upper","restarts","rmse","mae","coverage90","coverage95","warning_count","warnings","kernels"])
        writer.writeheader()
        for row in candidates: writer.writerow(row)
    np.savez_compressed(output/"validation_predictions.npz",X=X_val,Y=Y_val,gp_mean=mean,gp_std=std,
        calibrated_std=calibrated_std,ridge_mean=ridge_mean,uncertainty_score=unc,
        disagreement_score=dis,state_novelty_score=dist,row_error=err,task_id=task_rows,episode_key=episode_rows)

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for task_id,label in ((1,"Spatial task 1"),(4,"Spatial task 4")):
        mask=task_rows==task_id
        axes[0].scatter(unc[mask],err[mask],label=label,alpha=.75)
        axes[1].scatter(dist[mask],err[mask],label=label,alpha=.75)
    axes[0].set(xlabel="GP uncertainty (RMS standardized predictive SD)",ylabel="Action error (standardized row RMSE)",title="Uncertainty vs surrogate error")
    axes[1].set(xlabel="Nearest training state distance",ylabel="Action error (standardized row RMSE)",title="State novelty vs surrogate error")
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"validation_signal_vs_error.png",dpi=160); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([.9,.95],[.9,.95],"k--",label="nominal")
    ax.plot([.9,.95],[coverage(Y_val,mean,std,Z90),coverage(Y_val,mean,std,Z95)],"o-",label="original GP")
    ax.plot([.9,.95],[coverage(Y_val,mean,calibrated_std,Z90),coverage(Y_val,mean,calibrated_std,Z95)],"s-",label="validation-scaled")
    ax.set(xlim=(.85,1),ylim=(0,1),xlabel="Nominal interval coverage",ylabel="Observed elementwise coverage",title="Pilot validation interval coverage")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(output/"validation_coverage.png",dpi=160);plt.close(fig)

    fig,ax1=plt.subplots(figsize=(9,4))
    labels=[f"{x['noise']:g}\nU{x['upper']:g}/r{x['restarts']}" for x in candidates]
    ax1.plot(range(len(candidates)),[x["rmse"] for x in candidates],"o-",label="RMSE")
    ax1.set_xticks(range(len(candidates)),labels,rotation=45,ha="right")
    ax1.set_ylabel("Validation RMSE")
    ax2=ax1.twinx();ax2.plot(range(len(candidates)),[x["coverage90"] for x in candidates],"s--",color="tab:orange",label="90% coverage");ax2.set_ylabel("90% interval coverage")
    ax1.grid(alpha=.2);fig.tight_layout();fig.savefig(output/"hyperparameter_sensitivity.png",dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
