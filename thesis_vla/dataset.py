"""GP-ready horizon-zero pairs, never treating queued actions as fresh predictions."""
import csv
import json
from pathlib import Path
import tarfile
import numpy as np
from thesis_vla.benchmark import atomic_json, completed_episodes, digest, validate_episode


def export_dataset(output):
    output = Path(output)
    records = completed_episodes(output)
    if not records:
        raise ValueError("No validated complete episodes")
    target = output / "dataset"
    target.mkdir(exist_ok=True)
    pairs, steps, episodes = [], [], []
    for record in records:
        path = output / "episodes" / f"episode_{record['episode_id']:03d}"
        validate_episode(path, require_video=json.loads((output / "experiment.json").read_text())["record_video"])
        episode = json.loads((path / "episodes.jsonl").read_text())
        episodes.append({**record, "started_at": episode["started_at"], "ended_at": episode["ended_at"]})
        for line in (path / "steps.jsonl").read_text().splitlines():
            r = json.loads(line)
            r["episode_outcome"] = record["success"]
            r["termination_reason"] = record["termination_reason"]
            steps.append(r)
            if r["chunk_start"]:
                pairs.append({"episode_id": record["episode_id"], "seed": r["seed"],
                    "saved_init_state_index": r["saved_init_state_index"], "task_success": record["success"],
                    "chunk_id": r["chunk_id"], "step": r["step"], "chunk_index": 0,
                    "X": r["prediction_state"]["vla_normalized"],
                    "X_physical": r["prediction_state"]["physical"],
                    "Y": r["predicted_action"], "Y_executed": r["executed_action"],
                    "Y_chunk": r["predicted_chunk"][0], "inference_seconds": r["inference_seconds"],
                    "wall_time_seconds": r["wall_time_seconds"], "prediction_started_at": r["prediction_started_at"],
                    "episode_elapsed_seconds": r["episode_elapsed_seconds"]})
    arrays = {key: np.asarray([p[key] for p in pairs], dtype=np.float32) for key in ("X", "X_physical", "Y", "Y_executed", "Y_chunk")}
    for key in ("episode_id", "seed", "saved_init_state_index", "chunk_id", "step", "chunk_index"):
        arrays[key] = np.asarray([p[key] for p in pairs], dtype=np.int64)
    arrays["task_success"] = np.asarray([p["task_success"] for p in pairs], dtype=bool)
    for key in ("inference_seconds", "wall_time_seconds", "prediction_started_at", "episode_elapsed_seconds"):
        arrays[key] = np.asarray([p[key] for p in pairs], dtype=np.float64)
    if arrays["X"].shape != (len(pairs), 8) or arrays["Y"].shape != (len(pairs), 7):
        raise ValueError("Invalid GP matrix shapes")
    if not all(np.isfinite(arrays[k]).all() for k in ("X", "X_physical", "Y", "Y_executed", "Y_chunk")):
        raise ValueError("Non-finite GP data")
    np.testing.assert_array_equal(arrays["Y"], arrays["Y_chunk"][:, 0])
    np.savez_compressed(target / "gp_chunk_starts.npz", **arrays)
    for name, rows in (("gp_chunk_starts.jsonl", pairs), ("full_step_trajectories.jsonl", steps), ("episode_outcomes.jsonl", episodes)):
        with (target / name).open("w") as stream:
            for r in rows:
                stream.write(json.dumps(r, separators=(",", ":")) + "\n")
    headers = ["episode_id","seed","saved_init_state_index","task_success","chunk_id","step","inference_seconds","wall_time_seconds"]
    vectors = [("X",8),("X_physical",8),("Y",7),("Y_executed",7)]
    with (target / "gp_chunk_starts.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(headers + [f"{k}_{i}" for k,n in vectors for i in range(n)])
        for p in pairs:
            writer.writerow([p[k] for k in headers] + [v for k,n in vectors for v in p[k]])
    schema = {"version":1,"samples":len(pairs),"step_samples":len(steps),"episodes":len(records),
        "X":"N x 8 exact checkpoint-normalized observation.state at a fresh prediction; no images/language",
        "X_physical":"N x 8 pre-normalizer state: eef xyz, quaternion converted by official LIBERO processor to axis-angle xyz, gripper qpos[2]",
        "Y":"N x 7 raw normalized pi0.5 prediction, horizon 0 of each fresh chunk; ACTION MEAN_STD representation",
        "Y_executed":"N x 7 first action after unchanged policy unnormalization, LIBERO postprocessing and existing clipping; relative OSC controller interface",
        "Y_chunk":"N x 50 x 7 complete raw normalized predicted chunk; future actions are not additional independent state observations",
        "pairing":"One row per chunk start. Queued actions retain original prediction state/step in separate full_step_trajectories.jsonl; never paired to later state as a fresh prediction.",
        "normalization":"Saved checkpoint MEAN_STD for STATE and ACTION; no new fitted normalization. Physical state and executed actions also retained. Runtime state tokens are derived by unchanged pi0.5 processor.",
        "timing":"Wall-clock Unix seconds UTC; episode elapsed seconds; chunk-generation latency including preprocessing/postprocessing",
        "identity":"episode_id + chunk_id + step; seeds and explicit saved initial-state indices; task outcome joined after episode completion",
        "limitations":"Correlated trajectories, single task/instruction, state alone is not a sufficient statistic for image-conditioned policy; no GP fit or uncertainty calibration performed. Split future GP data by episode, not random adjacent rows."}
    atomic_json(target / "schema.json", schema)
    files={p.name:digest(p) for p in target.iterdir() if p.is_file() and p.name != "sha256.json"}
    atomic_json(target / "sha256.json", files)
    with tarfile.open(output / "gp_dataset_bundle.tar.gz", "w:gz") as archive:
        archive.add(target,arcname="dataset")
        archive.add(output/"experiment.json",arcname="experiment.json")
        archive.add(output/"progress.json",arcname="progress.json")
    return schema
