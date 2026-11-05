"""Sequential, resumable episodes; progress never depends on model checkpoints."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from thesis_vla.evaluation import run_evaluation


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextlib.contextmanager
def experiment_lock(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    with (Path(output) / ".writer.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class IndexedEpisodeEnv:
    def __init__(self, env, specification):
        self.env = env
        self.specification = specification
        self.state_index = specification["saved_init_state_index"]
    def reset(self, *, seed):
        options = {"saved_init_state_index": self.state_index}
        if "scene_perturbation" in self.specification:
            options["scene_perturbation"] = self.specification["scene_perturbation"]
        observation, info = self.env.reset(seed=seed, options=options)
        if info.get("saved_init_state_index") != self.state_index:
            raise RuntimeError("Environment did not confirm requested saved initial state")
        return observation, info
    def step(self, action):
        return self.env.step(action)


def validate_episode(path, *, require_video=True, require_gp=True):
    path = Path(path)
    metrics = json.loads((path / "metrics.json").read_text())
    episodes = [json.loads(s) for s in (path / "episodes.jsonl").read_text().splitlines()]
    rows = [json.loads(s) for s in (path / "steps.jsonl").read_text().splitlines()]
    if len(episodes) != 1 or not rows or episodes[0]["steps"] != len(rows):
        raise ValueError("Incomplete episode records")
    episode = episodes[0]
    if not isinstance(episode["success"], bool):
        raise ValueError("Unknown task outcome")
    if not (episode["terminated"] or episode["truncated"] or episode["termination_reason"] == "evaluation_limit"):
        raise ValueError("Episode has no valid stopping reason")
    if metrics["mean_steps"] != len(rows) or metrics["success_rate"] != float(episode["success"]):
        raise ValueError("Inconsistent metrics")
    latest = None
    prediction_count = 0
    passive_enabled = bool(episode.get("configuration", {}).get("passive_gp_monitor", {}).get("enabled", False))
    for index, row in enumerate(rows):
        if row["step"] != index or row["seed"] != episode["seed"]:
            raise ValueError("Inconsistent step identity")
        for key in ("predicted_action", "executed_action"):
            a = np.asarray(row[key])
            if a.shape != (7,) or not np.isfinite(a).all():
                raise ValueError("Invalid action")
        if np.max(np.abs(row["executed_action"])) > 1:
            raise ValueError("Executed action out of bounds")
        passive = row.get("passive_gp_prediction")
        if passive_enabled:
            if row["chunk_start"]:
                if not isinstance(passive, dict):
                    raise ValueError("Fresh prediction is missing its passive GP record")
                state = np.asarray(passive.get("input_state_vla_normalized"))
                mean = np.asarray(passive.get("mean_raw_normalized_action"))
                std = np.asarray(passive.get("predictive_std_raw_normalized_action"))
                source_state = np.asarray(row.get("prediction_state", {}).get("vla_normalized"))
                if state.shape != (8,) or mean.shape != (7,) or std.shape != (7,) or source_state.shape != (8,):
                    raise ValueError("Passive GP record has incompatible state/action dimensions")
                if not np.isfinite(state).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                    raise ValueError("Passive GP record contains non-finite predictions")
                np.testing.assert_array_equal(state, source_state)
            elif passive is not None:
                raise ValueError("Queued action must not receive a new passive GP prediction")
        elif passive is not None:
            raise ValueError("Passive GP output was logged while monitor was disabled")
        if require_gp:
            if row["saved_init_state_index"] != episode["reset_info"]["saved_init_state_index"]:
                raise ValueError("Initial-state identity changed")
            for key in ("physical", "vla_normalized"):
                x = np.asarray(row["prediction_state"][key])
                if x.shape != (8,) or not np.isfinite(x).all():
                    raise ValueError("Invalid prediction-state vector")
            if row["chunk_start"]:
                latest = row
                prediction_count += 1
                chunk = np.asarray(row["predicted_chunk"])
                if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != 7 or not np.isfinite(chunk).all():
                    raise ValueError("Invalid prediction chunk")
                np.testing.assert_array_equal(chunk[0, 0], row["predicted_action"])
                if row["chunk_index"] != 0 or row["prediction_step"] != index:
                    raise ValueError("Invalid prediction point")
            if latest is None or row["chunk_id"] != latest["chunk_id"]:
                raise ValueError("Missing chunk-start observation")
            if row["prediction_step"] != latest["step"] or row["chunk_index"] != index - latest["step"]:
                raise ValueError("Invalid chunk timing")
            if row["prediction_state"] != latest["prediction_state"]:
                raise ValueError("Queued action incorrectly paired with later state")
            np.testing.assert_array_equal(np.asarray(latest["predicted_chunk"])[0, row["chunk_index"]], row["predicted_action"])
    files = ["metrics.json", "episodes.jsonl", "steps.jsonl"]
    if require_video:
        import imageio.v2 as imageio
        with imageio.get_reader(path / episode["video"]) as reader:
            count = 0
            for frame in reader:
                if frame.shape != (256, 512, 3):
                    raise ValueError("Invalid dual-camera video shape")
                count += 1
        if count != len(rows) + 1:
            raise ValueError("Incomplete video")
        files.append(episode["video"])
    scene_record = episode.get("reset_info", {}).get("scene_perturbation", {})
    return {"episode_id": rows[0]["episode_id"], "scenario_id": scene_record.get("scenario_id"),
            "scenario_sha256": scene_record.get("scenario_sha256"), "seed": episode["seed"],
            "saved_init_state_index": episode["reset_info"]["saved_init_state_index"],
            "saved_init_state_sha256": episode["reset_info"]["saved_init_state_sha256"],
            "success": episode["success"], "steps": len(rows), "prediction_count": prediction_count,
            "termination_reason": episode["termination_reason"],
            "files": {name: digest(path / name) for name in files}}


def completed_episodes(output, *, verify=True):
    completed = []
    for path in sorted((Path(output) / "episodes").glob("episode_*")):
        marker = path / "complete.json"
        if not marker.exists():
            continue
        record = json.loads(marker.read_text())
        if verify:
            for name, expected in record["files"].items():
                if digest(path / name) != expected:
                    raise ValueError(f"Committed evidence changed: {path / name}")
        completed.append(record)
    return completed


def wilson(successes, total):
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [max(0., float(center-half)), min(1., float(center+half))]


def run_benchmark(env, policy, output, *, configuration, task, total=20, pilot=5,
                  base_seed=0, first_state=0, record_video=True, stop_after=None,
                  pilot_validator=None, scenario_specifications=None):
    if scenario_specifications is not None:
        scenario_specifications = [dict(item) for item in scenario_specifications]
        total = len(scenario_specifications)
    if not 1 <= total <= 20 or not 1 <= pilot <= total:
        raise ValueError("Require 1 <= pilot <= total <= 20")
    output = Path(output)
    episode_plan = scenario_specifications if scenario_specifications is not None else [
        {"episode_id": i, "seed": base_seed+i, "saved_init_state_index": first_state+i} for i in range(total)
    ]
    if len({item["episode_id"] for item in episode_plan}) != len(episode_plan):
        raise ValueError("scenario episode IDs must be unique")
    plan = {"schema_version": 2 if scenario_specifications is not None else 1,
            "configuration": configuration, "task": task,
            "total": total, "pilot": pilot, "base_seed": base_seed, "first_state": first_state,
            "timeout_steps": 280, "record_video": record_video, "episodes": episode_plan}
    with experiment_lock(output):
        manifest = output / "experiment.json"
        if manifest.exists() and json.loads(manifest.read_text()) != plan:
            raise ValueError("Resume protocol/configuration does not match saved experiment")
        atomic_json(manifest, plan)
        (output / "episodes").mkdir(exist_ok=True)
        done = completed_episodes(output)
        def update(status, **extra):
            successes = sum(r["success"] for r in done)
            r = {"status": status, "completed": len(done), "successes": successes,
                 "failures": len(done)-successes, "success_rate": successes/len(done) if done else None,
                 "wilson_95": wilson(successes, len(done)), "episodes": done, **extra}
            atomic_json(output / "progress.json", r)
            return r
        completed_ids = {r["episode_id"] for r in done}
        if len(done) >= pilot and not (output / "pilot_report.json").exists():
            if pilot_validator:
                pilot_validator()
            atomic_json(output / "pilot_report.json", {"validated": True, "completed": pilot, "episodes": done[:pilot]})
        for specification in plan["episodes"]:
            i = specification["episode_id"]
            if i in completed_ids:
                continue
            if stop_after is not None and len(done) >= stop_after:
                return update("paused_by_episode_limit")
            path = output / "episodes" / f"episode_{i:03d}"
            if path.exists():
                archive = output / "interrupted_attempts"
                archive.mkdir(exist_ok=True)
                archived_attempt = archive / f"episode_{i:03d}_{time.time_ns()}"
                path.rename(archived_attempt)
                previous_progress = output / "progress.json"
                if previous_progress.exists():
                    previous = json.loads(previous_progress.read_text())
                    if previous.get("status") == "error" and previous.get("current_episode", {}).get("episode_id") == i:
                        atomic_json(archived_attempt / "attempt_error.json", {
                            "error_type": previous.get("error_type"),
                            "error": previous.get("error"),
                            "scenario": previous.get("current_episode"),
                        })
            update("running", current_episode=specification)
            try:
                import torch
                attempt_started_at = time.time()
                devices = list(range(torch.cuda.device_count())) if configuration.get("device", "").startswith("cuda") and torch.cuda.is_available() else []
                for d in devices:
                    torch.cuda.reset_peak_memory_stats(d)
                run_evaluation(IndexedEpisodeEnv(env, specification), policy, path,
                               episodes=1, base_seed=specification["seed"], task=task,
                               configuration={**configuration, **specification}, max_steps=280,
                               record_video=record_video)
                record = validate_episode(path, require_video=record_video)
                record["attempt_started_at"] = attempt_started_at
                record["attempt_ended_at"] = time.time()
                scene_hash = record.get("scenario_sha256")
                completed_scene_hashes = {
                    row.get("scenario_sha256") for row in done if row.get("scenario_sha256")
                }
                if scene_hash and scene_hash in completed_scene_hashes:
                    raise ValueError("Repeated scene configuration in benchmark plan")
                if scenario_specifications is None and record["saved_init_state_sha256"] in {
                    r["saved_init_state_sha256"] for r in done
                }:
                    raise ValueError("Repeated saved scene state in benchmark plan")
                record["scenario_id"] = specification.get("scenario_id")
                record["scenario_sha256"] = scene_hash
                record["gpu_allocator_peaks"] = {str(d): {"allocated": torch.cuda.max_memory_allocated(d),
                    "reserved": torch.cuda.max_memory_reserved(d),
                    "free_after_episode": torch.cuda.mem_get_info(d)[0]} for d in devices}
                atomic_json(path / "complete.json", record)
                done.append(record)
                status = update("complete" if len(done) == total else "running")
                print(f"EPISODE_COMPLETE {i} success={record['success']} steps={record['steps']} total={len(done)}", flush=True)
                if len(done) == pilot:
                    pilot_dataset = None
                    if pilot_validator:
                        pilot_dataset = pilot_validator()
                    atomic_json(output / "pilot_report.json", {**status, "validated": True, "dataset_validation": pilot_dataset})
            except BaseException as error:
                update("error", current_episode=specification, error_type=type(error).__name__, error=str(error))
                raise
        return update("complete")
