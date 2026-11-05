"""Single-GPU BEAST wrapper around the validated resumable LIBERO benchmark."""
import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / ".cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/7152f13e34c6c115d808aea0e5d08aca74eac14b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-plan", type=Path,
                        help="Pre-registered scene manifest; held-out test scenarios are never executable.")
    parser.add_argument("--scenario-split", choices=("discovery", "validation"), default="discovery")
    parser.add_argument("--scene-validation", type=Path,
                        help="Required physical-preflight report matching --scenario-plan")
    parser.add_argument("--scenario-ids", nargs="+",
                        help="Optional explicit scenario IDs for a bounded pilot or continuation")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument("--pilot", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--first-state", type=int, default=0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--gp-model", type=Path, help="Optional trained GP models.joblib for passive CPU monitoring")
    parser.add_argument("--gp-python", type=Path, default=Path.home() / ".venvs/shared-autonomy-robot-learning-gp-beast/bin/python",
                        help="Python executable for the isolated CPU-only GP environment")
    parser.add_argument("--monitor-mode", choices=("autonomy_only", "passive_monitoring", "gp_uncertainty",
                        "gp_vla_disagreement", "state_distance", "random_matched_budget"),
                        default="passive_monitoring")
    parser.add_argument("--monitor-threshold-config", type=Path,
                        help="Validation-frozen configs/gp_uncertainty_selected_model.json")
    parser.add_argument("--random-budget", type=int, default=1,
                        help="Fixed request budget for random_matched_budget over 28 possible chunks")
    parser.add_argument("--random-seed", type=int, default=20261104)
    args = parser.parse_args()
    if args.base_seed < 0 or args.first_state < 0:
        raise ValueError("Require non-negative seed/state")
    scenario_specifications = None
    scenario_manifest_sha256 = None
    scene_validation_sha256 = None
    scene_validation = None
    if args.scenario_plan is not None:
        from thesis_vla.perturbations import validate_scenario
        if args.scene_validation is None:
            raise ValueError("--scenario-plan requires a matching --scene-validation physical preflight")
        scenario_manifest = json.loads(args.scenario_plan.expanduser().read_text(encoding="utf-8"))
        scene_validation = json.loads(args.scene_validation.expanduser().read_text(encoding="utf-8"))
        scenario_manifest_sha256 = hashlib.sha256(args.scenario_plan.expanduser().read_bytes()).hexdigest()
        scene_validation_sha256 = hashlib.sha256(args.scene_validation.expanduser().read_bytes()).hexdigest()
        if scene_validation.get("manifest_sha256") != scenario_manifest_sha256:
            raise ValueError("physical preflight report belongs to a different scenario manifest")
        valid_ids = {item["scenario_id"] for item in scene_validation.get("results", [])
                     if item.get("status") == "valid"}
        if scenario_manifest.get("schema_version") != 1 or scenario_manifest.get("task_id") != args.task_id:
            raise ValueError("scenario plan schema/task does not match runner")
        if scenario_manifest.get("final_test_status") != "reserved_not_run":
            raise ValueError("scenario manifest does not mark final held-out test as reserved")
        scenario_specifications = []
        for candidate in scenario_manifest.get("scenarios", []):
            if candidate.get("split") != args.scenario_split or candidate.get("scenario_id") not in valid_ids:
                continue
            if args.scenario_ids and candidate.get("scenario_id") not in set(args.scenario_ids):
                continue
            validate_scenario(candidate)
            if candidate.get("task_id") != args.task_id:
                raise ValueError("scenario task ID differs from runner")
            scenario_specifications.append({
                "episode_id": len(scenario_specifications),
                "scenario_id": candidate["scenario_id"],
                "seed": candidate["seed"],
                "saved_init_state_index": candidate["saved_init_state_index"],
                "scene_perturbation": candidate,
            })
        if not scenario_specifications:
            raise ValueError(f"scenario manifest has no {args.scenario_split} scenarios")
        if args.scenario_split == "discovery" and scenario_manifest.get("final_test_status") != "reserved_not_run":
            raise ValueError("held-out test reservation is missing")
        args.total = len(scenario_specifications)
        args.pilot = min(args.pilot, args.total)
    elif not 1 <= args.pilot <= args.total <= 20:
        raise ValueError("Require 1 <= pilot <= total <= 20")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=0 for the single-GPU BEAST run")
    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise ValueError("Set MUJOCO_GL=osmesa; EGL is not validated on this WSL2 host")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Set HF_HUB_OFFLINE=1; checkpoint and processor are already cached")

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError(f"Unexpected GPU architecture: {torch.cuda.get_device_capability(0)}")

    from thesis_vla.benchmark import atomic_json, digest, run_benchmark
    from thesis_vla.dataset import export_dataset
    from scripts.libero_smoke import prepare_config_and_assets

    checkpoint = args.checkpoint.expanduser().resolve()
    expected_revision = "7152f13e34c6c115d808aea0e5d08aca74eac14b"
    expected_sha = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"
    if not checkpoint.is_dir() or digest(checkpoint / "model.safetensors") != expected_sha:
        raise ValueError("Pinned pi0.5 checkpoint is absent or has an unexpected weight SHA-256")
    if expected_revision not in str(checkpoint):
        raise ValueError("Checkpoint path does not identify the pinned immutable revision")
    provenance = json.loads(importlib.metadata.distribution("lerobot").read_text("direct_url.json"))
    expected_lerobot_commit = "146d8474932c8b566717eb2738ba459337ab71c4"
    if provenance["vcs_info"]["commit_id"] != expected_lerobot_commit:
        raise ValueError("Installed LeRobot source commit differs from the validated UCL pin")
    if importlib.metadata.version("torch") != "2.10.0+cu128" or torch.version.cuda != "12.8":
        raise ValueError("BEAST requires the validated PyTorch 2.10.0+cu128 stack")

    prepare_config_and_assets()
    from libero.libero.benchmark import get_benchmark_dict
    from thesis_vla.libero import LiberoTaskConfig, LiberoVectorAdapter, create_vector_env
    from thesis_vla.pi05 import Pi05PolicyAdapter
    from thesis_vla.placement import verify_single_gpu_action_contract, verify_single_gpu_placement

    passive_gp = None
    monitoring_thresholds = {}
    random_schedule = None
    if args.monitor_mode not in {"autonomy_only", "passive_monitoring"} and args.monitor_mode != "random_matched_budget":
        if args.monitor_threshold_config is None:
            raise ValueError("Signal-trigger modes require validation-frozen --monitor-threshold-config")
    if args.monitor_threshold_config is not None:
        threshold_config = json.loads(args.monitor_threshold_config.expanduser().read_text(encoding="utf-8"))
        raw_thresholds = threshold_config["request_thresholds_from_validation_only"]
        monitoring_thresholds = {
            "gp_uncertainty": raw_thresholds["gp_uncertainty_standardized_rms_90th"],
            "gp_vla_disagreement": raw_thresholds["gp_vla_disagreement_standardized_rms_90th"],
            "state_distance": raw_thresholds["nearest_training_state_distance_90th"],
        }
    if args.monitor_mode == "random_matched_budget":
        from thesis_vla.gp.monitoring import random_matched_schedule
        def random_schedule(episode_id):
            digest = hashlib.sha256(str(episode_id).encode("utf-8")).digest()
            episode_seed = int.from_bytes(digest[:4], "big")
            return random_matched_schedule(max_boundaries=28, budget=args.random_budget,
                                           seed=(args.random_seed + episode_seed) % (2**32))
    if args.gp_model is not None:
        from thesis_vla.gp.client import GPInferenceClient
        passive_gp = GPInferenceClient(args.gp_model, python_executable=args.gp_python)
    if args.monitor_mode not in {"autonomy_only", "passive_monitoring"} and passive_gp is None:
        raise ValueError("Triggered assistance requests require --gp-model")

    config = LiberoTaskConfig(suite=args.suite, task_id=args.task_id)
    task = str(get_benchmark_dict()[config.suite]().get_task(config.task_id).language)
    env = LiberoVectorAdapter(create_vector_env(config, gl="osmesa"))
    session = args.output / "sessions" / str(time.time_ns())
    session.mkdir(parents=True, exist_ok=True)
    try:
        state_count = len(env.env.envs[0].unwrapped._init_states)
        requested_states = ([item["saved_init_state_index"] for item in scenario_specifications]
                            if scenario_specifications is not None
                            else list(range(args.first_state, args.first_state + args.total)))
        if any(index < 0 or index >= state_count for index in requested_states):
            raise ValueError(f"Scenario plan requests an unavailable initial state (count={state_count})")
        started = time.perf_counter()
        policy = Pi05PolicyAdapter.from_pretrained(
            checkpoint, task=task, env_config=config, execute_steps=10,
            local_files_only=True, device="cuda:0", dtype=None, cpu_output_heads=True,
            meta_init=True, verify_placement_integrity=True, expected_weights_sha256=expected_sha,
            passive_gp=passive_gp, monitoring_mode=args.monitor_mode,
            monitoring_thresholds=monitoring_thresholds, random_schedule=random_schedule,
        )
        identity = {
            "suite": config.suite, "task_id": config.task_id, "fps": config.fps,
            "width": config.width, "height": config.height, "control_mode": config.control_mode,
            "camera_names": config.camera_name.split(","), "renderer": "osmesa",
            "checkpoint_id": "lerobot/pi05_libero_finetuned_v044",
            "checkpoint_revision": expected_revision, "checkpoint_sha256": expected_sha,
            "historical_alias": "lerobot/pi05_libero_finetuned",
            "processor_file_sha256": {p.name: digest(p) for p in checkpoint.glob("policy_*") if p.is_file()},
            "lerobot_commit": expected_lerobot_commit, "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda, "device": "cuda:0", "cuda_visible_devices": "0",
            "execute_steps": 10, "predicted_action_chunk": 50, "dtype": str(policy.policy.config.dtype),
            "compile_model": False, "gradient_checkpointing": False,
            "weight_loading": "strict_meta_assign_then_device",
            "meta_buffer_reinitialization": getattr(policy.policy, "meta_buffer_reinitialization", []),
            "placement": policy.policy.placement_report,
            "normalization": {k: str(v) for k, v in policy.policy.config.normalization_mapping.items()},
            "source_file_sha256": {name: digest(Path(__file__).resolve().parents[1] / name) for name in (
                "scripts/benchmark_libero_pi05_beast.py", "thesis_vla/benchmark.py",
                "thesis_vla/perturbations.py",
                "thesis_vla/evaluation.py", "thesis_vla/pi05.py", "thesis_vla/placement.py",
                "thesis_vla/libero.py", "thesis_vla/dataset.py",
                "thesis_vla/gp/client.py", "scripts/gp_inference_worker.py")},
            "assistance_monitor": {
                "mode": args.monitor_mode,
                "thresholds": monitoring_thresholds,
                "random_budget": args.random_budget if args.monitor_mode == "random_matched_budget" else None,
                "random_seed": args.random_seed if args.monitor_mode == "random_matched_budget" else None,
                "request_only": True,
                "action_effect": "none; no correction provider was used",
            },
            "passive_gp_monitor": {
                "enabled": passive_gp is not None,
                "model_path": str(args.gp_model.expanduser().resolve()) if args.gp_model else None,
                "model_sha256": passive_gp.model_sha256 if passive_gp else None,
                "python_executable": str(args.gp_python.expanduser().resolve()) if passive_gp else None,
                "api": "GPInference in isolated CPU-only subprocess; fresh action-chunk boundaries only",
                "action_effect": "passive; never modifies VLA predictions or executed actions",
            },
            "state_protocol": "one explicit saved state index per episode; gate reset is overridden by the episode plan",
            "scene_scenario_manifest_sha256": scenario_manifest_sha256,
            "scene_validation_sha256": scene_validation_sha256,
            "scene_preflight_rejections": ([{"scenario_id": item.get("scenario_id"), "reason": item.get("reason")}
                                           for item in scene_validation.get("results", [])
                                           if item.get("status") != "valid"] if scene_validation else []),
            "scenario_split": args.scenario_split if scenario_specifications is not None else None,
            "selected_scenario_ids": ([item["scenario_id"] for item in scenario_specifications]
                                      if scenario_specifications is not None else None),
            "held_out_test_status": "reserved_not_run" if scenario_manifest_sha256 else None,
        }
        atomic_json(session / "load.json", {"load_seconds": time.perf_counter() - started, "identity": identity})
        observation, reset_info = env.reset(seed=args.base_seed, options={"saved_init_state_index": args.first_state})
        if reset_info.get("saved_init_state_index") != args.first_state:
            raise RuntimeError("Single-GPU prediction gate reset did not use the requested first state")
        placement = verify_single_gpu_placement(policy.policy, "cuda:0")
        gate = verify_single_gpu_action_contract(policy, observation)
        atomic_json(session / "single_gpu_prediction_gate.json", {
            "passed": True, "placement": placement, "action_contract": gate,
            "checkpoint_weight_loading": "strict", "checkpoint_sha256": expected_sha,
            "numerical_comparison": "two-GPU reference unavailable on BEAST",
        })
        print("BEAST_PRETRAINED_GATE_PASSED", flush=True)
        policy.reset()
        result = run_benchmark(
            env, policy, args.output, configuration=identity, task=task,
            total=args.total, pilot=args.pilot, base_seed=args.base_seed,
            first_state=args.first_state, stop_after=args.stop_after,
            pilot_validator=lambda: export_dataset(args.output),
            scenario_specifications=scenario_specifications,
        )
        dataset = export_dataset(args.output)
        from scripts.report_libero_benchmark import report
        summary = report(args.output)
        print(json.dumps({
            "status": result["status"], "completed": result["completed"],
            "successes": result["successes"], "failures": result["failures"],
            "success_rate": result["success_rate"], "wilson_95": result["wilson_95"],
            "dataset_samples": dataset["samples"], "dataset_episodes": dataset["episodes"],
            "prediction_points": summary["prediction_points"],
            "chunk_latency_seconds": summary["chunk_latency_seconds"],
            "gpu_allocator_peaks_bytes": summary["gpu_allocator_peaks_bytes"],
        }, indent=2), flush=True)
    finally:
        if passive_gp is not None:
            passive_gp.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
