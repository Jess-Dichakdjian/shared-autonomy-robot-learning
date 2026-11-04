"""Single-GPU BEAST wrapper around the validated resumable LIBERO benchmark."""
import argparse
import importlib.metadata
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
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument("--pilot", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--first-state", type=int, default=0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--gp-model", type=Path, help="Optional trained GP models.joblib for passive CPU monitoring")
    parser.add_argument("--gp-python", type=Path, default=Path.home() / ".venvs/shared-autonomy-robot-learning-gp-beast/bin/python",
                        help="Python executable for the isolated CPU-only GP environment")
    args = parser.parse_args()
    if not 1 <= args.pilot <= args.total <= 20 or args.base_seed < 0 or args.first_state < 0:
        raise ValueError("Require 1 <= pilot <= total <= 20 and non-negative seed/state")
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
    if args.gp_model is not None:
        from thesis_vla.gp.client import GPInferenceClient
        passive_gp = GPInferenceClient(args.gp_model, python_executable=args.gp_python)

    config = LiberoTaskConfig(suite="libero_spatial", task_id=0)
    task = str(get_benchmark_dict()[config.suite]().get_task(config.task_id).language)
    env = LiberoVectorAdapter(create_vector_env(config, gl="osmesa"))
    session = args.output / "sessions" / str(time.time_ns())
    session.mkdir(parents=True, exist_ok=True)
    try:
        state_count = len(env.env.envs[0].unwrapped._init_states)
        if args.first_state + args.total > state_count:
            raise ValueError(f"Plan requests {args.first_state + args.total} states; LIBERO provides {state_count}")
        started = time.perf_counter()
        policy = Pi05PolicyAdapter.from_pretrained(
            checkpoint, task=task, env_config=config, execute_steps=10,
            local_files_only=True, device="cuda:0", dtype=None, cpu_output_heads=True,
            meta_init=True, verify_placement_integrity=True, expected_weights_sha256=expected_sha,
            passive_gp=passive_gp,
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
                "thesis_vla/evaluation.py", "thesis_vla/pi05.py", "thesis_vla/placement.py",
                "thesis_vla/libero.py", "thesis_vla/dataset.py",
                "thesis_vla/gp/client.py", "scripts/gp_inference_worker.py")},
            "passive_gp_monitor": {
                "enabled": passive_gp is not None,
                "model_path": str(args.gp_model.expanduser().resolve()) if args.gp_model else None,
                "model_sha256": passive_gp.model_sha256 if passive_gp else None,
                "python_executable": str(args.gp_python.expanduser().resolve()) if passive_gp else None,
                "api": "GPInference in isolated CPU-only subprocess; fresh action-chunk boundaries only",
                "action_effect": "passive; never modifies VLA predictions or executed actions",
            },
            "state_protocol": "one explicit saved state index per episode; gate reset is overridden by the episode plan",
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
