"""Run the shared LIBERO evaluator with a local pretrained LeRobot pi0.5 checkpoint.

This script never downloads checkpoint weights. Prepare them on the target Linux
host, authenticate to Hugging Face there for the gated tokenizer, then pass the
local pretrained_model directory via --checkpoint. CPU and CUDA devices are supported.
"""
from __future__ import annotations

import argparse
import json
import platform
import os
import resource
import time
from pathlib import Path
import sys

import torch


def _resolve_cuda_device(device):
    selected = torch.device(device)
    if selected.type != "cuda":
        return None
    return torch.device("cuda", torch.cuda.current_device() if selected.index is None else selected.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="local LeRobot pi0.5 pretrained_model directory; weights are never downloaded")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--execute-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda", help="inference device; cpu is accepted when RAM headroom permits")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/pi05_libero"))
    parser.add_argument("--gl", choices=("osmesa", "egl", "glfw"), default="osmesa")
    parser.add_argument("--cpu-output-heads", action="store_true", help="retain unused vocabulary heads in CPU RAM with call guards")
    parser.add_argument("--prediction-gate", action="store_true", help="verify a real prediction and exact agreement with heads on the other GPU before evaluating")
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"local checkpoint directory not found: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false on this host.")
    cuda_device = _resolve_cuda_device(args.device)
    monitored_devices = list(range(torch.cuda.device_count())) if cuda_device is not None else []
    if cuda_device is not None:
        torch.cuda.set_device(cuda_device)
        for index in monitored_devices:
            torch.cuda.reset_peak_memory_stats(index)
    args.output.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from scripts.libero_smoke import prepare_config_and_assets
    prepare_config_and_assets()

    from libero.libero.benchmark import get_benchmark_dict
    from thesis_vla.evaluation import run_evaluation
    from thesis_vla.libero import LiberoTaskConfig, LiberoVectorAdapter, create_vector_env
    from thesis_vla.pi05 import Pi05PolicyAdapter

    config = LiberoTaskConfig(suite=args.suite, task_id=args.task_id)
    task = str(get_benchmark_dict()[args.suite]().get_task(args.task_id).language)
    env = LiberoVectorAdapter(create_vector_env(config, gl=args.gl))
    policy = None
    try:
        load_started = time.perf_counter()
        policy = Pi05PolicyAdapter.from_pretrained(
            args.checkpoint, task=task, env_config=config,
            execute_steps=args.execute_steps, local_files_only=True, device=args.device,
            dtype=None if args.dtype == "auto" else getattr(torch, args.dtype),
            cpu_output_heads=args.cpu_output_heads,
        )
        policy_load_seconds = time.perf_counter() - load_started
        load_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        load_peak_bytes = int(load_peak if sys.platform == "darwin" else load_peak * 1024)
        model_config = policy.policy.config
        run_config = {
            "suite": args.suite, "task_id": args.task_id, "task_instruction": task,
            "width": config.width, "height": config.height, "fps": config.fps,
            "control_mode": config.control_mode, "camera_names": config.camera_name.split(","),
            "renderer": args.gl, "policy": "lerobot_pi05",
            "checkpoint": str(args.checkpoint.resolve()), "device": args.device,
            "dtype": str(getattr(model_config, "dtype", "checkpoint_default")),
            "compile_model": model_config.compile_model,
            "gradient_checkpointing": model_config.gradient_checkpointing,
            "weight_loading": "strict_cpu_then_device",
            "execute_steps": args.execute_steps, "seed_rule": "base_seed + episode_index",
            "torch_version": torch.__version__, "policy_load_seconds": policy_load_seconds,
            "process_peak_rss_bytes_after_load": load_peak_bytes,
            "cpu_name": platform.processor() if args.device == "cpu" else None,
            "cpu_threads": torch.get_num_threads() if args.device == "cpu" else None,
            "gpu_name": torch.cuda.get_device_name(cuda_device) if cuda_device is not None else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_version": torch.version.cuda,
            "egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "placement": getattr(policy.policy, "placement_report", {"mode": "whole_policy"}),
            "prediction_gate_extra_resets": int(args.prediction_gate),
        }
        (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        if args.prediction_gate:
            if not args.cpu_output_heads or cuda_device is None or len(monitored_devices) != 2:
                raise ValueError("Prediction gate requires CPU output heads and exactly two visible CUDA GPUs")
            from thesis_vla.placement import verify_prediction_gate
            observation, _ = env.reset(seed=args.seed)
            gate = verify_prediction_gate(policy, observation, seed=args.seed,
                                          reference_head_device=f"cuda:{1 - cuda_device.index}")
            (args.output / "prediction_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
            print("PREDICTION_GATE_PASSED " + json.dumps({k: v for k, v in gate.items() if k != "predicted_chunk"}), flush=True)
        result = run_evaluation(
            env, policy, args.output, episodes=args.episodes, base_seed=args.seed,
            task=task, configuration=run_config, max_steps=args.max_steps, record_video=True,
        )
        print(json.dumps({k: result[k] for k in ("episodes", "success_rate", "mean_return", "mean_steps")}))
    finally:
        if cuda_device is not None:
            memory = {
                "device": args.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_device),
                "completed": sys.exc_info()[0] is None,
                "all_visible_devices": {str(index): {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
                } for index in monitored_devices},
                "note": "PyTorch allocator peaks; external GPU usage requires nvidia-smi monitoring",
            }
            (args.output / "gpu_memory.json").write_text(json.dumps(memory, indent=2) + "\n")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
