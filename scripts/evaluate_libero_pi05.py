"""Run the shared LIBERO evaluator with a local pretrained LeRobot pi0.5 checkpoint.

This script never downloads checkpoint weights. Prepare them on the target Linux
host, authenticate to Hugging Face there for the gated tokenizer, then pass the
local pretrained_model directory via --checkpoint. CPU and CUDA devices are supported.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path
import sys

import torch


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
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/pi05_libero"))
    parser.add_argument("--gl", choices=("osmesa", "egl", "glfw"), default="osmesa")
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"local checkpoint directory not found: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false on this host.")

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
            "execute_steps": args.execute_steps, "seed_rule": "base_seed + episode_index",
            "torch_version": torch.__version__, "policy_load_seconds": policy_load_seconds,
            "process_peak_rss_bytes_after_load": load_peak_bytes,
            "cpu_name": platform.processor() if args.device == "cpu" else None,
            "cpu_threads": torch.get_num_threads() if args.device == "cpu" else None,
            "gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        }
        result = run_evaluation(
            env, policy, args.output, episodes=args.episodes, base_seed=args.seed,
            task=task, configuration=run_config, max_steps=args.max_steps, record_video=True,
        )
        print(json.dumps({k: result[k] for k in ("episodes", "success_rate", "mean_return", "mean_steps")}))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
