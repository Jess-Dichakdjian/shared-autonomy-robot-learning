"""Validated fixed-config pi0.5 pilot -> 20 episode experiment; never downloads."""
import argparse
import importlib.metadata
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path.home()/"models/pi05_libero_finetuned_v044")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument("--pilot", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--first-state", type=int, default=0)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if not 1 <= args.pilot <= args.total <= 20 or args.base_seed < 0 or args.first_state < 0:
        raise ValueError("Invalid episode plan")
    args.output.mkdir(parents=True, exist_ok=True)
    session_lock = (args.output / ".session.lock").open("a")
    fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from thesis_vla.benchmark import atomic_json, completed_episodes, digest, run_benchmark
    from thesis_vla.dataset import export_dataset
    if (args.output / "progress.json").exists():
        plan = json.loads((args.output / "experiment.json").read_text())
        for key, value in (("total", args.total), ("pilot", args.pilot), ("base_seed", args.base_seed), ("first_state", args.first_state)):
            if plan[key] != value:
                raise ValueError(f"Resume changes {key}")
        if len(completed_episodes(args.output)) == args.total:
            export_dataset(args.output)
            print("Already complete; validated evidence and exported dataset without loading weights")
            return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1" or os.environ.get("MUJOCO_EGL_DEVICE_ID") != "0":
        raise ValueError("Use CUDA_VISIBLE_DEVICES=0,1 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl HF_HUB_OFFLINE=1")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Require validated EGL and offline model loading")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Two CUDA GPUs required")
    torch.cuda.set_device(1)
    expected_sha = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"
    if digest(args.checkpoint / "model.safetensors") != expected_sha:
        raise ValueError("Checkpoint hash differs from pinned pretrained weights")
    provenance = json.loads(importlib.metadata.distribution("lerobot").read_text("direct_url.json"))
    commit = "146d8474932c8b566717eb2738ba459337ab71c4"
    if provenance["vcs_info"]["commit_id"] != commit:
        raise ValueError("LeRobot source pin changed")
    from scripts.libero_smoke import prepare_config_and_assets
    prepare_config_and_assets()
    from libero.libero.benchmark import get_benchmark_dict
    from thesis_vla.libero import LiberoTaskConfig, LiberoVectorAdapter, create_vector_env
    from thesis_vla.pi05 import Pi05PolicyAdapter
    from thesis_vla.placement import verify_prediction_gate
    config = LiberoTaskConfig()
    task = str(get_benchmark_dict()[config.suite]().get_task(0).language)
    env = LiberoVectorAdapter(create_vector_env(config, gl="egl"))
    session = args.output / "sessions" / str(time.time_ns())
    session.mkdir(parents=True)
    try:
        if args.first_state + args.total > len(env.env.envs[0].unwrapped._init_states):
            raise ValueError("Plan exceeds available saved initial states")
        started = time.perf_counter()
        policy = Pi05PolicyAdapter.from_pretrained(args.checkpoint, task=task, env_config=config,
            execute_steps=10, local_files_only=True, device="cuda:1", dtype=None, cpu_output_heads=True)
        identity = {"suite": config.suite, "task_id": 0, "fps": 20, "renderer": "egl",
            "checkpoint_id": "lerobot/pi05_libero_finetuned_v044",
            "checkpoint_revision": "7152f13e34c6c115d808aea0e5d08aca74eac14b", "checkpoint_sha256": expected_sha,
            "processor_file_sha256": {p.name: digest(p) for p in args.checkpoint.glob("policy_*") if p.is_file()},
            "lerobot_commit": commit, "torch_version": torch.__version__, "device": "cuda:1",
            "cuda_visible_devices": "0,1", "egl_device_id": 0, "execute_steps": 10,
            "dtype": str(policy.policy.config.dtype), "placement": policy.policy.placement_report,
            "normalization": {k: str(v) for k,v in policy.policy.config.normalization_mapping.items()},
            "source_file_sha256": {name: digest(Path(__file__).resolve().parents[1]/name) for name in (
                "scripts/benchmark_libero_pi05.py", "thesis_vla/benchmark.py", "thesis_vla/evaluation.py",
                "thesis_vla/pi05.py", "thesis_vla/placement.py", "thesis_vla/libero.py")},
            "state_protocol": "explicit saved state index per episode; gate reset is overridden, no index offset"}
        atomic_json(session / "load.json", {"load_seconds": time.perf_counter()-started, "identity": identity})
        observation, _ = env.reset(seed=args.base_seed, options={"saved_init_state_index": args.first_state})
        gate = verify_prediction_gate(policy, observation, seed=args.base_seed, reference_head_device="cuda:0")
        atomic_json(session / "prediction_gate.json", gate)
        print("PRETRAINED_GATE_PASSED", flush=True)
        result = run_benchmark(env, policy, args.output, configuration=identity, task=task,
            total=args.total, pilot=args.pilot, base_seed=args.base_seed, first_state=args.first_state,
            stop_after=args.stop_after, pilot_validator=lambda: export_dataset(args.output))
        export_dataset(args.output)
        print(json.dumps({k:result[k] for k in ("status","completed","successes","failures","success_rate","wilson_95")}))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
