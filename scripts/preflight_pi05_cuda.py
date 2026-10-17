#!/usr/bin/env python3
"""Weight-free resource and renderer preflight for pretrained pi0.5 LIBERO."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CHECKPOINT_BYTES = 7_473_096_344
# Conservative load headroom, not an upstream-specified hardware requirement.
CPU_RAM_HEADROOM_FACTOR = 1.5
DISK_HEADROOM_FACTOR = 2


def _available_memory_bytes() -> int:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value, *_ = line.replace(":", "").split()
        if key in {"MemAvailable"}:
            values[key] = int(value) * 1024
    available = values["MemAvailable"]
    limit_path = Path("/sys/fs/cgroup/memory.max")
    current_path = Path("/sys/fs/cgroup/memory.current")
    if limit_path.exists() and current_path.exists():
        limit = limit_path.read_text().strip()
        if limit.isdigit():
            available = min(available, max(0, int(limit) - int(current_path.read_text())))
    return available


def _existing_parent(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gl", choices=("osmesa", "egl", "glfw"), default="osmesa")
    args = parser.parse_args()

    if args.device == "cuda":
        try:
            import torch
        except ImportError as exc:
            parser.error(f"PyTorch is not installed in this environment: {exc}")
        if not torch.cuda.is_available():
            print("PREFLIGHT_FAILED: torch.cuda.is_available() is false", file=sys.stderr)
            return 2
        props = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        print(f"CUDA_DEVICE={props.name}")
        print(f"CUDA_CAPABILITY={props.major}.{props.minor}")
        print(f"VRAM_TOTAL_GIB={total_bytes / 2**30:.2f}")
        print(f"VRAM_FREE_GIB={free_bytes / 2**30:.2f}")
    else:
        free_bytes = _available_memory_bytes()
        total_bytes = int(next(
            line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemTotal:")
        )) * 1024
        required_bytes = int(CHECKPOINT_BYTES * CPU_RAM_HEADROOM_FACTOR)
        print(f"CPU_RAM_TOTAL_GIB={total_bytes / 2**30:.2f}")
        print(f"CPU_RAM_AVAILABLE_GIB={free_bytes / 2**30:.2f}")
        print(f"CPU_RAM_REQUIRED_GIB={required_bytes / 2**30:.2f} (1.5x weight-file size; conservative headroom heuristic)")
        if free_bytes < required_bytes:
            print("PREFLIGHT_FAILED: insufficient RAM headroom for safe CPU model loading", file=sys.stderr)
            return 5

    disk_path = _existing_parent(args.checkpoint_dir)
    free_disk = shutil.disk_usage(disk_path).free
    required_disk = CHECKPOINT_BYTES * DISK_HEADROOM_FACTOR
    print(f"CHECKPOINT_TARGET={args.checkpoint_dir.expanduser().absolute()}")
    print(f"CHECKPOINT_SIZE_GB={CHECKPOINT_BYTES / 1e9:.2f}")
    print(f"DISK_FREE_GB={free_disk / 1e9:.2f}")
    print(f"DISK_REQUIRED_GB={required_disk / 1e9:.2f} (2x checkpoint size)")
    if free_disk < required_disk:
        print("PREFLIGHT_FAILED: insufficient disk headroom for pinned checkpoint", file=sys.stderr)
        return 3

    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, str(repo_root / "scripts" / "libero_smoke.py"),
        "--gl", args.gl, "--suite", "libero_spatial", "--task-id", "0",
    ]
    print("RENDERER_SMOKE=running LIBERO Spatial task 0")
    completed = subprocess.run(command, cwd=repo_root, check=False)
    if completed.returncode:
        print(f"PREFLIGHT_FAILED: LIBERO renderer smoke exited {completed.returncode}", file=sys.stderr)
        return 4

    print("PREFLIGHT_OK: device resources, disk headroom, and renderer smoke passed")
    if args.device == "cpu":
        print("NOTE: CPU RAM threshold is a safety heuristic, not an official LeRobot requirement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
