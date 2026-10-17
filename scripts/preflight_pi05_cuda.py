#!/usr/bin/env python3
"""Weight-free preflight for the one-episode pretrained pi0.5 LIBERO run."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CHECKPOINT_BYTES = 7_470_000_000  # Hub displays model.safetensors as 7.47 GB.
DISK_HEADROOM_FACTOR = 2


def _existing_parent(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--gl", choices=("osmesa", "egl", "glfw"), default="osmesa")
    args = parser.parse_args()

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

    print("PREFLIGHT_OK: CUDA initialized, disk headroom available, renderer smoke passed")
    print("NOTE: VRAM values are reported for host review; successful model loading is the definitive inference-memory check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
