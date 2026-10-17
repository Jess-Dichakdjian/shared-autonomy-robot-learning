"""Checkpoint-free LIBERO OSMesa reset/action smoke test.

Requires the LIBERO geometry assets to be cached first; this script never
downloads them implicitly and never loads a learned policy.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import sysconfig

import numpy as np


def prepare_config_and_assets() -> None:
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path.home() / ".cache/shared-autonomy-robot-learning/libero-config"))
    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    package_root = Path(sysconfig.get_paths()["purelib"]) / "libero" / "libero"
    if not config_file.exists():
        import yaml
        paths = {
            "benchmark_root": str(package_root),
            "bddl_files": str(package_root / "bddl_files"),
            "init_states": str(package_root / "init_files"),
            "datasets": str(package_root.parent / "datasets"),
            "assets": str(package_root / "assets"),
        }
        config_file.write_text(yaml.safe_dump(paths), encoding="utf-8")

    local_assets = package_root / "assets"
    cached_assets = Path.home() / ".cache/libero/assets"
    assets = local_assets if local_assets.is_dir() else cached_assets
    required = ("articulated_objects", "stable_scanned_objects", "turbosquid_objects", "stable_hope_objects")
    missing = [name for name in required if not (assets / name).is_dir()]
    if missing:
        raise RuntimeError(
            "LIBERO simulation assets are not cached. Explicitly download the "
            "official geometry/texture assets before running this test; "
            f"missing directories: {missing}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gl", choices=("egl", "osmesa", "glfw"), default="osmesa")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()

    os.environ["MUJOCO_GL"] = args.gl
    prepare_config_and_assets()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from thesis_vla.libero import LiberoTaskConfig, create_vector_env

    env = create_vector_env(LiberoTaskConfig(suite=args.suite, task_id=args.task_id), gl=args.gl)
    try:
        observation, _info = env.reset(seed=0)
        images = observation.get("pixels", {})
        required = {"image", "image2"}
        missing = required.difference(images)
        if missing:
            raise RuntimeError(f"LIBERO reset missing camera images: {sorted(missing)}")
        expected = (1, 256, 256, 3)
        shapes = {key: tuple(images[key].shape) for key in sorted(required)}
        if any(shapes[key] != expected for key in required):
            raise RuntimeError(f"unexpected camera shapes: {shapes}")
        action = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
        _observation, _reward, _terminated, _truncated, _info = env.step(action)
        print(f"LIBERO_SMOKE_OK suite={args.suite} task={args.task_id} gl={args.gl} cameras={shapes}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
