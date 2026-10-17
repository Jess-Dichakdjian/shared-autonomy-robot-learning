"""Run the checkpoint-free mock-policy LIBERO evaluator."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/mock_libero"))
    parser.add_argument("--gl", choices=("osmesa", "egl", "glfw"), default="osmesa")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from scripts.libero_smoke import prepare_config_and_assets
    prepare_config_and_assets()
    from thesis_vla.evaluation import make_lerobot_mock_adapter, run_evaluation
    from thesis_vla.libero import LiberoTaskConfig, LiberoVectorAdapter, create_vector_env

    config = LiberoTaskConfig(suite=args.suite, task_id=args.task_id)
    benchmark_name = args.suite
    try:
        from libero.libero.benchmark import get_benchmark_dict
        task_object = get_benchmark_dict()[args.suite]().get_task(args.task_id)
        instruction = str(task_object.language)
    except Exception:
        instruction = f"{args.suite} task {args.task_id}"
    raw_env = create_vector_env(config, gl=args.gl)
    env = LiberoVectorAdapter(raw_env)

    class ZeroChunkPolicy:
        def __init__(self, chunk=50):
            self.chunk, self.calls, self.resets = chunk, 0, 0
        def predict_action_chunk(self, batch):
            self.calls += 1
            actions = np.zeros((1, self.chunk, 7), dtype=np.float32)
            actions[..., 6] = -1.0
            return actions
        def reset(self):
            self.resets += 1

    mock = ZeroChunkPolicy()
    policy = make_lerobot_mock_adapter(task=instruction, env_config=config, policy=mock,
                                       execute_steps=args.execute_steps)
    configuration = {
        "suite": args.suite, "task_id": args.task_id, "task_instruction": instruction,
        "width": config.width, "height": config.height, "fps": config.fps,
        "control_mode": config.control_mode, "camera_names": config.camera_name.split(","),
        "renderer": args.gl, "policy": "mock_zero_chunk", "chunk_length": mock.chunk,
        "execute_steps": args.execute_steps, "seed_rule": "base_seed + episode_index",
    }
    try:
        metrics = run_evaluation(env, policy, args.output, episodes=args.episodes,
                                 base_seed=args.seed, task=instruction, configuration=configuration,
                                 max_steps=args.max_steps, record_video=not args.no_video)
    finally:
        env.close()
    print(f"EVALUATION_OK episodes={metrics['episodes']} mean_steps={metrics['mean_steps']} "
          f"success_rate={metrics['success_rate']} output={args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
