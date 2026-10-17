"""Opt-in integration test against the installed LIBERO simulator and cached assets."""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIBERO_INTEGRATION") != "1",
    reason="set RUN_LIBERO_INTEGRATION=1 to run the real OSMesa LIBERO integration test",
)


def test_real_libero_mock_evaluation_writes_rollout_artifacts(tmp_path):
    from scripts.libero_smoke import prepare_config_and_assets
    from thesis_vla.evaluation import make_lerobot_mock_adapter, run_evaluation
    from thesis_vla.libero import LiberoTaskConfig, LiberoVectorAdapter, create_vector_env

    os.environ["MUJOCO_GL"] = "osmesa"
    prepare_config_and_assets()
    config = LiberoTaskConfig(suite="libero_spatial", task_id=0)
    from libero.libero.benchmark import get_benchmark_dict
    task = str(get_benchmark_dict()[config.suite]().get_task(config.task_id).language)
    raw_env = create_vector_env(config, gl="osmesa")
    env = LiberoVectorAdapter(raw_env)

    class MockChunkPolicy:
        def predict_action_chunk(self, batch):
            assert batch["task"] == [task]
            action = np.zeros((1, 4, 7), dtype=np.float32)
            action[..., 6] = -1
            return action
        def reset(self):
            pass

    policy = make_lerobot_mock_adapter(
        task=task, env_config=config, policy=MockChunkPolicy(), execute_steps=2
    )
    try:
        result = run_evaluation(
            env, policy, tmp_path, episodes=1, base_seed=900, task=task,
            configuration={"suite": config.suite, "task_id": config.task_id, "fps": config.fps,
                           "renderer": "osmesa"},
            max_steps=2, record_video=True,
        )
    finally:
        env.close()

    assert result["episodes"] == 1
    assert result["mean_steps"] == 2
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "steps.jsonl").is_file()
    assert (tmp_path / "episodes.jsonl").is_file()
    assert (tmp_path / "episode_000.mp4").stat().st_size > 0
