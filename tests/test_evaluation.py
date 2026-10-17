import json

import numpy as np

from thesis_vla.evaluation import run_evaluation
from thesis_vla.pi05 import ActionProposal


class FakePolicy:
    def __init__(self):
        self.resets = 0
        self.calls = 0
    def reset(self):
        self.resets += 1
    def act(self, observation):
        self.calls += 1
        raw = np.array([2.0, 0.1, 0, 0, 0, 0, -1], dtype=np.float32)
        executed = np.array([1.0, 0.1, 0, 0, 0, 0, -1], dtype=np.float32)
        return ActionProposal(raw, executed, 1, self.calls - 1, self.calls == 1)


class FakeEnv:
    def __init__(self):
        self.seeds = []
        self.actions = []
        self.step_index = 0
    def reset(self, *, seed):
        self.seeds.append(seed)
        self.step_index = 0
        return self._obs(), {"initial_state": "test"}
    def step(self, action):
        self.actions.append(action.copy())
        self.step_index += 1
        return self._obs(), 1.0, self.step_index == 2, False, {"is_success": self.step_index == 2}
    def _obs(self):
        return {
            "pixels": {
                "image": np.zeros((8, 8, 3), dtype=np.uint8),
                "image2": np.full((8, 8, 3), self.step_index, dtype=np.uint8),
            },
            "robot_state": {"eef": {"pos": np.array([self.step_index, 0, 0], dtype=np.float32)}},
        }


def test_evaluator_logs_seed_actions_state_outcome_metrics_and_video(tmp_path):
    env, policy = FakeEnv(), FakePolicy()
    metrics = run_evaluation(
        env, policy, tmp_path, episodes=2, base_seed=41, task="pick the block",
        configuration={"fps": 10, "suite": "fixture"}, record_video=True,
    )
    assert env.seeds == [41, 42]
    assert policy.resets == 2
    assert metrics["success_rate"] == 1.0
    assert metrics["mean_return"] == 2.0
    assert metrics["mean_steps"] == 2.0
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "episode_000.mp4").stat().st_size > 0
    step_rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    episode_rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert len(step_rows) == 4 and len(episode_rows) == 2
    assert step_rows[0]["predicted_action"][0] == 2.0
    assert step_rows[0]["executed_action"][0] == 1.0
    assert step_rows[0]["robot_state"]["eef"]["pos"] == [0.0, 0.0, 0.0]
    assert step_rows[0]["chunk_start"] is True
    assert episode_rows[1]["seed"] == 42 and episode_rows[1]["success"] is True
    assert episode_rows[0]["video"] == "episode_000.mp4"


def test_evaluator_honors_step_cap_and_optional_video(tmp_path):
    env, policy = FakeEnv(), FakePolicy()
    metrics = run_evaluation(
        env, policy, tmp_path, episodes=1, base_seed=0, task="fixture",
        configuration={"fps": 20}, max_steps=1, record_video=False,
    )
    assert metrics["mean_steps"] == 1
    assert metrics["success_rate"] == 0.0
    assert not list(tmp_path.glob("*.mp4"))
    row = json.loads((tmp_path / "steps.jsonl").read_text().strip())
    assert row["truncated"] is False and row["terminated"] is False
