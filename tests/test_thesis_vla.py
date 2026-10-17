import numpy as np
import pytest
from thesis_vla.libero import LiberoTaskConfig, LiberoVectorAdapter
from thesis_vla.pi05 import Pi05PolicyAdapter

def test_libero_defaults_and_validation():
    c = LiberoTaskConfig()
    assert c.suite == "libero_spatial" and c.control_mode == "relative"
    assert c.width == c.height == 256 and c.fps == 20
    assert c.camera_name == "agentview_image,robot0_eye_in_hand_image"
    with pytest.raises(ValueError):
        LiberoTaskConfig(fps=0)
    np.testing.assert_array_equal(LiberoVectorAdapter.validate_action(np.full(7, 2)), np.ones(7))
    with pytest.raises(ValueError):
        LiberoVectorAdapter.validate_action(np.zeros(8))

class FakePolicy:
    def __init__(self):
        self.calls = self.resets = 0
    def predict_action_chunk(self, batch):
        self.calls += 1
        assert batch["task"] == ["move"]
        return np.arange(35, dtype=np.float32).reshape(1, 5, 7) / 10
    def reset(self):
        self.resets += 1

class FakeVectorEnv:
    single_action_space = type("Space", (), {"shape": (7,)})()
    def __init__(self):
        self.last_action = None
        self.closed = False
    def reset(self, seed=None, options=None):
        return {"camera": np.zeros((1, 2, 2, 3), np.uint8)}, {"seed": np.array([seed or 0])}
    def step(self, action):
        self.last_action = action
        return {"camera": np.ones((1, 2, 2, 3), np.uint8)}, np.array([1.]), np.array([False]), np.array([False]), {"ok": np.array([True])}
    def close(self):
        self.closed = True

def test_vector_adapter_unbatches_and_clips():
    env = FakeVectorEnv()
    adapter = LiberoVectorAdapter(env)
    obs, info = adapter.reset(seed=8)
    assert obs["camera"].shape == (2, 2, 3) and info["seed"] == 8
    obs, reward, term, trunc, info = adapter.step(np.full(7, 2))
    assert obs["camera"].shape == (2, 2, 3) and reward == 1.0
    assert not term and not trunc and info["ok"]
    np.testing.assert_array_equal(env.last_action, np.ones((1, 7)))
    adapter.close()
    assert env.closed

def test_chunk_queue_and_intervention_reset():
    p = FakePolicy()
    a = Pi05PolicyAdapter(p, lambda obs, task: {"task": [task]}, task="move", execute_steps=2)
    x, y = a.act({}), a.act({})
    assert x.chunk_id == y.chunk_id == 1 and x.is_chunk_start and not y.is_chunk_start
    a.clear_action_queue()
    assert p.resets == 1 and a.act({}).chunk_id == 2 and p.calls == 2

def test_invalid_chunk_rejected():
    p = FakePolicy()
    p.predict_action_chunk = lambda batch: np.zeros((1, 3, 6))
    with pytest.raises(ValueError, match="shape"):
        Pi05PolicyAdapter(p, lambda o, t: {}, task="x").act({})

def test_official_lerobot_libero_observation_preprocessing_contract():
    import torch
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.utils import preprocess_observation

    pixels = np.arange(60, dtype=np.uint8).reshape(1, 4, 5, 3)
    raw = {
        "pixels": {"image": pixels, "image2": pixels.copy()},
        "robot_state": {
            "eef": {
                "pos": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
                "quat": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            },
            "gripper": {
                "qpos": np.array([[0.4, 0.5]], dtype=np.float32),
                "qvel": np.zeros((1, 2), dtype=np.float32),
            },
        },
    }
    observation = preprocess_observation(raw)
    env_preprocessor, _ = LiberoEnv(task="libero_spatial").get_env_processors()
    observation = env_preprocessor(observation)

    assert observation["observation.images.image"].shape == (1, 3, 4, 5)
    assert observation["observation.images.image"].dtype == torch.float32
    expected = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255
    torch.testing.assert_close(observation["observation.images.image"], torch.flip(expected, dims=[2, 3]))
    assert observation["observation.images.image2"].shape == (1, 3, 4, 5)
    assert observation["observation.state"].shape == (1, 8)
    torch.testing.assert_close(
        observation["observation.state"],
        torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4, 0.5]]),
    )

def test_create_vector_env_honors_selected_renderer(monkeypatch):
    import thesis_vla.libero as libero

    class FakeConfig:
        def create_envs(self, *, n_envs, use_async_envs):
            assert n_envs == 1 and not use_async_envs
            return {"libero_spatial": {0: "fake-env"}}

    monkeypatch.setattr(libero, "make_lerobot_env_config", lambda config: FakeConfig())
    monkeypatch.setenv("MUJOCO_GL", "egl")
    assert libero.create_vector_env(gl="osmesa") == "fake-env"
    assert __import__("os").environ["MUJOCO_GL"] == "osmesa"
