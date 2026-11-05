"""Thin LIBERO environment config and action adapter."""
from dataclasses import dataclass
import os
import numpy as np

@dataclass(frozen=True)
class LiberoTaskConfig:
    suite: str = "libero_spatial"
    task_id: int = 0
    width: int = 256
    height: int = 256
    fps: int = 20
    control_mode: str = "relative"
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    max_episode_steps: int | None = None

    def __post_init__(self):
        if not self.suite or self.task_id < 0:
            raise ValueError("suite must be set and task_id non-negative")
        if min(self.width, self.height, self.fps) <= 0:
            raise ValueError("dimensions and fps must be positive")
        if self.control_mode not in {"relative", "absolute"}:
            raise ValueError("control mode must be relative or absolute")

def make_lerobot_env_config(config=LiberoTaskConfig()):
    from lerobot.envs.configs import LiberoEnv
    return LiberoEnv(task=config.suite, task_ids=[config.task_id], fps=config.fps,
        episode_length=config.max_episode_steps, obs_type="pixels_agent_pos",
        render_mode="rgb_array", camera_name=config.camera_name, init_states=True,
        hard_reset=True, observation_width=config.width, observation_height=config.height,
        control_mode=config.control_mode)

def create_vector_env(config=LiberoTaskConfig(), *, gl="osmesa"):
    if gl not in {"egl", "osmesa", "glfw"}:
        raise ValueError("gl must be egl, osmesa, or glfw")
    os.environ["MUJOCO_GL"] = gl
    return make_lerobot_env_config(config).create_envs(n_envs=1, use_async_envs=False)[config.suite][config.task_id]

class LiberoVectorAdapter:
    """Validate seven-dimensional actions and unbatch one-env vector results."""
    def __init__(self, env):
        self.env = env
        if tuple(env.single_action_space.shape) != (7,):
            raise ValueError("expected LIBERO single action shape (7,)")

    @staticmethod
    def validate_action(action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("LIBERO action must be a finite seven-vector")
        return np.clip(action, -1.0, 1.0)

    @staticmethod
    def _first(value):
        if isinstance(value, dict):
            return {k: LiberoVectorAdapter._first(v) for k, v in value.items()}
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == 1:
            return value[0]
        return value

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        state_index = options.pop("saved_init_state_index", None)
        state_metadata = {}
        if state_index is not None:
            import hashlib
            single = self.env.envs[0].unwrapped
            states = single._init_states
            if states is None or not 0 <= state_index < len(states):
                raise ValueError("Saved initial-state index out of range")
            single.init_state_id = state_index
            state = np.asarray(states[state_index])
            state_metadata = {"saved_init_state_index": state_index,
                              "saved_init_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                              "saved_init_state_count": len(states)}
        scenario = options.pop("scene_perturbation", None)
        obs, info = self.env.reset(seed=seed, options=options)
        info = self._first(info)
        info.update(state_metadata)
        if scenario is not None:
            from thesis_vla.perturbations import apply_scene_perturbation
            if scenario.get("seed") != seed or scenario.get("saved_init_state_index") != state_index:
                raise ValueError("scenario seed/state must match the reset request")
            expected_state_sha = scenario.get("saved_init_state_sha256")
            if expected_state_sha and expected_state_sha != info.get("saved_init_state_sha256"):
                raise ValueError("scenario saved-state hash differs from the restored LIBERO state")
            single = self.env.envs[0].unwrapped
            obs, scene_record = apply_scene_perturbation(single, scenario)
            info["scene_perturbation"] = scene_record
        if state_metadata and "scene_perturbation" not in info:
            info["initial_camera_sha256"] = {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
                for k, v in obs.get("pixels", {}).items()}
        return self._first(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(self.validate_action(action)[None, :])
        return (self._first(obs), float(np.asarray(reward).reshape(-1)[0]),
            bool(np.asarray(term).reshape(-1)[0]), bool(np.asarray(trunc).reshape(-1)[0]),
            self._first(info))

    def close(self):
        self.env.close()
