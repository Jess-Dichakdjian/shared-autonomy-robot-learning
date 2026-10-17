"""Chunked pi0.5 inference adapter with explicit prediction/execution logging."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class ActionProposal:
    predicted_action: np.ndarray
    executed_action: np.ndarray
    chunk_id: int
    chunk_index: int
    is_chunk_start: bool

    @property
    def action(self):
        return self.executed_action

class Pi05PolicyAdapter:
    """Bridge batched LeRobot inference to LIBERO's seven-dimensional action."""
    def __init__(self, policy, prepare_batch, postprocess_action=lambda x: x, *, task, execute_steps=10):
        if execute_steps < 1:
            raise ValueError("execute_steps must be positive")
        self.policy, self.prepare_batch = policy, prepare_batch
        self.postprocess_action, self.task = postprocess_action, task
        self.execute_steps = execute_steps
        self._queue = deque()
        self._chunk_id = self._chunk_index = 0

    @classmethod
    def from_pretrained(cls, checkpoint, *, task, execute_steps=10, local_files_only=True,
                        env_config=None, device=None, dtype=None):
        """Load a local checkpoint on demand; never download weights by default."""
        if local_files_only and not Path(checkpoint).exists():
            raise FileNotFoundError(f"local checkpoint does not exist: {checkpoint}")
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        policy = PI05Policy.from_pretrained(checkpoint, local_files_only=local_files_only)
        if device is not None or dtype is not None:
            policy.to(device=device, dtype=dtype)
        if device is not None:
            policy.config.device = device
        if dtype is not None:
            policy.config.dtype = dtype
        return cls.from_lerobot_policy(
            policy, task=task, execute_steps=execute_steps, env_config=env_config,
            pretrained_path=checkpoint,
        )

    @classmethod
    def from_lerobot_policy(cls, policy, *, task, execute_steps=10, env_config=None,
                            pretrained_path=None):
        """Wire an existing LeRobot π0.5 policy to the LIBERO evaluator.

        This constructor permits processor/contract tests with a mock inference
        object that carries a real PI05Config, without loading checkpoint weights.
        """
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.utils.constants import ACTION
        from thesis_vla.libero import LiberoTaskConfig, make_lerobot_env_config

        preprocessor, policy_postprocessor = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=pretrained_path)
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=make_lerobot_env_config(env_config or LiberoTaskConfig()),
            policy_cfg=policy.config)

        def prepare(raw_observation, instruction):
            batch = preprocess_observation({k: _add_batch(v) for k, v in raw_observation.items()})
            batch["task"] = [instruction]
            return preprocessor(env_preprocessor(batch))

        def process_action(chunk):
            return _process_chunk(chunk, policy_postprocessor, env_postprocessor, ACTION)

        return cls(policy, prepare, process_action, task=task, execute_steps=execute_steps)

    def act(self, observation):
        if not self._queue:
            batch = self.prepare_batch(observation, self.task)
            prediction = self.policy.predict_action_chunk(batch)
            raw = prediction.detach().cpu().numpy() if hasattr(prediction, "detach") else np.asarray(prediction)
            if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 7:
                raise ValueError(f"expected chunk shape (1, T, 7), got {raw.shape}")
            if raw.shape[1] == 0 or not np.isfinite(raw).all():
                raise ValueError("policy returned empty or non-finite actions")
            processed = self.postprocess_action(prediction)
            processed = processed.detach().cpu().numpy() if hasattr(processed, "detach") else np.asarray(processed)
            if processed.shape != raw.shape:
                raise ValueError(f"postprocessed chunk shape {processed.shape} differs from prediction {raw.shape}")
            if not np.isfinite(processed).all():
                raise ValueError("postprocessing returned non-finite actions")
            count = min(self.execute_steps, raw.shape[1])
            self._queue.extend((raw[0, i].astype(np.float32, copy=True),
                                np.clip(processed[0, i], -1, 1).astype(np.float32, copy=True))
                               for i in range(count))
            self._chunk_id += 1
            self._chunk_index = 0
        index = self._chunk_index
        self._chunk_index += 1
        predicted, executed = self._queue.popleft()
        return ActionProposal(predicted, executed, self._chunk_id, index, index == 0)

    def clear_action_queue(self):
        """Discard stale actions and clear policy temporal inference state."""
        self._queue.clear()
        self._chunk_index = 0
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
    reset = clear_action_queue

def _process_chunk(chunk, policy_postprocessor, env_postprocessor, action_key):
    import torch
    chunk = torch.as_tensor(chunk)
    outputs = []
    for step in chunk.unbind(dim=1):
        action = policy_postprocessor(step)
        outputs.append(env_postprocessor({action_key: action})[action_key])
    return torch.stack(outputs, dim=1)

def _add_batch(value):
    if isinstance(value, dict):
        return {k: _add_batch(v) for k, v in value.items()}
    return np.asarray(value)[None, ...]
