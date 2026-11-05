"""Chunked pi0.5 inference adapter with explicit prediction/execution logging."""
from collections import deque
import hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np

PINNED_PI05_MODEL_SHA256 = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"

@dataclass(frozen=True)
class ActionProposal:
    predicted_action: np.ndarray
    executed_action: np.ndarray
    chunk_id: int
    chunk_index: int
    is_chunk_start: bool
    passive_gp_prediction: dict | None = None
    assistance_request: dict | None = None

    @property
    def action(self):
        return self.executed_action

class Pi05PolicyAdapter:
    """Bridge batched LeRobot inference to LIBERO's seven-dimensional action."""
    def __init__(self, policy, prepare_batch, postprocess_action=lambda x: x, *, task, execute_steps=10,
                 passive_gp=None, monitoring_mode="passive_monitoring", monitoring_thresholds=None,
                 random_schedule=None):
        if execute_steps < 1:
            raise ValueError("execute_steps must be positive")
        self.policy, self.prepare_batch = policy, prepare_batch
        self.postprocess_action, self.task = postprocess_action, task
        self.execute_steps = execute_steps
        self._queue = deque()
        self._chunk_id = self._chunk_index = 0
        self.passive_gp = passive_gp
        self.monitoring_mode = monitoring_mode
        self.monitoring_thresholds = dict(monitoring_thresholds or {})
        self.random_schedule = random_schedule
        self._boundary_index = 0
        self._chunk_passive_gp_prediction = None
        self._chunk_assistance_request = None

    @classmethod
    def from_pretrained(cls, checkpoint, *, task, execute_steps=10, local_files_only=True,
                        env_config=None, device=None, dtype=None, cpu_output_heads=False,
                        meta_init=False, verify_placement_integrity=False, expected_weights_sha256=None,
                        passive_gp=None, monitoring_mode="passive_monitoring",
                        monitoring_thresholds=None, random_schedule=None):
        """Load a local checkpoint on demand; never download weights by default."""
        if local_files_only and not Path(checkpoint).exists():
            raise FileNotFoundError(f"local checkpoint does not exist: {checkpoint}")
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.configs import PreTrainedConfig
        from safetensors.torch import load_file
        import torch

        checkpoint = Path(checkpoint)
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"local checkpoint weights not found: {weights}")
        if expected_weights_sha256 is not None:
            digest = hashlib.sha256()
            with weights.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_weights_sha256:
                raise ValueError(f"checkpoint SHA-256 mismatch: expected {expected_weights_sha256}, got {actual_sha256}")
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=local_files_only)
        # Construct/load on CPU to avoid a second GPU copy. The pinned upstream
        # from_pretrained catches loading errors and can return random weights;
        # use its exact key conversion with strict loading and propagate failures.
        # Meta initialization avoids allocating random CPU parameters beside the
        # loaded checkpoint in memory-limited WSL environments.
        config.device = "meta" if meta_init else "cpu"
        config.compile_model = False
        config.gradient_checkpointing = False
        if dtype is not None:
            config.dtype = dtype
        if meta_init:
            with torch.device("meta"):
                policy = PI05Policy(config)
            if any(not parameter.is_meta for parameter in policy.parameters()):
                raise RuntimeError("meta initialization produced materialized parameters")
        else:
            policy = PI05Policy(config)
        state = load_file(str(weights), device="cpu")
        converted = policy._convert_openpi_state_dict(state)
        policy.load_state_dict(converted, strict=True, assign=meta_init)
        del converted, state
        if any(parameter.is_meta for parameter in policy.parameters()):
            raise RuntimeError("strict checkpoint loading left meta parameters")
        if meta_init:
            from thesis_vla.placement import materialize_pi05_meta_buffers
            policy.meta_buffer_reinitialization = materialize_pi05_meta_buffers(policy)
        config.device = "cpu"
        policy.eval()
        if cpu_output_heads:
            from thesis_vla.placement import place_action_policy
            policy.placement_report = place_action_policy(policy, device, verify_integrity=verify_placement_integrity)
            policy.config.device = device
        elif device is not None:
            policy.to(device=device)
            policy.config.device = device
        return cls.from_lerobot_policy(
            policy, task=task, execute_steps=execute_steps, env_config=env_config,
            pretrained_path=checkpoint, passive_gp=passive_gp,
            monitoring_mode=monitoring_mode, monitoring_thresholds=monitoring_thresholds,
            random_schedule=random_schedule,
        )

    @classmethod
    def from_lerobot_policy(cls, policy, *, task, execute_steps=10, env_config=None,
                            pretrained_path=None, passive_gp=None, monitoring_mode="passive_monitoring",
                            monitoring_thresholds=None, random_schedule=None):
        """Wire an existing LeRobot π0.5 policy to the LIBERO evaluator.

        This constructor permits processor/contract tests with a mock inference
        object that carries a real PI05Config, without loading checkpoint weights.
        """
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.utils.constants import ACTION
        from thesis_vla.libero import LiberoTaskConfig, make_lerobot_env_config
        state_capture = {}

        preprocessor, policy_postprocessor = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=pretrained_path,
            preprocessor_overrides={"device_processor": {"device": policy.config.device}})
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=make_lerobot_env_config(env_config or LiberoTaskConfig()),
            policy_cfg=policy.config)

        def prepare(raw_observation, instruction):
            batch = preprocess_observation({k: _add_batch(v) for k, v in raw_observation.items()})
            batch["task"] = [instruction]
            batch = env_preprocessor(batch)
            state_capture["physical"] = batch["observation.state"].detach().cpu().float().numpy()[0].copy()
            batch = preprocessor(batch)
            state_capture["vla_normalized"] = batch["observation.state"].detach().cpu().float().numpy()[0].copy()
            return batch

        def process_action(chunk):
            return _process_chunk(chunk, policy_postprocessor, env_postprocessor, ACTION)

        adapter = cls(policy, prepare, process_action, task=task, execute_steps=execute_steps,
                      passive_gp=passive_gp, monitoring_mode=monitoring_mode,
                      monitoring_thresholds=monitoring_thresholds, random_schedule=random_schedule)
        adapter.prediction_state_capture = state_capture
        return adapter

    def act(self, observation):
        if not self._queue:
            batch = self.prepare_batch(observation, self.task)
            prediction = self.policy.predict_action_chunk(batch)
            raw = prediction.detach().cpu().numpy() if hasattr(prediction, "detach") else np.asarray(prediction)
            if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 7:
                raise ValueError(f"expected chunk shape (1, T, 7), got {raw.shape}")
            if raw.shape[1] == 0 or not np.isfinite(raw).all():
                finite = raw[np.isfinite(raw)]
                detail = {
                    "shape": list(raw.shape),
                    "nan_count": int(np.isnan(raw).sum()),
                    "positive_inf_count": int(np.isposinf(raw).sum()),
                    "negative_inf_count": int(np.isneginf(raw).sum()),
                    "finite_min": float(finite.min()) if finite.size else None,
                    "finite_max": float(finite.max()) if finite.size else None,
                    "prediction_state_finite": bool(np.isfinite(
                        getattr(self, "prediction_state_capture", {}).get("vla_normalized", [])
                    ).all()),
                }
                raise ValueError(f"policy returned empty or non-finite actions: {detail}")
            self.last_predicted_chunk = raw.astype(np.float32, copy=True)
            self._chunk_passive_gp_prediction = None
            self._chunk_assistance_request = None
            boundary_index = self._boundary_index
            self._boundary_index += 1
            if self.passive_gp is not None:
                state = getattr(self, "prediction_state_capture", {}).get("vla_normalized")
                state = np.asarray(state, dtype=np.float64) if state is not None else None
                if state is None or state.shape != (8,) or not np.isfinite(state).all():
                    raise ValueError("Passive GP requires the finite checkpoint-normalized 8D prediction state")
                gp_started = __import__("time").perf_counter()
                gp_mean, gp_std = self.passive_gp.predict(state)
                round_trip_seconds = __import__("time").perf_counter() - gp_started
                gp_mean = np.asarray(gp_mean, dtype=np.float64)
                gp_std = np.asarray(gp_std, dtype=np.float64)
                if gp_mean.shape != (7,) or gp_std.shape != (7,) or not np.isfinite(gp_mean).all() or not np.isfinite(gp_std).all() or np.any(gp_std <= 0):
                    raise ValueError("Passive GP must return finite seven-action means and positive standard deviations")
                self._chunk_passive_gp_prediction = {
                    "input_state_vla_normalized": state.tolist(),
                    "mean_raw_normalized_action": gp_mean.tolist(),
                    "predictive_std_raw_normalized_action": gp_std.tolist(),
                    "target_representation": "checkpoint-normalized pi0.5 horizon-zero action",
                    "variance_definition": "latent posterior variance plus fixed observation noise",
                    "model_sha256": getattr(self.passive_gp, "model_sha256", None),
                    "worker_prediction_seconds": getattr(self.passive_gp, "last_worker_prediction_seconds", None),
                    "worker_round_trip_seconds": getattr(self.passive_gp, "last_round_trip_seconds", round_trip_seconds),
                }
                if callable(getattr(self.passive_gp, "monitoring_scores", None)):
                    scores = self.passive_gp.monitoring_scores(state, raw[0, 0], gp_mean, gp_std)
                    from thesis_vla.gp.monitoring import MonitoringSignals, decide_assistance
                    signal_values = MonitoringSignals(
                        float(scores["gp_uncertainty_standardized_rms"]),
                        float(scores["gp_vla_disagreement_standardized_rms"]),
                        float(scores["nearest_training_state_distance"]),
                    )
                    decision = decide_assistance(
                        self.monitoring_mode, signal_values, self.monitoring_thresholds,
                        episode_id=str(getattr(self, "episode_id", "episode")),
                        boundary_index=boundary_index, is_chunk_boundary=True,
                        random_schedule=self.random_schedule,
                    )
                    self._chunk_passive_gp_prediction["monitoring_scores"] = scores
                    self._chunk_assistance_request = {
                        "mode": decision.mode, "requested": decision.requested,
                        "reason": decision.reason,
                        "request": (vars(decision.request) if decision.request is not None else None),
                        "control_effect": "none; request-only monitor",
                    }
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
        return ActionProposal(predicted, executed, self._chunk_id, index, index == 0,
                              self._chunk_passive_gp_prediction if index == 0 else None,
                              self._chunk_assistance_request if index == 0 else None)

    def clear_action_queue(self):
        """Discard stale actions and clear policy temporal inference state."""
        self._queue.clear()
        self._chunk_index = 0
        self._chunk_passive_gp_prediction = None
        self._chunk_assistance_request = None
        self._boundary_index = 0
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
