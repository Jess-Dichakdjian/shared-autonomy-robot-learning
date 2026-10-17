"""Reproducible LIBERO episode evaluation with JSONL, metrics, and video output."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import time
from pathlib import Path
from typing import Any
import numpy as np


def make_lerobot_mock_adapter(*, task: str, env_config, policy, execute_steps: int = 10):
    """Use LeRobot's real LIBERO image/state/action processors without model assets.

    The checkpoint-specific policy processor is skipped because constructing
    LeRobot's pi0.5 text processor fetches a gated PaliGemma tokenizer.
    """
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.constants import ACTION
    from thesis_vla.libero import make_lerobot_env_config
    from thesis_vla.pi05 import Pi05PolicyAdapter

    cfg = PI05Config(device="cpu")
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=make_lerobot_env_config(env_config), policy_cfg=cfg)

    def prepare(raw_observation, instruction):
        batch = preprocess_observation({k: _add_batch(v) for k, v in raw_observation.items()})
        batch["task"] = [instruction]
        return env_preprocessor(batch)

    def postprocess(chunk):
        return _process_chunk(chunk, lambda action: action, env_postprocessor, ACTION)

    return Pi05PolicyAdapter(policy, prepare, postprocess, task=task, execute_steps=execute_steps)


def run_evaluation(env, policy, output_dir: str | Path, *, episodes: int, base_seed: int,
                   task: str, configuration: dict[str, Any], max_steps: int | None = None,
                   record_video: bool = True):
    """Evaluate a one-environment LiberoVectorAdapter and save full rollout artifacts."""
    if episodes < 1 or base_seed < 0 or (max_steps is not None and max_steps < 1):
        raise ValueError("episodes and seed must be non-negative; max_steps must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_record = _jsonable(configuration)
    all_episodes = []
    step_path, episode_path = output / "steps.jsonl", output / "episodes.jsonl"

    with step_path.open("w", encoding="utf-8") as steps_file, episode_path.open("w", encoding="utf-8") as episode_file:
        for episode_index in range(episodes):
            seed = base_seed + episode_index
            policy.reset()
            observation, reset_info = env.reset(seed=seed)
            frames = [_camera_montage(observation)] if record_video else []
            total_reward, success = 0.0, _success_value(reset_info)
            terminated = truncated = False
            step_count = 0
            video_path = output / f"episode_{episode_index:03d}.mp4"
            while not (terminated or truncated) and (max_steps is None or step_count < max_steps):
                import torch
                inference_started = time.perf_counter()
                with torch.inference_mode():
                    proposal = policy.act(observation)
                inference_seconds = time.perf_counter() - inference_started
                next_observation, reward, terminated, truncated, info = env.step(proposal.executed_action)
                total_reward += reward
                step_count += 1
                found_success = _success_value(info)
                if found_success is not None:
                    success = found_success
                if record_video:
                    frames.append(_camera_montage(next_observation))
                record = {
                    "episode": episode_index, "step": step_count - 1, "seed": seed, "task": task,
                    "predicted_action": proposal.predicted_action, "executed_action": proposal.executed_action,
                    "chunk_id": proposal.chunk_id, "chunk_index": proposal.chunk_index,
                    "chunk_start": proposal.is_chunk_start, "inference_seconds": inference_seconds,
                    "robot_state": _robot_state(observation), "next_robot_state": _robot_state(next_observation),
                    "reward": reward, "terminated": terminated, "truncated": truncated,
                    "success": found_success, "configuration": config_record,
                }
                steps_file.write(json.dumps(_jsonable(record), separators=(",", ":")) + "\n")
                observation = next_observation

            if record_video and frames:
                _write_video(video_path, frames, int(configuration.get("fps", 20)))
            episode_record = {
                "episode": episode_index, "seed": seed, "task": task, "success": success,
                "return": total_reward, "steps": step_count, "terminated": terminated,
                "truncated": truncated, "video": video_path.name if record_video else None,
                "reset_info": reset_info, "configuration": config_record,
            }
            episode_file.write(json.dumps(_jsonable(episode_record), separators=(",", ":")) + "\n")
            all_episodes.append(episode_record)

    known = [item["success"] for item in all_episodes if item["success"] is not None]
    returns = np.asarray([item["return"] for item in all_episodes], dtype=np.float64)
    metrics = {
        "task": task, "episodes": episodes, "base_seed": base_seed, "successes_observed": len(known),
        "success_rate": float(np.mean(known)) if known else None,
        "mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean([item["steps"] for item in all_episodes])),
        "episode_results": all_episodes, "configuration": config_record,
        "process_peak_rss_bytes": _peak_rss_bytes(),
    }
    (output / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2) + "\n", encoding="utf-8")
    return metrics


def _process_chunk(chunk, policy_postprocessor, env_postprocessor, action_key):
    import torch
    chunk = torch.as_tensor(chunk)
    result = []
    for step in chunk.unbind(dim=1):
        result.append(env_postprocessor({action_key: policy_postprocessor(step)})[action_key])
    return torch.stack(result, dim=1)


def _add_batch(value):
    if isinstance(value, dict):
        return {key: _add_batch(item) for key, item in value.items()}
    return np.asarray(value)[None, ...]


def _camera_montage(observation):
    pixels = observation.get("pixels", {})
    views = [pixels[key] for key in ("image", "image2") if key in pixels]
    if not views:
        raise ValueError("LIBERO observation must include at least one RGB camera image")
    normalized = []
    for view in views:
        image = np.asarray(view, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"expected HWC RGB image, got {image.shape}")
        normalized.append(image)
    if len(normalized) == 1:
        return normalized[0]
    if normalized[0].shape[0] != normalized[1].shape[0]:
        raise ValueError("camera image heights differ")
    return np.concatenate(normalized, axis=1)


def _write_video(path: Path, frames: list[np.ndarray], fps: int):
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def _robot_state(observation):
    return _jsonable(observation.get("robot_state", observation.get("observation.state")))


def _success_value(info):
    if not isinstance(info, dict):
        return None
    for key in ("is_success", "success", "task_success"):
        value = info.get(key)
        if value is not None:
            try:
                return bool(np.asarray(value).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                pass
    for value in info.values():
        if isinstance(value, dict):
            found = _success_value(value)
            if found is not None:
                return found
    return None


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _peak_rss_bytes():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, AttributeError):
        return None
    # Linux reports KiB; macOS reports bytes.
    return int(peak if __import__("sys").platform == "darwin" else peak * 1024)
