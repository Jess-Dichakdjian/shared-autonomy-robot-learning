"""Private browser teleoperation for LIBERO; simulator and OSMesa stay on one thread."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import secrets
import signal
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
HTML = SCRIPT_DIR / "teleop.html"
PORT = 8765
TASK_ID = 1
SEED = 1101
STATE_INDEX = 0
STATE_SHA256 = "d116842e01703fbea595511db813a3816c8d9f2302b4e979ebec4f15b7a750a2"
TASK_TEXT = "Pick up the black bowl next to the ramekin and place it on the plate."
CAMERA_IDS = {"scene": "agentview", "wrist": "robot0_eye_in_hand"}
PRESETS = {
    "fine": {"distance_m": 0.002, "max_steps": 7, "tolerance_m": 0.00035, "rotation_rad": np.deg2rad(1)},
    "normal": {"distance_m": 0.005, "max_steps": 10, "tolerance_m": 0.00055, "rotation_rad": np.deg2rad(3)},
    "coarse": {"distance_m": 0.030, "max_steps": 18, "tolerance_m": 0.0012, "rotation_rad": np.deg2rad(8)},
}
MAX_COMMAND_SECONDS = 2.2
HOLD_LEASE_SECONDS = 0.75
HOLD_MAX_SECONDS = 5.0
GRIPPER_MAX_STEPS = 20


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    a = np.asarray(value)
    if a.ndim == 0:
        x = float(a)
        if not np.isfinite(x):
            raise ValueError("non-finite simulator value")
        return x
    return a.tolist()


def _jpeg(image):
    stream = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(stream, format="JPEG", quality=84)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _jsonable_axes(axes):
    return {name: {key: np.asarray(value).astype(float).tolist() for key, value in basis.items()}
            for name, basis in axes.items()}


class TeleopSession:
    def __init__(self, env, output_dir: Path):
        self.env = env
        self.output_dir = output_dir
        self.run_id = str(uuid.uuid4())
        self.lock = threading.Lock()
        self.hold_lock = threading.Lock()
        self.active_hold = None
        self.mode = "HUMAN"
        self.epoch = 0
        self.last_seq = -1
        self.step = 0
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.last_action = None
        self.last_motion = None
        self.control_reference = "scene"
        self.wrist_display_rotation_deg = 180
        self.operator_view_history = []
        self.started_wall = time.time()
        self.log_path = output_dir / "teleoperation.jsonl"
        self.video_path = output_dir / "teleoperation.mp4"
        self.camera_dir = output_dir / "camera_raw"
        (self.camera_dir / "scene").mkdir(parents=True, exist_ok=True)
        (self.camera_dir / "wrist").mkdir(parents=True, exist_ok=True)
        self.video = None
        self.video_error = None
        self.observation = None
        self.reset_info = None
        self.last_info = {}
        self.camera_axes = {}
        self._reset_episode(initial=True)
        self._write_config()

    @property
    def raw_env(self):
        return self.env.env.envs[0].unwrapped

    @property
    def sim(self):
        return self.raw_env._env.sim

    def _update_camera_axes(self):
        from thesis_vla.teleoperation import camera_plane_basis
        result = {}
        for label, camera_name in CAMERA_IDS.items():
            cid = self.sim.model.camera_name2id(camera_name)
            rotation = np.asarray(self.sim.data.cam_xmat[cid]).reshape(3, 3).copy()
            position = np.asarray(self.sim.data.cam_xpos[cid]).copy()
            basis = camera_plane_basis(rotation)
            basis.update({"position_world": position,
                          "view_world": -rotation[:, 2].copy(),
                          "rotation_camera_to_world": rotation})
            result[label] = basis
        self.camera_axes = result

    def _write_config(self):
        try:
            import lerobot, libero, robosuite, mujoco
            versions = {"lerobot": getattr(lerobot, "__version__", "unknown"),
                        "libero": getattr(libero, "__version__", "unknown"),
                        "robosuite": getattr(robosuite, "__version__", "unknown"),
                        "mujoco": getattr(mujoco, "__version__", "unknown")}
        except Exception:
            versions = {}
        data = {"run_id": self.run_id, "suite": "libero_spatial", "task_id": TASK_ID,
                "task": TASK_TEXT, "seed": SEED, "saved_init_state_index": STATE_INDEX,
                "saved_init_state_sha256": STATE_SHA256, "renderer": "OSMesa",
                "camera_observations": {"scene": "agentview_image", "wrist": "robot0_eye_in_hand_image"},
                "camera_display_transform": {
                    "scene": "none",
                    "wrist_default_rotation_deg": 180,
                    "wrist_current_rotation_deg": self.wrist_display_rotation_deg,
                    "scope": "browser presentation only; raw RGB observations, PNGs, hashes, and policy inputs are unchanged"},
                "control_reference_default": self.control_reference,
                "operator_view_initial": {"control_reference": "scene", "wrist_display_rotation_deg": 180},
                "operator_view_current": self._operator_view(),
                "operator_view_history": self.operator_view_history,
                "presets": PRESETS, "max_command_seconds": MAX_COMMAND_SECONDS,
                "hold_heartbeat_lease_seconds": HOLD_LEASE_SECONDS,
                "hold_hard_timeout_seconds": HOLD_MAX_SECONDS,
                "gripper_max_steps": GRIPPER_MAX_STEPS, "software_versions": versions,
                "camera_extrinsics_initial": _jsonable_axes(self.camera_axes)}
        (self.output_dir / "configuration.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")

    def _save_raw_cameras(self, observation, step):
        saved = {}
        for label, key in (("scene", "image"), ("wrist", "image2")):
            image = np.asarray(observation["pixels"][key], dtype=np.uint8)
            if image.shape != (256, 256, 3):
                raise RuntimeError(f"unexpected canonical {label} RGB frame shape: {image.shape}")
            # PNG stores the untouched simulator observation; browser JPEG and video are display copies.
            path = self.camera_dir / label / f"step_{step:06d}.png"
            Image.fromarray(image).save(path, format="PNG")
            saved[label] = {"path": str(path.relative_to(self.output_dir)),
                            "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                            "shape": list(image.shape), "dtype": str(image.dtype)}
        return saved

    def _write_video_frame(self, observation):
        try:
            import cv2
            frame = np.concatenate([observation["pixels"]["image"], observation["pixels"]["image2"]], axis=1)
            if self.video is None:
                self.video = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 256))
                if not self.video.isOpened():
                    self.video.release()
                    self.video = None
                    raise RuntimeError("OpenCV MP4 writer failed to open")
            self.video.write(frame[:, :, ::-1].copy())
        except Exception as exc:
            self.video_error = str(exc)

    def _log(self, record):
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _reset_episode(self, *, initial=False):
        self.observation, self.reset_info = self.env.reset(
            seed=SEED, options={"saved_init_state_index": STATE_INDEX})
        if self.reset_info.get("saved_init_state_sha256") != STATE_SHA256:
            raise RuntimeError("pinned LIBERO initial state hash mismatch")
        self.last_info = self.reset_info
        self.step = 0
        self.reward = 0.0
        self.terminated = self.truncated = False
        self._update_camera_axes()
        captures = self._save_raw_cameras(self.observation, 0)
        self._write_video_frame(self.observation)
        self._log({"record_type": "episode_reset", "run_id": self.run_id,
                   "time_unix": time.time(), "time_monotonic": time.monotonic(),
                   "owner": "SYSTEM", "human_input": False, "episode_index": 0,
                   "task_id": TASK_ID, "seed": SEED, "saved_init_state_index": STATE_INDEX,
                   "saved_init_state_sha256": STATE_SHA256, "camera_files": captures,
                   "robot_state": _plain(self.observation["robot_state"]),
                   "operator_view": self._operator_view(),
                   "camera_extrinsics": _jsonable_axes(self.camera_axes)})

    def _state(self):
        return _plain(self.observation["robot_state"])

    def _image_display(self):
        pix = self.observation["pixels"]
        return {"scene": _jpeg(pix["image"]), "wrist": _jpeg(pix["image2"])}

    def _operator_view(self, *, control_reference=None, wrist_rotation=None):
        reference = self.control_reference if control_reference is None else control_reference
        rotation = self.wrist_display_rotation_deg if wrist_rotation is None else wrist_rotation
        return {"control_reference": reference,
                "wrist_display_rotation_deg": rotation,
                "display_transform_applied_to_control": reference == "wrist" and rotation == 180}

    def _record_operator_view_change(self, record_type):
        entry = {"record_type": record_type, "time_unix": time.time(),
                 **self._operator_view(),
                 "camera_extrinsics": _jsonable_axes(self.camera_axes)}
        self.operator_view_history.append({k: v for k, v in entry.items() if k != "camera_extrinsics"})
        self._log(entry)
        self._write_config()

    def snapshot(self):
        state = self._state()
        qpos = state["gripper"]["qpos"]
        aperture = float(sum(abs(v) for v in qpos))
        return {"mode": self.mode, "owner": self.mode, "vla_loaded": False,
                "assistance_request": "inactive; no VLA connected", "recording": True,
                "run_id": self.run_id, "epoch": self.epoch, "next_seq": self.last_seq + 1,
                "step": self.step, "episode": 0, "seed": SEED, "task_id": TASK_ID,
                "task": TASK_TEXT, "saved_init_state_index": STATE_INDEX,
                "saved_init_state_sha256": STATE_SHA256,
                "robot_state": state, "eef_position_m": state["eef"]["pos"],
                "gripper_qpos": qpos, "gripper_aperture_proxy": aperture,
                "gripper_status": self.last_action.get("gripper_status", "idle") if self.last_action else "idle",
                "reward": self.reward, "is_success": bool(self.last_info.get("is_success", False)),
                "terminated": self.terminated, "truncated": self.truncated,
                "last_action": self.last_action, "last_motion": self.last_motion,
                "control_reference": self.control_reference,
                "wrist_display_rotation_deg": self.wrist_display_rotation_deg,
                "operator_view": self._operator_view(),
                "camera_axes": _jsonable_axes(self.camera_axes),
                "video_status": self.video_error or ("recording" if self.video is not None else "ready"),
                "images": self._image_display()}

    def begin_hold(self, hold_id):
        if not isinstance(hold_id, str) or not hold_id or len(hold_id) > 100:
            raise ValueError("invalid hold identifier")
        with self.hold_lock:
            active = self.active_hold
            if active and active["id"] != hold_id and not active["cancel"].is_set():
                raise RuntimeError("another input is still active")
            if active and active["id"] == hold_id:
                if active["cancel"].is_set():
                    raise ValueError("this hold has been stopped")
                active["last_heartbeat"] = time.monotonic()
            else:
                now = time.monotonic()
                self.active_hold = {"id": hold_id, "started": now, "last_heartbeat": now,
                                    "cancel": threading.Event()}
        return True

    def heartbeat(self, hold_id):
        with self.hold_lock:
            active = self.active_hold
            if not active or active["id"] != hold_id or active["cancel"].is_set():
                return False
            active["last_heartbeat"] = time.monotonic()
            return True

    def stop_hold(self, hold_id=None):
        with self.hold_lock:
            active = self.active_hold
            if active and (hold_id is None or active["id"] == hold_id):
                active["cancel"].set()
                return True
        return False

    def end_hold(self, hold_id):
        with self.hold_lock:
            active = self.active_hold
            if active and active["id"] == hold_id:
                active["cancel"].set()
                self.active_hold = None

    def _lease_valid(self, hold_id):
        with self.hold_lock:
            active = self.active_hold
            now = time.monotonic()
            if not active or active["id"] != hold_id or active["cancel"].is_set():
                return False
            if now - active["last_heartbeat"] > HOLD_LEASE_SECONDS or now - active["started"] > HOLD_MAX_SECONDS:
                active["cancel"].set()
                return False
            return True

    def set_reference(self, reference):
        if reference not in {"world", "scene", "wrist"}:
            raise ValueError("reference must be world, scene, or wrist")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("simulator busy; try again after the action completes")
        try:
            self.stop_hold()
            self.control_reference = reference
            self._record_operator_view_change("operator_view_change")
            return self.snapshot()
        finally:
            self.lock.release()

    def set_wrist_display_rotation(self, rotation_deg):
        if isinstance(rotation_deg, bool) or rotation_deg not in {0, 180}:
            raise ValueError("wrist display rotation must be 0 or 180 degrees")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("simulator busy; try again after the action completes")
        try:
            self.stop_hold()
            self.wrist_display_rotation_deg = int(rotation_deg)
            self._record_operator_view_change("operator_view_change")
            return self.snapshot()
        finally:
            self.lock.release()

    def execute(self, body):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("simulator busy; no command was queued")
        try:
            if self.mode != "HUMAN":
                raise RuntimeError(f"control disabled in mode {self.mode}")
            if body.get("epoch") != self.epoch:
                raise ValueError("stale control epoch")
            seq = body.get("seq")
            if isinstance(seq, bool) or not isinstance(seq, int) or seq <= self.last_seq:
                raise ValueError("duplicate or stale command sequence")
            self.last_seq = seq
            hold_id = body.get("hold_id")
            if not self._lease_valid(hold_id):
                with self.hold_lock:
                    active = self.active_hold
                    state = None if active is None else {
                        "id_matches": active["id"] == hold_id,
                        "cancelled": active["cancel"].is_set(),
                        "heartbeat_age": round(time.monotonic()-active["last_heartbeat"], 3),
                        "hold_age": round(time.monotonic()-active["started"], 3)}
                raise ValueError(f"input lease expired or was released: {state}")
            input_method = body.get("input_method")
            if input_method not in {"keyboard", "button", "automated_test"}:
                raise ValueError("unknown input provenance")
            kind = body.get("kind")
            if kind == "move":
                result = self._move(body, hold_id, input_method, seq)
            elif kind == "rotate":
                result = self._rotate(body, hold_id, input_method, seq)
            elif kind == "gripper":
                result = self._gripper(body, hold_id, input_method, seq)
            else:
                raise ValueError("kind must be move, rotate, or gripper")
            self.last_action = result["last_action"]
            self.last_motion = result.get("motion")
            return self.snapshot()
        finally:
            self.lock.release()

    def _execute_env_step(self, action, *, command, step_index, planned_steps, input_method,
                          before, started_mono):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.observation = obs
        self.reward, self.terminated, self.truncated = reward, terminated, truncated
        self.last_info = info or {}
        self.step += 1
        self._update_camera_axes()
        after = self._state()
        delta = np.asarray(after["eef"]["pos"]) - np.asarray(before["eef"]["pos"])
        captures = self._save_raw_cameras(obs, self.step)
        self._write_video_frame(obs)
        self._log({"record_type": "human_action_step", "run_id": self.run_id,
                   "time_unix": time.time(), "time_monotonic": time.monotonic(),
                   "command_started_monotonic": started_mono, "step": self.step,
                   "owner": "HUMAN", "human_input": input_method != "automated_test",
                   "input_method": input_method, "command": command,
                   "controller_step_index": step_index, "planned_max_steps": planned_steps,
                   "executed_action_7d": np.asarray(action, dtype=float).tolist(),
                   "robot_state_before": before, "robot_state_after": after,
                   "measured_eef_delta_world_m": delta.astype(float).tolist(),
                   "camera_files": captures, "camera_extrinsics": _jsonable_axes(self.camera_axes),
                   "operator_view": self._operator_view(),
                   "reward": reward, "is_success": bool(self.last_info.get("is_success", False)),
                   "terminated": terminated, "truncated": truncated})
        return after, delta

    def _move(self, body, hold_id, input_method, seq):
        from thesis_vla.teleoperation import translation_action, world_delta_for_direction
        preset = body.get("preset")
        if preset not in PRESETS:
            raise ValueError("preset must be fine, normal, or coarse")
        direction = body.get("direction")
        reference = body.get("reference", self.control_reference)
        if reference not in {"world", "scene", "wrist"}:
            raise ValueError("reference must be world, scene, or wrist")
        cfg = PRESETS[preset]
        camera_scene = {k: np.asarray(v) for k, v in self.camera_axes["scene"].items()}
        camera_wrist = {k: np.asarray(v) for k, v in self.camera_axes["wrist"].items()}
        requested = world_delta_for_direction(direction, cfg["distance_m"], reference=reference,
                                               scene_basis=camera_scene, wrist_basis=camera_wrist,
                                               wrist_display_rotation_deg=self.wrist_display_rotation_deg)
        start_pos = np.asarray(self.observation["robot_state"]["eef"]["pos"], dtype=float).copy()
        start_mono = time.monotonic()
        command = {"kind": "move", "direction": direction, "preset": preset,
                   "control_reference": reference, "requested_controller_displacement_m": requested.tolist(),
                   "operator_view": self._operator_view(control_reference=reference),
                   "requested_distance_m": cfg["distance_m"], "input_method": input_method,
                   "hold_id": hold_id, "sequence": seq}
        max_steps = cfg["max_steps"]
        status = "step_limit"
        steps = 0
        for i in range(max_steps):
            if not self._lease_valid(hold_id):
                status = "stopped_or_lease_expired"
                break
            if time.monotonic() - start_mono >= MAX_COMMAND_SECONDS:
                status = "command_time_limit"
                break
            current_pos = np.asarray(self.observation["robot_state"]["eef"]["pos"], dtype=float)
            residual = requested - (current_pos - start_pos)
            if np.linalg.norm(residual) <= cfg["tolerance_m"]:
                status = "target_reached"
                break
            action = translation_action(residual)
            before = self._state()
            after, _ = self._execute_env_step(action, command=command, step_index=i+1,
                planned_steps=max_steps, input_method=input_method, before=before, started_mono=start_mono)
            steps += 1
            if self.terminated or self.truncated:
                status = "episode_finished"
                break
        measured = np.asarray(self.observation["robot_state"]["eef"]["pos"], dtype=float) - start_pos
        requested_norm = float(np.linalg.norm(requested))
        progress = float(np.dot(measured, requested) / max(requested_norm, 1e-12))
        alignment = float(np.dot(measured, requested) / max(np.linalg.norm(measured)*requested_norm, 1e-12))
        motion = {"requested_controller_displacement_m": requested.tolist(),
                 "measured_eef_displacement_m": measured.tolist(),
                 "requested_distance_m": requested_norm, "measured_distance_m": float(np.linalg.norm(measured)),
                 "progress_along_requested_m": progress, "direction_alignment_cosine": alignment,
                 "controller_steps": steps, "elapsed_seconds": time.monotonic()-start_mono,
                 "preset": preset, "reference": reference, "status": status}
        command["motion_summary"] = motion
        return {"motion": motion, "last_action": {"type": "move", "direction": direction,
                "preset": preset, "reference": reference, "status": status}}

    def _rotate(self, body, hold_id, input_method, seq):
        from thesis_vla.teleoperation import rotation_action
        preset = body.get("preset")
        if preset not in PRESETS:
            raise ValueError("preset must be fine, normal, or coarse")
        if not self._lease_valid(hold_id):
            raise ValueError("input lease expired or was released")
        axis, sign = body.get("axis"), body.get("sign")
        angle = float(PRESETS[preset]["rotation_rad"])
        action = rotation_action(axis, sign, angle)
        before = self._state()
        started = time.monotonic()
        command = {"kind": "rotate", "axis": axis, "sign": sign, "requested_angle_rad": sign*angle,
                   "preset": preset, "input_method": input_method, "hold_id": hold_id, "sequence": seq}
        self._execute_env_step(action, command=command, step_index=1, planned_steps=1,
                               input_method=input_method, before=before, started_mono=started)
        return {"last_action": {"type": "rotate", **command}}

    def _gripper(self, body, hold_id, input_method, seq):
        from thesis_vla.teleoperation import translation_action
        mode = body.get("gripper")
        if mode not in {"open", "close", "hold"}:
            raise ValueError("gripper must be open, close, or hold")
        start_mono = time.monotonic()
        command = {"kind": "gripper", "gripper": mode, "input_method": input_method,
                   "hold_id": hold_id, "sequence": seq}
        max_steps = 1 if mode == "hold" else GRIPPER_MAX_STEPS
        status = "holding" if mode == "hold" else "step_limit"
        steps = 0
        before_aperture = self._aperture_proxy(self._state())
        last_aperture = before_aperture
        stagnant = 0
        g = {"open": -1.0, "close": 1.0, "hold": 0.0}[mode]
        for i in range(max_steps):
            if not self._lease_valid(hold_id):
                status = "stopped_or_lease_expired"
                break
            before = self._state()
            action = translation_action(np.zeros(3), gripper=g)
            self._execute_env_step(action, command=command, step_index=i+1,
                planned_steps=max_steps, input_method=input_method, before=before, started_mono=start_mono)
            steps += 1
            aperture = self._aperture_proxy(self._state())
            change = aperture - last_aperture
            if abs(change) < 1e-5:
                stagnant += 1
            else:
                stagnant = 0
            last_aperture = aperture
            if mode == "close" and aperture < 0.008:
                status = "closed_limit"
                break
            if mode == "open" and aperture > 0.075:
                status = "open_limit"
                break
            if stagnant >= 4:
                status = "no_further_aperture_change"
                break
        after_aperture = self._aperture_proxy(self._state())
        result = {"gripper_status": f"{mode}:{status}", "gripper_steps": steps,
                  "aperture_before_proxy": float(before_aperture),
                  "aperture_after_proxy": after_aperture, "status": status}
        command["result"] = result
        return {"last_action": {"type": "gripper", "gripper": mode,
                "gripper_status": result["gripper_status"], "steps": steps}}

    @staticmethod
    def _aperture_proxy(state):
        return float(sum(abs(float(v)) for v in state["gripper"]["qpos"]))

    def transition(self, target):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("simulator busy; try again after the action completes")
        try:
            self.stop_hold()
            if target == "PAUSED" and self.mode == "HUMAN":
                self.mode = "PAUSED"
            elif target == "HUMAN" and self.mode == "PAUSED":
                self.mode = "HUMAN"
            elif target == "ESTOP" and self.mode in {"HUMAN", "PAUSED"}:
                self.mode = "ESTOP"
            else:
                raise ValueError(f"invalid transition {self.mode} -> {target}")
            self.epoch += 1
            self._log({"record_type": "control_transition", "run_id": self.run_id,
                       "time_unix": time.time(), "step": self.step,
                       "owner": self.mode, "epoch": self.epoch})
            return self.snapshot()
        finally:
            self.lock.release()

    def close(self):
        if self.video is not None:
            self.video.release()
        self.env.close()


def create_env():
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path.home() / ".cache/shared-autonomy-robot-learning/libero-config"))
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.libero_smoke import prepare_config_and_assets
    prepare_config_and_assets()
    from thesis_vla.libero import LiberoTaskConfig, LiberoVectorAdapter, create_vector_env
    config = LiberoTaskConfig(suite="libero_spatial", task_id=TASK_ID, width=256, height=256,
                              control_mode="relative", camera_name="agentview_image,robot0_eye_in_hand_image")
    return LiberoVectorAdapter(create_vector_env(config, gl="osmesa"))


class SimulationWorker:
    """Own all simulator calls on one thread; reject overlapping jobs, never queue stale actions."""
    def __init__(self, output_dir: Path):
        self.jobs = queue.Queue(maxsize=1)
        self.gate = threading.Lock()
        self.ready = threading.Event()
        self.session = None
        self.start_error = None
        self.thread = threading.Thread(target=self._run, args=(output_dir,),
                                       name="libero-simulator-owner", daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=120):
            raise RuntimeError("timed out creating LIBERO on its owner thread")
        if self.start_error:
            raise RuntimeError("LIBERO simulator initialization failed") from self.start_error

    def _run(self, output_dir):
        env = None
        try:
            env = create_env()
            self.session = TeleopSession(env, output_dir)
            self.ready.set()
            while True:
                job = self.jobs.get()
                if job is None:
                    break
                operation, result, finished = job
                try:
                    result["value"] = operation()
                except BaseException as exc:
                    result["error"] = exc
                finally:
                    finished.set()
        except BaseException as exc:
            self.start_error = exc
            self.ready.set()
        finally:
            if self.session is not None:
                self.session.close()
            elif env is not None:
                env.close()

    def invoke(self, operation):
        if not self.gate.acquire(blocking=False):
            raise RuntimeError("simulator busy; no action was queued")
        try:
            result, finished = {}, threading.Event()
            self.jobs.put_nowait((operation, result, finished))
            finished.wait()
            if "error" in result:
                raise result["error"]
            return result.get("value")
        finally:
            self.gate.release()

    def close(self):
        if self.thread.is_alive():
            with self.gate:
                self.jobs.put_nowait(None)
                self.thread.join(timeout=30)


def _calibration_self_test():
    from thesis_vla.teleoperation import world_delta_for_direction
    with tempfile.TemporaryDirectory(prefix="libero-teleop-calibration-") as tmp:
        worker = SimulationWorker(Path(tmp))
        session = worker.session
        seq = 0
        def run_command(body):
            nonlocal seq
            seq += 1
            hold_id = str(uuid.uuid4())
            body.update({"hold_id": hold_id, "epoch": session.epoch, "seq": seq})
            stop = threading.Event()
            session.begin_hold(hold_id)
            def heartbeat():
                while not stop.wait(.15):
                    session.heartbeat(hold_id)
            hb = threading.Thread(target=heartbeat, daemon=True)
            hb.start()
            try:
                return worker.invoke(lambda: session.execute(body))
            finally:
                stop.set()
                session.end_hold(hold_id)
                hb.join(timeout=1)
        results = []
        try:
            for ref, direction in [("world", "forward"), ("world", "right"), ("world", "up"),
                                   ("scene", "forward"), ("scene", "backward"),
                                   ("scene", "right"), ("scene", "left"),
                                   ("wrist", "forward"), ("wrist", "backward"),
                                   ("wrist", "right"), ("wrist", "left")]:
                body = {"kind": "move", "direction": direction, "preset": "fine", "reference": ref,
                        "input_method": "automated_test"}
                response = run_command(body)
                motion = response["last_motion"]
                cosine = motion["direction_alignment_cosine"]
                if motion["measured_distance_m"] < .00005 or cosine < .30:
                    raise RuntimeError(f"{ref}/{direction} did not follow the camera/world vector: {motion}")
                results.append({"reference": ref, "direction": direction,
                                "requested_m": motion["requested_controller_displacement_m"],
                                "measured_m": motion["measured_eef_displacement_m"],
                                "cosine": cosine, "steps": motion["controller_steps"]})
            # Validate both wrist display orientations against the same live
            # camera extrinsics. This is automated controller testing only.
            for display_rotation in (0, 180):
                worker.invoke(lambda rotation=display_rotation: session.set_wrist_display_rotation(rotation))
                body = {"kind": "move", "direction": "forward", "preset": "fine", "reference": "wrist",
                        "input_method": "automated_test"}
                response = run_command(body)
                motion = response["last_motion"]
                if motion["direction_alignment_cosine"] < .30:
                    raise RuntimeError(f"wrist display {display_rotation} direction mismatch: {motion}")
                results.append({"reference": "wrist", "display_rotation_deg": display_rotation,
                                "direction": "forward", "requested_m": motion["requested_controller_displacement_m"],
                                "measured_m": motion["measured_eef_displacement_m"],
                                "cosine": motion["direction_alignment_cosine"], "steps": motion["controller_steps"]})
            previous_wrist_right = np.asarray(session.camera_axes["wrist"]["right_world"]).copy()
            rotate_body = {"kind": "rotate", "axis": 2, "sign": 1, "preset": "fine",
                           "reference": "wrist", "input_method": "automated_test"}
            run_command(rotate_body)
            current_wrist_right = np.asarray(session.camera_axes["wrist"]["right_world"]).copy()
            if np.linalg.norm(current_wrist_right - previous_wrist_right) < 1e-4:
                raise RuntimeError("wrist camera extrinsics did not update after end-effector rotation")
            worker.invoke(lambda: session.set_wrist_display_rotation(180))
            rotated_body = {"kind": "move", "direction": "right", "preset": "fine", "reference": "wrist",
                            "input_method": "automated_test"}
            rotated_response = run_command(rotated_body)
            rotated_motion = rotated_response["last_motion"]
            if rotated_motion["direction_alignment_cosine"] < .30:
                raise RuntimeError(f"post-rotation wrist mapping did not align: {rotated_motion}")
            results.append({"reference": "wrist", "display_rotation_deg": 180,
                            "direction": "right_after_camera_rotation",
                            "camera_right_world_before": previous_wrist_right.tolist(),
                            "camera_right_world_after": current_wrist_right.tolist(),
                            "measured_m": rotated_motion["measured_eef_displacement_m"],
                            "cosine": rotated_motion["direction_alignment_cosine"],
                            "steps": rotated_motion["controller_steps"]})
            # Preset speed check: coarse target must yield more measured travel than fine.
            base = np.asarray(session.observation["robot_state"]["eef"]["pos"], dtype=float).copy()
            b = {"kind": "move", "direction": "forward", "preset": "fine", "reference": "world",
                 "input_method": "automated_test"}
            run_command(b)
            fine_distance = float(np.linalg.norm(np.asarray(session.observation["robot_state"]["eef"]["pos"]) - base))
            base = np.asarray(session.observation["robot_state"]["eef"]["pos"], dtype=float).copy()
            b = {"kind": "move", "direction": "forward", "preset": "coarse", "reference": "world",
                 "input_method": "automated_test"}
            run_command(b)
            coarse_distance = float(np.linalg.norm(np.asarray(session.observation["robot_state"]["eef"]["pos"]) - base))
            base = np.asarray(session.observation["robot_state"]["eef"]["pos"], dtype=float).copy()
            b = {"kind": "move", "direction": "forward", "preset": "normal", "reference": "world",
                 "input_method": "automated_test"}
            run_command(b)
            normal_distance = float(np.linalg.norm(np.asarray(session.observation["robot_state"]["eef"]["pos"]) - base))
            if not (coarse_distance > normal_distance > fine_distance):
                raise RuntimeError(f"preset travel is not ordered fine < normal < coarse: fine={fine_distance}, normal={normal_distance}, coarse={coarse_distance}")
            q0 = session._aperture_proxy(session._state())
            b = {"kind": "gripper", "gripper": "close", "input_method": "automated_test",
                 }
            run_command(b)
            q1 = session._aperture_proxy(session._state())
            b = {"kind": "gripper", "gripper": "open", "input_method": "automated_test",
                 }
            run_command(b)
            q2 = session._aperture_proxy(session._state())
            if not (q1 < q0 and q2 > q1):
                raise RuntimeError(f"gripper did not close then open: {q0:.5f}->{q1:.5f}->{q2:.5f}")
            with session.log_path.open(encoding="utf-8") as stream:
                logged = [json.loads(line) for line in stream]
            action_rows = [row for row in logged if row.get("record_type") == "human_action_step"]
            if not action_rows or any(
                    len(row.get("executed_action_7d", [])) != 7
                    or "operator_view" not in row
                    or any(len(row.get("camera_files", {}).get(cam, {}).get("sha256", "")) != 64
                           for cam in ("scene", "wrist"))
                    for row in action_rows):
                raise RuntimeError("simulator logs lack 7D action, operator view, or raw-camera hashes")
            return {"SELF_TEST": "PASS", "task_id": TASK_ID, "seed": SEED,
                    "state_sha256": STATE_SHA256, "camera_direction_checks": results,
                    "fine_measured_m": fine_distance, "normal_measured_m": normal_distance,
                    "coarse_measured_m": coarse_distance,
                    "gripper_aperture_proxy_close_open": [q0, q1, q2],
                    "action_log_records_checked": len(action_rows),
                    "owner_thread": worker.thread.name, "human_input": False, "vla_loaded": False}
        finally:
            worker.close()


def _selftest_entry():
    result = _calibration_self_test()
    print(json.dumps(result, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", default="outputs/human_teleop_task1_state0_seed1101_retry2")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if args.self_test:
        _selftest_entry()
        return
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    pid_path = out / "server.pid"
    pid_path.write_text(str(os.getpid()), encoding="ascii")
    worker = SimulationWorker(out)
    session = worker.session
    token = secrets.token_urlsafe(24)
    page = HTML.read_text(encoding="utf-8").replace("__TOKEN__", token)

    class Handler(BaseHTTPRequestHandler):
        server_version = "LIBERO-Teleop/0.2"
        def _send(self, code, data, kind="application/json"):
            payload = data.encode() if isinstance(data, str) else json.dumps(data, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try: self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError): pass
        def _auth(self):
            if self.headers.get("X-Teleop-Token") != token:
                self._send(403, {"error": "local session token required"}); return False
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{args.port}":
                self._send(403, {"error": "cross-origin request rejected"}); return False
            return True
        def _body(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096: raise ValueError("invalid request size")
            return json.loads(self.rfile.read(length))
        def do_GET(self):
            if self.path == "/":
                self._send(200, page, "text/html; charset=utf-8"); return
            if self.path != "/api/state":
                self._send(404, {"error": "not found"}); return
            if not self._auth(): return
            if not session.lock.acquire(blocking=False):
                self._send(409, {"error": "busy"}); return
            try: self._send(200, session.snapshot())
            finally: session.lock.release()
        def do_POST(self):
            if not self._auth(): return
            try:
                if self.path in {"/api/hold/start", "/api/heartbeat", "/api/hold/stop", "/api/hold/end"}:
                    body = self._body()
                    hold_id = body.get("hold_id")
                    if self.path == "/api/hold/start": session.begin_hold(hold_id); result = {"ok": True}
                    elif self.path == "/api/heartbeat": result = {"ok": session.heartbeat(hold_id)}
                    elif self.path == "/api/hold/stop": result = {"ok": session.stop_hold(hold_id)}
                    else: session.end_hold(hold_id); result = {"ok": True}
                    self._send(200, result); return
                if self.path == "/api/reference":
                    result = session.set_reference(self._body().get("reference"))
                    self._send(200, result); return
                if self.path == "/api/display":
                    result = session.set_wrist_display_rotation(self._body().get("wrist_rotation_deg"))
                    self._send(200, result); return
                if self.path == "/api/action":
                    body = self._body()
                    self._send(200, worker.invoke(lambda: session.execute(body))); return
                if self.path in {"/api/pause", "/api/resume", "/api/estop"}:
                    target = {"/api/pause": "PAUSED", "/api/resume": "HUMAN", "/api/estop": "ESTOP"}[self.path]
                    self._send(200, worker.invoke(lambda: session.transition(target))); return
                self._send(404, {"error": "not found"})
            except RuntimeError as exc: self._send(409, {"error": str(exc)})
            except Exception as exc: self._send(400, {"error": str(exc)})
        def log_message(self, fmt, *values):
            print("teleop-http", self.address_string(), fmt % values, flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"TELEOP_READY address=127.0.0.1:{args.port} task={TASK_ID} seed={SEED} state={STATE_INDEX} state_sha256={STATE_SHA256}", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try: server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        worker.close()
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
