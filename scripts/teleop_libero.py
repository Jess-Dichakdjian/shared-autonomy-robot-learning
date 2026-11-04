"""Private, step-by-step LIBERO teleoperation server; no learned policy loaded."""
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


def _float_tree(value):
    if isinstance(value, dict):
        return {str(k): _float_tree(v) for k, v in value.items()}
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def _jpeg(image):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TeleopSession:
    def __init__(self, env, output_dir: Path):
        self.env = env
        self.output_dir = output_dir
        self.lock = threading.Lock()
        self.mode = "HUMAN"
        self.epoch = 0
        self.last_seq = -1
        self.step = 0
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.last_action = None
        self.started = time.time()
        self.log_path = output_dir / "teleoperation.jsonl"
        self.video_path = output_dir / "teleoperation.mp4"
        self.video = None
        self.observation = None
        self.reset_info = None
        self._reset()

    def _reset(self):
        self.observation, self.reset_info = self.env.reset(
            seed=SEED, options={"saved_init_state_index": STATE_INDEX})
        if self.reset_info.get("saved_init_state_sha256") != STATE_SHA256:
            raise RuntimeError("Pinned initial state hash does not match the approved fixture")

    @property
    def raw_env(self):
        return self.env.env.envs[0].unwrapped

    def snapshot(self):
        info = getattr(self, "last_info", {}) or {}
        pixels = self.observation["pixels"]
        return {
            "mode": self.mode, "owner": "HUMAN" if self.mode == "HUMAN" else self.mode,
            "vla_loaded": False, "epoch": self.epoch, "step": self.step,
            "next_seq": self.last_seq + 1,
            "seed": SEED, "task_id": TASK_ID, "task": TASK_TEXT,
            "saved_init_state_index": STATE_INDEX,
            "saved_init_state_sha256": STATE_SHA256,
            "robot_state": _float_tree(self.observation["robot_state"]),
            "reward": self.reward, "is_success": bool(info.get("is_success", False)),
            "terminated": self.terminated, "truncated": self.truncated,
            "last_action": self.last_action,
            "images": {"front": _jpeg(pixels["image"]), "wrist": _jpeg(pixels["image2"])},
        }

    def _log(self, record):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")

    def execute(self, body, map_key):
        # Nonblocking lock deliberately rejects concurrent commands instead of queuing them.
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("busy; retry only after refreshing the current state")
        try:
            if self.mode != "HUMAN":
                raise RuntimeError(f"controls disabled while mode is {self.mode}")
            if body.get("epoch") != self.epoch:
                raise ValueError("stale control epoch; refresh the page state")
            seq = body.get("seq")
            if isinstance(seq, bool) or not isinstance(seq, int) or seq <= self.last_seq:
                raise ValueError("duplicate or stale command sequence")
            key = body.get("key", "")
            grip = body.get("gripper")
            if (bool(key) == bool(grip)):
                raise ValueError("send exactly one movement key or gripper command")
            action = np.asarray(map_key(key, position_step_m=float(body.get("position_step_m", .005)),
                                       rotation_step_rad=float(body.get("rotation_step_rad", .0523598776)),
                                       gripper=grip), dtype=np.float32)
            before = _float_tree(self.observation["robot_state"])
            obs, reward, term, trunc, info = self.env.step(action)
            self.observation = obs
            self.reward, self.terminated, self.truncated = reward, term, trunc
            self.last_info = info
            self.step += 1
            self.last_seq = seq
            self.last_action = {"source": "HUMAN", "key": key or None, "gripper": grip,
                                "action_7d": action.astype(float).tolist()}
            if term or trunc:
                self.mode = "FINISHED"
                self.epoch += 1
            after = _float_tree(obs["robot_state"])
            self._log({"time_unix": time.time(), "step": self.step, "owner": "HUMAN",
                       "sequence": seq, "epoch": self.epoch, "command": self.last_action,
                       "predicted_action": None, "executed_action": action.astype(float).tolist(),
                       "robot_state_before": before, "robot_state_after": after,
                       "reward": reward, "is_success": bool(info.get("is_success", False)),
                       "terminated": term, "truncated": trunc})
            if self.video is None:
                import cv2
                self.video = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             10, (512, 256))
                if not self.video.isOpened():
                    self.video.release()
                    self.video = None
                    raise RuntimeError("OpenCV could not open the tested MP4 video writer")
            frame = np.concatenate([obs["pixels"]["image"], obs["pixels"]["image2"]], axis=1)
            self.video.write(frame[:, :, ::-1].copy())
            return self.snapshot()
        finally:
            self.lock.release()

    def transition(self, target):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("busy; retry after refresh")
        try:
            if target == "PAUSED" and self.mode == "HUMAN":
                self.mode = "PAUSED"
            elif target == "HUMAN" and self.mode == "PAUSED":
                self.mode = "HUMAN"
            elif target == "ESTOP" and self.mode in {"HUMAN", "PAUSED"}:
                self.mode = "ESTOP"
            else:
                raise ValueError(f"invalid transition {self.mode} -> {target}")
            self.epoch += 1
            self._log({"time_unix": time.time(), "step": self.step, "owner": self.mode,
                       "event": "control_transition", "epoch": self.epoch})
            return self.snapshot()
        finally:
            self.lock.release()

    def close(self):
        if self.video is not None:
            self.video.release()
        self.env.close()


class SimulationWorker:
    """Keep the MuJoCo/OSMesa context on one owner thread for its full lifetime."""
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
            raise RuntimeError("timed out creating the LIBERO environment on its owner thread")
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
        # One in-flight request maximum: a dropped connection cannot leave a
        # stale action waiting behind a currently executing action.
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
        if not self.thread.is_alive():
            return
        with self.gate:
            self.jobs.put_nowait(None)
            self.thread.join(timeout=30)


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


def self_test(output_root: Path):
    from thesis_vla.teleoperation import map_key
    with tempfile.TemporaryDirectory(prefix="libero-teleop-selftest-") as tmp:
        worker = SimulationWorker(Path(tmp))
        try:
            def check_on_owner_thread():
                session = worker.session
                obs0, info0 = session.observation, session.reset_info
                if info0.get("saved_init_state_sha256") != STATE_SHA256:
                    raise RuntimeError("initial state hash mismatch")
                x0 = float(obs0["robot_state"]["eef"]["pos"][0])
                action = np.asarray(map_key("w", position_step_m=.005), dtype=np.float32)
                current = obs0
                for _ in range(3):
                    current, _, _, _, _ = session.env.step(action)
                x1 = float(current["robot_state"]["eef"]["pos"][0])
                if not np.isfinite(x1) or x1 <= x0:
                    raise RuntimeError(f"W/world-X command did not produce positive world-X motion: {x0:.6f} -> {x1:.6f}")
                metrics = {}
                for key in ("image", "image2"):
                    image = np.asarray(current["pixels"][key])
                    if image.shape != (256, 256, 3):
                        raise RuntimeError(f"unexpected camera shape for {key}: {image.shape}")
                    metrics[key] = {"std": float(image.std()), "range": int(image.max()-image.min())}
                    if image.std() < 25 or image.max()-image.min() < 100:
                        raise RuntimeError(f"{key} image appears blank/corrupt: {metrics[key]}")
                restored, reset_info = session.env.reset(seed=SEED, options={"saved_init_state_index": STATE_INDEX})
                if reset_info.get("saved_init_state_sha256") != STATE_SHA256:
                    raise RuntimeError("reset after motion did not restore pinned initial state")
                session.observation, session.reset_info = restored, reset_info
                return {"TELEOP_SELF_TEST": "PASS", "task_id": TASK_ID, "seed": SEED,
                        "state_index": STATE_INDEX, "state_sha256": STATE_SHA256,
                        "world_x_before_m": x0, "world_x_after_3x_W_m": x1,
                        "delta_x_m": x1-x0, "camera_metrics": metrics,
                        "owner_thread": threading.current_thread().name,
                        "human_input": False, "vla_loaded": False}
            print(json.dumps(worker.invoke(check_on_owner_thread), allow_nan=False))
        finally:
            worker.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", default="outputs/human_teleop_task1_state0_seed1101_retry1")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if args.self_test:
        self_test(Path(args.output))
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
        server_version = "LIBERO-Teleop/0.1"
        def _send(self, code, data, kind="application/json"):
            payload = data.encode() if isinstance(data, str) else json.dumps(data, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def _auth(self):
            if self.headers.get("X-Teleop-Token") != token:
                self._send(403, {"error": "local session token required"}); return False
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{args.port}":
                self._send(403, {"error": "cross-origin request rejected"}); return False
            return True
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
            if self.path == "/api/action":
                if not self._auth(): return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 2048: raise ValueError("invalid request size")
                    body = json.loads(self.rfile.read(length))
                    self._send(200, worker.invoke(lambda: session.execute(body, __import__("thesis_vla.teleoperation", fromlist=["map_key"]).map_key)))
                except RuntimeError as e: self._send(409, {"error": str(e)})
                except Exception as e: self._send(400, {"error": str(e)})
                return
            if self.path in {"/api/pause", "/api/resume", "/api/estop"}:
                if not self._auth(): return
                target = {"/api/pause": "PAUSED", "/api/resume": "HUMAN", "/api/estop": "ESTOP"}[self.path]
                try: self._send(200, worker.invoke(lambda: session.transition(target)))
                except RuntimeError as e: self._send(409, {"error": str(e)})
                except Exception as e: self._send(400, {"error": str(e)})
                return
            self._send(404, {"error": "not found"})
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
