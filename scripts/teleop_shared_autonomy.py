"""Integrated LIBERO pi0.5/GP/human handoff server with one simulator owner."""
from __future__ import annotations
import argparse, hashlib, json, os, secrets, signal, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import teleop_libero as cockpit
from thesis_vla.shared_autonomy import SharedAutonomyProtocol

REV = "7152f13e34c6c115d808aea0e5d08aca74eac14b"
WEIGHT_SHA = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"
CHECKPOINT = Path.home() / ".cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots" / REV
GP_MODEL = Path.home() / "projects/shared-autonomy-robot-learning-gp-uncertainty/outputs/gp_uncertainty_validation_2026-11-04/models.joblib"
GP_PYTHON = Path.home() / ".venvs/shared-autonomy-robot-learning-gp-beast/bin/python"
GP_CONFIG = ROOT / "configs/gp_uncertainty_selected_model.json"


class SharedSession(cockpit.TeleopSession):
    def __init__(self, env, output_dir, policy, gp, checkpoint, gp_model, thresholds, monitor_mode, preflight):
        self.policy, self.gp = policy, gp
        self.protocol = SharedAutonomyProtocol(policy)
        self.mode = "HUMAN"
        self.autonomy_started = False
        self.manual_request_id = None
        self.current_intervention_id = None
        self.assistance_request = None
        self.assistance_source = None
        self.last_error = None
        self.fresh_resume_pending = False
        self.autonomy_thread = None
        self.locked_worker = None
        super().__init__(env, output_dir)
        self.integration = {
            "policy": "frozen LeRobot pi0.5", "checkpoint_id": "lerobot/pi05_libero_finetuned_v044",
            "checkpoint_revision": REV, "checkpoint_sha256": WEIGHT_SHA,
            "checkpoint_path": str(checkpoint), "gp_model_path": str(gp_model),
            "gp_model_sha256": getattr(gp, "model_sha256", None),
            "monitoring_mode": monitor_mode, "thresholds_validation_frozen": thresholds,
            "preflight": preflight, "camera_contract": ["agentview_image", "robot0_eye_in_hand_image"],
            "state_contract": "checkpoint-normalized 8D pi0.5 state",
            "action_contract": "7D normalized relative OSC; predicts 50, executes 10",
            "renderer": "OSMesa CPU rasterization", "handoff": "chunk-boundary only; queued actions and policy temporal state reset",
            "task_instruction": cockpit.TASK_TEXT,
        }
        config_path = output_dir / "configuration.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg.update(self.integration)
        config_path.write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        self.correction_path = output_dir / "human_corrections.jsonl"

    def attach_worker(self, worker):
        self.locked_worker = worker

    def snapshot(self):
        data = super().snapshot()
        if self.mode == "ASSISTANCE_REQUESTED":
            label = f"{self.assistance_source}; accept to take control"
        elif self.mode == "AUTONOMY":
            label = "frozen pi0.5 active; GP checked at chunk boundaries"
        elif self.autonomy_started and self.mode == "HUMAN":
            label = "human control; pi0.5 paused"
        else:
            label = "ready; start frozen pi0.5"
        data.update(vla_loaded=True, autonomy_started=self.autonomy_started,
                    assistance_request=label, assistance_source=self.assistance_source,
                    assistance_request_record=self.assistance_request, last_error=self.last_error)
        return data

    def execute(self, body):
        if not self.autonomy_started:
            raise RuntimeError("start pi0.5 before sending manual robot actions")
        return super().execute(body)

    def start_autonomy(self):
        if self.terminated or self.truncated:
            raise RuntimeError("episode ended; restart with a new output directory")
        self.protocol.start_autonomy()
        self.mode = self.protocol.state
        self.autonomy_started = True
        self.assistance_request = self.assistance_source = None
        self.current_intervention_id = None
        self.epoch += 1
        self._log({"record_type": "control_transition", "event": "start_pi05", "owner": "AUTONOMY",
                   "run_id": self.run_id, "step": self.step, "policy_reset": True, "time_unix": time.time()})
        return self.snapshot()

    def request_human(self, origin="human_operator"):
        candidate_id = f"{self.run_id}:manual:{uuid.uuid4()}"
        result = self.protocol.request_human(candidate_id, origin=origin)
        if result.get("already_pending"):
            return result
        self.manual_request_id = result["request_id"]
        self._log({"record_type": "manual_assistance_request", "owner": "SYSTEM",
                   "source": "manual_operator_request", "request_origin": origin,
                   "request_id": self.manual_request_id, "run_id": self.run_id, "step": self.step,
                   "takes_effect": "next fresh pi0.5 chunk boundary", "time_unix": time.time()})
        return result

    def accept_human(self):
        self.protocol.accept_human()
        self.mode = self.protocol.state
        self.epoch += 1
        self._log({"record_type": "control_transition", "event": "operator_accepted_handoff",
                   "owner": "HUMAN", "run_id": self.run_id, "step": self.step,
                   "intervention_id": self._intervention_id(), "robot_state": self._state(), "time_unix": time.time()})
        return self.snapshot()

    def _intervention_id(self):
        return self.current_intervention_id or self.manual_request_id or (self.assistance_request or {}).get("request_id")

    def resume_autonomy(self):
        if not self.autonomy_started:
            raise RuntimeError("start pi0.5 before requesting autonomy resumption")
        self.protocol.resume_autonomy()
        self.mode = self.protocol.state
        self.epoch += 1
        self.fresh_resume_pending = True
        self._log({"record_type": "control_transition", "event": "human_returned_control_to_pi05",
                   "owner": "AUTONOMY", "run_id": self.run_id, "step": self.step,
                   "policy_reset": True, "fresh_observation_state": self._state(),
                   "intervention_id": self._intervention_id(), "time_unix": time.time()})
        self.assistance_request = self.assistance_source = None
        self.current_intervention_id = None
        return self.snapshot()

    def emergency_stop(self):
        self.protocol.emergency_stop()
        self.mode = self.protocol.state
        self.stop_hold()
        self.epoch += 1
        self._log({"record_type": "control_transition", "event": "emergency_stop", "owner": "ESTOP",
                   "run_id": self.run_id, "step": self.step, "policy_reset": True, "time_unix": time.time()})
        return self.snapshot()

    def autonomous_step(self):
        if self.mode != "AUTONOMY":
            return {"state": self.mode}
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("simulator busy; autonomous step will retry")
        try:
            import torch
            before = self._state()
            started = time.perf_counter()
            with torch.inference_mode():
                proposal = self.policy.act(self.observation)
            latency = time.perf_counter() - started
            gpu_allocated = int(torch.cuda.memory_allocated(0)) if torch.cuda.is_available() else 0
            gpu_peak = int(torch.cuda.max_memory_allocated(0)) if torch.cuda.is_available() else 0
            if proposal.is_chunk_start:
                event = self.protocol.at_chunk_boundary(proposal)
                if event:
                    self.mode = self.protocol.state
                    self.assistance_source = event.source
                    self.current_intervention_id = f"{self.run_id}:handoff:{uuid.uuid4()}"
                    self.assistance_request = event.request or {
                        "request_id": self.manual_request_id, "source": "manual_operator_request"}
                    self.epoch += 1
                    self.last_action = {"type": "pi05_action_preempted", "chunk_id": proposal.chunk_id,
                                        "chunk_index": proposal.chunk_index, "executed": False}
                    self._log({"record_type": "assistance_handoff_request", "owner": "SYSTEM",
                        "human_input": False, "run_id": self.run_id, "step": self.step,
                        "request_source": event.source, "request": self.assistance_request,
                        "monitoring": proposal.assistance_request, "passive_gp_prediction": proposal.passive_gp_prediction,
                        "chunk_id": proposal.chunk_id, "chunk_index": proposal.chunk_index,
                        "candidate_predicted_action_7d": np.asarray(proposal.predicted_action).tolist(),
                        "candidate_postprocessed_action_7d": np.asarray(proposal.executed_action).tolist(),
                        "candidate_executed": False, "queued_pi05_actions_cleared": True,
                        "policy_temporal_state_reset": True, "robot_state": before,
                        "inference_seconds": latency, "time_unix": time.time()})
                    return {"state": self.mode, "handoff": True}
            obs, reward, term, trunc, info = self.env.step(proposal.executed_action)
            self.observation = obs
            self.reward, self.terminated, self.truncated = float(reward), bool(term), bool(trunc)
            self.last_info = info or {}
            self.step += 1
            self._update_camera_axes()
            after = self._state()
            captures = self._save_raw_cameras(obs, self.step)
            self._write_video_frame(obs)
            fresh = self.fresh_resume_pending and proposal.is_chunk_start
            self._log({"record_type": "vla_action_step", "owner": "AUTONOMY", "human_input": False,
                "action_source": "pi05", "run_id": self.run_id, "step": self.step,
                "predicted_raw_normalized_action_7d": np.asarray(proposal.predicted_action).tolist(),
                "executed_relative_osc_action_7d": np.asarray(proposal.executed_action).tolist(),
                "action_executed": True, "chunk_id": proposal.chunk_id, "chunk_index": proposal.chunk_index,
                "is_chunk_start": proposal.is_chunk_start, "inference_seconds": latency,
                "fresh_post_correction_inference": fresh, "policy_reset_before_inference": fresh,
                "passive_gp_prediction": proposal.passive_gp_prediction,
                "assistance_request": proposal.assistance_request, "robot_state_before": before,
                "robot_state_after": after, "camera_files": captures,
                "camera_extrinsics": cockpit._jsonable_axes(self.camera_axes),
                "reward": self.reward, "is_success": bool(self.last_info.get("is_success", False)),
                "terminated": self.terminated, "truncated": self.truncated,
                "gpu_allocated_bytes": gpu_allocated, "gpu_peak_allocated_bytes": gpu_peak,
                "time_unix": time.time()})
            if fresh:
                self.fresh_resume_pending = False
            self.last_action = {"type": "pi05", "chunk_id": proposal.chunk_id, "chunk_index": proposal.chunk_index}
            if self.terminated or self.truncated:
                self.protocol.finish()
                self.mode = "FINISHED"
                self._log({"record_type": "episode_outcome", "owner": "SYSTEM", "run_id": self.run_id,
                    "step": self.step, "success": bool(self.last_info.get("is_success", False)),
                    "terminated": self.terminated, "truncated": self.truncated,
                    "total_reward": self.reward, "time_unix": time.time()})
            return {"state": self.mode, "step": self.step, "inference_seconds": latency}
        finally:
            self.lock.release()

    def _execute_env_step(self, action, *, command, step_index, planned_steps, input_method, before, started_mono):
        old_obs, old_step = self.observation, self.step
        result = super()._execute_env_step(action, command=command, step_index=step_index,
            planned_steps=planned_steps, input_method=input_method, before=before, started_mono=started_mono)
        if input_method != "automated_test":
            cams = {}
            for name, key in (("scene", "image"), ("wrist", "image2")):
                frame = np.asarray(old_obs["pixels"][key], dtype=np.uint8)
                cams[name] = {"path": f"camera_raw/{name}/step_{old_step:06d}.png",
                              "sha256": hashlib.sha256(frame.tobytes()).hexdigest()}
            after_cameras = {}
            for name, key in (("scene", "image"), ("wrist", "image2")):
                frame = np.asarray(self.observation["pixels"][key], dtype=np.uint8)
                after_cameras[name] = {"path": f"camera_raw/{name}/step_{self.step:06d}.png",
                    "sha256": hashlib.sha256(frame.tobytes()).hexdigest()}
            record = {"record_type": "human_corrective_demo_step", "schema_version": 1,
                "run_id": self.run_id, "episode_id": self.run_id, "intervention_id": self._intervention_id(),
                "source": "human_keyboard" if input_method == "keyboard" else "human_button",
                "input_method": input_method, "step": self.step, "task_id": cockpit.TASK_ID,
                "seed": cockpit.SEED, "saved_init_state_index": cockpit.STATE_INDEX,
                "saved_init_state_sha256": cockpit.STATE_SHA256,
                "observation_before": {"robot_state": before, "camera_frames": cams},
                "executed_action_7d": np.asarray(action, dtype=float).tolist(),
                "action_representation": "normalized 7D relative OSC action",
                "observation_after": {"robot_state": self._state(), "camera_frames": after_cameras},
                "policy_checkpoint_sha256": WEIGHT_SHA, "controller_owner": "HUMAN"}
            with (self.output_dir / "human_corrections.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            if self.terminated or self.truncated:
                self._log({"record_type": "episode_outcome", "owner": "HUMAN", "run_id": self.run_id,
                    "step": self.step, "success": bool(self.last_info.get("is_success", False)),
                    "terminated": self.terminated, "truncated": self.truncated, "total_reward": self.reward,
                    "time_unix": time.time()})
        return result


class SharedWorker(cockpit.SimulationWorker):
    def __init__(self, out, args, policy, gp, thresholds, preflight):
        import queue
        self.jobs = queue.Queue(maxsize=1)
        self.gate = threading.Lock()
        self.ready = threading.Event()
        self.session = None
        self.start_error = None
        self.worker_args = (policy, gp, args.checkpoint, args.gp_model, thresholds, args.monitoring_mode, preflight)
        self.thread = threading.Thread(target=self._run, args=(out,), name="libero-shared-autonomy-owner", daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=120): raise RuntimeError("timed out creating LIBERO owner")
        if self.start_error: raise RuntimeError("LIBERO setup failed") from self.start_error

    def _run(self, out):
        env = None
        try:
            env = cockpit.create_env()
            self.session = SharedSession(env, out, *self.worker_args)
            self.ready.set()
            while True:
                job = self.jobs.get()
                if job is None: break
                operation, result, done = job
                try: result["value"] = operation()
                except BaseException as exc: result["error"] = exc
                finally: done.set()
        except BaseException as exc:
            self.start_error = exc
            self.ready.set()
        finally:
            if self.session: self.session.close()
            elif env: env.close()

    def start_loop(self):
        s = self.session
        if s.autonomy_thread and s.autonomy_thread.is_alive(): return
        def run():
            while s.mode == "AUTONOMY":
                try: self.invoke(s.autonomous_step)
                except RuntimeError as exc:
                    if "busy" in str(exc): time.sleep(.02); continue
                    s.last_error = f"{type(exc).__name__}: {exc}"
                    try: self.invoke(s.emergency_stop)
                    except Exception: pass
                    break
                except Exception as exc:
                    s.last_error = f"{type(exc).__name__}: {exc}"
                    try: self.invoke(s.emergency_stop)
                    except Exception: pass
                    break
                time.sleep(.005)
        s.autonomy_thread = threading.Thread(target=run, name="pi05-autonomy-loop", daemon=True)
        s.autonomy_thread.start()

    def close(self):
        if self.session and self.session.autonomy_thread:
            self.session.protocol.emergency_stop()
            self.session.mode = "ESTOP"
            self.session.autonomy_thread.join(timeout=5)
        super().close()
        if self.session and self.session.gp:
            self.session.gp.close()


def load_model(args, thresholds):
    import torch
    from libero.libero.benchmark import get_benchmark_dict
    from thesis_vla.libero import LiberoTaskConfig
    from thesis_vla.pi05 import Pi05PolicyAdapter
    from thesis_vla.gp.client import GPInferenceClient
    from thesis_vla.placement import verify_single_gpu_placement
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    free, total = torch.cuda.mem_get_info(torch.device("cuda:0"))
    if free < int(args.minimum_free_gb * 1024**3):
        raise RuntimeError(f"only {free/1024**3:.2f} GiB GPU memory is free")
    torch.cuda.reset_peak_memory_stats(0)
    probe = torch.ones((4, 4), device="cuda:0", dtype=torch.bfloat16)
    if float(probe.sum().item()) != 16: raise RuntimeError("CUDA tensor operation failed")
    del probe
    weight_path = args.checkpoint / "model.safetensors"
    if not weight_path.is_file(): raise FileNotFoundError(weight_path)
    gp = GPInferenceClient(args.gp_model, python_executable=args.gp_python)
    try:
        config = LiberoTaskConfig(suite="libero_spatial", task_id=cockpit.TASK_ID, width=256, height=256,
            control_mode="relative", camera_name="agentview_image,robot0_eye_in_hand_image")
        task = str(get_benchmark_dict()["libero_spatial"]().get_task(cockpit.TASK_ID).language)
        policy = Pi05PolicyAdapter.from_pretrained(args.checkpoint, task=task, env_config=config,
            execute_steps=10, local_files_only=True, device="cuda:0", dtype=torch.bfloat16,
            cpu_output_heads=True, meta_init=True, verify_placement_integrity=True,
            expected_weights_sha256=WEIGHT_SHA, passive_gp=gp, monitoring_mode=args.monitoring_mode,
            monitoring_thresholds=thresholds)
        placement = verify_single_gpu_placement(policy.policy, torch.device("cuda:0"))
        torch.cuda.reset_peak_memory_stats(0)
        info = {"gpu_free_bytes_before_load": int(free), "gpu_total_bytes": int(total),
                "cuda_tensor_probe": "passed", "placement": placement}
        return policy, gp, task, info
    except Exception:
        gp.close()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--gp-model", type=Path, default=GP_MODEL)
    parser.add_argument("--gp-python", type=Path, default=GP_PYTHON)
    parser.add_argument("--gp-config", type=Path, default=GP_CONFIG)
    parser.add_argument("--monitoring-mode", choices=("gp_uncertainty", "gp_vla_disagreement", "state_distance"), default="gp_uncertainty")
    parser.add_argument("--minimum-free-gb", type=float, default=8.0)
    args = parser.parse_args()
    if args.output.exists(): raise SystemExit(f"Refusing to overwrite output directory: {args.output}")
    if not args.gp_model.is_file(): raise FileNotFoundError(args.gp_model)
    data = json.loads(args.gp_config.read_text(encoding="utf-8"))["request_thresholds_from_validation_only"]
    choices = {"gp_uncertainty": ("gp_uncertainty", "gp_uncertainty_standardized_rms_90th"),
               "gp_vla_disagreement": ("gp_vla_disagreement", "gp_vla_disagreement_standardized_rms_90th"),
               "state_distance": ("state_distance", "nearest_training_state_distance_90th")}
    key, src = choices[args.monitoring_mode]
    thresholds = {key: float(data[src])}
    policy, gp, task, preflight = load_model(args, thresholds)
    args.output.mkdir(parents=True)
    worker = None
    try:
        worker = SharedWorker(args.output, args, policy, gp, thresholds, preflight)
        session = worker.session
        session.integration.update(task_instruction=task, monitoring_config=str(args.gp_config))
        cfg = json.loads((args.output / "configuration.json").read_text())
        cfg.update(session.integration)
        (args.output / "configuration.json").write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n")
        (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        token = secrets.token_urlsafe(24)
        page = cockpit.HTML.read_text(encoding="utf-8")
        page = page.replace("<title>LIBERO Human Teleoperation</title>", "<title>LIBERO pi0.5 + GP Shared Autonomy</title>")
        buttons = '''<div class="group"><span class="section-label">Shared autonomy</span><div class="gripper-row">
<button id="startPi05" class="primary">Start frozen pi0.5</button><button id="requestControl">Request human control</button>
<button id="acceptControl" class="primary">Accept handoff</button><button id="returnPi05" class="primary">Return control to pi0.5</button>
</div><span id="autonomyHelp" class="tiny">Human control begins only after a chunk-boundary request is accepted.</span></div>
<div class="safety-row">'''
        page = page.replace('<div class="safety-row">', buttons)
        page = page.replace("const active=s.mode==='HUMAN'&&!busy;", "const active=s.mode==='HUMAN'&&!busy&&(!s.vla_loaded||s.autonomy_started);$('startPi05').disabled=!s.vla_loaded||s.autonomy_started||s.mode!=='HUMAN'||busy;$('requestControl').disabled=s.mode!=='AUTONOMY'||busy;$('acceptControl').disabled=s.mode!=='ASSISTANCE_REQUESTED'||busy;$('returnPi05').disabled=!s.vla_loaded||!s.autonomy_started||s.mode!=='HUMAN'||busy;$('autonomyHelp').textContent=s.last_error||('Policy '+(s.vla_loaded?'loaded':'unavailable')+' · '+s.assistance_request);")
        page = page.replace("$('estop').disabled=!['HUMAN','PAUSED'].includes(s.mode)||busy;", "$('estop').disabled=['ESTOP','FINISHED'].includes(s.mode)||busy;")
        page = page.replace("$('pause').onclick=async()=>", '''$('startPi05').onclick=async()=>{try{update(await request('/api/autonomy/start',{}));}catch(e){$('message').textContent=e.message;}};
$('requestControl').onclick=async()=>{try{const r=await request('/api/control/request',{origin:'human_operator'});$('message').textContent=r.pending_chunk_boundary?'Request queued for the next 10-action chunk boundary.':'Request recorded.';}catch(e){$('message').textContent=e.message;}};
$('acceptControl').onclick=async()=>{stopMove();try{update(await request('/api/control/accept',{}));}catch(e){$('message').textContent=e.message;}};
$('returnPi05').onclick=async()=>{stopMove();try{update(await request('/api/autonomy/resume',{}));}catch(e){$('message').textContent=e.message;}};
$('pause').onclick=async()=>''')
        page = page.replace("__TOKEN__", token)

        class Handler(BaseHTTPRequestHandler):
            server_version = "LIBERO-SharedAutonomy/0.1"
            def send_data(self, status, value, kind="application/json"):
                raw = value.encode() if isinstance(value, str) else json.dumps(value, allow_nan=False).encode()
                self.send_response(status); self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try: self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError): pass
            def auth(self):
                if self.headers.get("X-Teleop-Token") != token:
                    self.send_data(403, {"error": "local session token required"}); return False
                origin = self.headers.get("Origin")
                if origin and origin != f"http://127.0.0.1:{args.port}":
                    self.send_data(403, {"error": "cross-origin request rejected"}); return False
                return True
            def body(self):
                n = int(self.headers.get("Content-Length", "0"))
                if n <= 0 or n > 4096: raise ValueError("invalid request size")
                return json.loads(self.rfile.read(n))
            def do_GET(self):
                if self.path == "/": self.send_data(200, page, "text/html; charset=utf-8"); return
                if self.path != "/api/state": self.send_data(404, {"error": "not found"}); return
                if not self.auth(): return
                if not session.lock.acquire(blocking=False): self.send_data(409, {"error": "simulator busy"}); return
                try: self.send_data(200, session.snapshot())
                finally: session.lock.release()
            def do_POST(self):
                if not self.auth(): return
                try:
                    if self.path in {"/api/hold/start", "/api/heartbeat", "/api/hold/stop", "/api/hold/end"}:
                        b = self.body(); hid = b.get("hold_id")
                        if self.path.endswith("/start"): session.begin_hold(hid); result = {"ok": True}
                        elif self.path.endswith("/heartbeat"): result = {"ok": session.heartbeat(hid)}
                        elif self.path.endswith("/stop"): result = {"ok": session.stop_hold(hid)}
                        else: session.end_hold(hid); result = {"ok": True}
                        self.send_data(200, result); return
                    if self.path == "/api/reference": self.send_data(200, session.set_reference(self.body().get("reference"))); return
                    if self.path == "/api/display": self.send_data(200, session.set_wrist_display_rotation(self.body().get("wrist_rotation_deg"))); return
                    if self.path == "/api/action": self.send_data(200, worker.invoke(lambda: session.execute(self.body()))); return
                    if self.path == "/api/autonomy/start":
                        worker.invoke(session.start_autonomy); worker.start_loop(); self.send_data(200, session.snapshot()); return
                    if self.path == "/api/control/request":
                        body = self.body()
                        self.send_data(200, session.request_human(body.get("origin", "human_operator"))); return
                    if self.path == "/api/control/accept": self.send_data(200, worker.invoke(session.accept_human)); return
                    if self.path == "/api/autonomy/resume":
                        worker.invoke(session.resume_autonomy); worker.start_loop(); self.send_data(200, session.snapshot()); return
                    if self.path == "/api/estop": self.send_data(200, worker.invoke(session.emergency_stop)); return
                    if self.path in {"/api/pause", "/api/resume"}:
                        target = "PAUSED" if self.path == "/api/pause" else "HUMAN"
                        self.send_data(200, worker.invoke(lambda: session.transition(target))); return
                    self.send_data(404, {"error": "not found"})
                except RuntimeError as exc: self.send_data(409, {"error": str(exc)})
                except Exception as exc: self.send_data(400, {"error": f"{type(exc).__name__}: {exc}"})
            def log_message(self, fmt, *values): print("shared-autonomy-http", fmt % values, flush=True)

        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        server.daemon_threads = True
        (args.output / "server.pid").write_text(str(os.getpid()), encoding="ascii")
        print(f"SHARED_AUTONOMY_READY address=127.0.0.1:{args.port} output={args.output}", flush=True)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try: server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt: pass
        finally: server.server_close()
    finally:
        if worker: worker.close()
        else: gp.close()
        try:
            import torch
            if torch.cuda.is_available() and args.output.exists():
                record = {"device": torch.cuda.get_device_name(0),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                    "free_bytes_at_shutdown": int(torch.cuda.mem_get_info(0)[0]),
                    "checkpoint_revision": REV, "checkpoint_sha256": WEIGHT_SHA}
                (args.output / "gpu_memory.json").write_text(json.dumps(record, indent=2) + "\n")
        except Exception as exc:
            print(f"GPU memory report failed: {exc}", flush=True)
        (args.output / "server.pid").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
