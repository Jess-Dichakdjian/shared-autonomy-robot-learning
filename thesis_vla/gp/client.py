"""Subprocess client for isolated CPU-only GP inference during VLA execution."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import numpy as np


class GPInferenceClient:
    """Keep GPInference in a separate CPU Python environment and stream states."""

    def __init__(self, model_path, *, python_executable):
        self.model_path = Path(model_path).expanduser().resolve()
        self.python_executable = Path(python_executable).expanduser().absolute()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GP model artifact does not exist: {self.model_path}")
        if not self.python_executable.is_file():
            raise FileNotFoundError(f"GP Python executable does not exist: {self.python_executable}")
        repo_root = Path(__file__).resolve().parents[2]
        worker = repo_root / "scripts" / "gp_inference_worker.py"
        self._process = subprocess.Popen(
            [str(self.python_executable), "-u", str(worker), "--model", str(self.model_path)],
            cwd=repo_root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        ready_line = self._process.stdout.readline()
        if not ready_line:
            self.close()
            raise RuntimeError("GP inference worker exited before becoming ready")
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"GP inference worker failed to load artifact: {ready.get('error')}")
        self.model_sha256 = ready["model_sha256"]
        metadata = ready.get("monitoring_metadata")
        if not isinstance(metadata, dict):
            self.close()
            raise RuntimeError("GP worker omitted training-scaler metadata for monitor scoring")
        self.target_scale = np.asarray(metadata.get("target_scale"), dtype=np.float64)
        self.input_mean = np.asarray(metadata.get("input_mean"), dtype=np.float64)
        self.input_scale = np.asarray(metadata.get("input_scale"), dtype=np.float64)
        self.train_states_scaled = np.asarray(metadata.get("train_states_scaled"), dtype=np.float64)
        if (self.target_scale.shape != (7,) or self.input_mean.shape != (8,) or
                self.input_scale.shape != (8,) or self.train_states_scaled.ndim != 2 or
                self.train_states_scaled.shape[1] != 8 or self.train_states_scaled.shape[0] == 0 or
                not all(np.isfinite(v).all() for v in (self.target_scale,self.input_mean,self.input_scale,self.train_states_scaled))):
            self.close()
            raise RuntimeError("GP worker supplied invalid monitor-scoring metadata")
        self.last_worker_prediction_seconds = None
        self.last_round_trip_seconds = None

    def predict(self, state):
        values = np.asarray(state, dtype=np.float64)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError(f"Passive GP input must be a finite normalized 8D state; got {values.shape}")
        if self._process.poll() is not None:
            raise RuntimeError(f"GP inference worker exited with code {self._process.returncode}")
        started = time.perf_counter()
        self._process.stdin.write(json.dumps({"state": values.tolist()}) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        elapsed = time.perf_counter() - started
        if not line:
            raise RuntimeError("GP inference worker closed its output unexpectedly")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"GP inference failed: {response.get('error')}")
        mean = np.asarray(response["mean"], dtype=np.float64)
        std = np.asarray(response["std"], dtype=np.float64)
        if mean.shape != (7,) or std.shape != (7,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise FloatingPointError("GP worker returned invalid seven-dimensional predictions")
        self.last_worker_prediction_seconds = float(response["prediction_seconds"])
        self.last_round_trip_seconds = elapsed
        return mean, std

    def monitoring_scores(self, state, vla_action, gp_mean, gp_std):
        from .monitoring import compute_monitoring_signals
        return compute_monitoring_signals(
            state, vla_action, gp_mean, gp_std,
            target_scale=self.target_scale, input_mean=self.input_mean,
            input_scale=self.input_scale, train_states_scaled=self.train_states_scaled,
        ).as_dict()

    def close(self):
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin:
                process.stdin.write(json.dumps({"op": "close"}) + "\n")
                process.stdin.flush()
                process.stdin.close()
            process.wait(timeout=5)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
