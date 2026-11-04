import json
import numpy as np

from thesis_vla.gp.client import GPInferenceClient


class FakePipe:
    def __init__(self, lines=()):
        self.lines = list(lines)
        self.writes = []
        self.closed = False

    def readline(self):
        return self.lines.pop(0) if self.lines else ""

    def write(self, text):
        self.writes.append(text)

    def flush(self):
        pass

    def close(self):
        self.closed = True


MONITORING_METADATA = {
    "target_scale": [1.0] * 7,
    "input_mean": [0.0] * 8,
    "input_scale": [1.0] * 8,
    "train_states_scaled": [[0.0] * 8],
}


class FakeProcess:
    def __init__(self, lines):
        self.stdin = FakePipe()
        self.stdout = FakePipe(lines)
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return self.returncode


def test_isolated_gp_client_streams_state_and_returns_contract(tmp_path, monkeypatch):
    model = tmp_path / "models.joblib"
    model.write_bytes(b"test model artifact")
    python = tmp_path / "python"
    python.write_text("fixture")
    response = {"ok": True, "mean": [0.1] * 7, "std": [0.2] * 7, "prediction_seconds": 0.003}
    process = FakeProcess([
        json.dumps({"ready": True, "model_sha256": "fixture-sha256",
                     "monitoring_metadata": MONITORING_METADATA}) + "\n",
        json.dumps(response) + "\n",
    ])
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr("thesis_vla.gp.client.subprocess.Popen", fake_popen)
    client = GPInferenceClient(model, python_executable=python)
    mean, std = client.predict(np.arange(8, dtype=np.float32))
    assert mean.shape == std.shape == (7,)
    np.testing.assert_allclose(mean, 0.1)
    np.testing.assert_allclose(std, 0.2)
    assert client.model_sha256 == "fixture-sha256"
    assert client.last_worker_prediction_seconds == 0.003
    scores = client.monitoring_scores(np.zeros(8), np.zeros(7), mean, std)
    assert scores["gp_uncertainty_standardized_rms"] == 0.2
    assert scores["gp_vla_disagreement_standardized_rms"] == 0.1
    assert scores["nearest_training_state_distance"] == 0.0
    assert json.loads(process.stdin.writes[0])["state"] == list(np.arange(8, dtype=float))
    client.close()
    assert json.loads(process.stdin.writes[1]) == {"op": "close"}
    assert calls[0][1]["cwd"] is not None


def test_isolated_gp_client_rejects_bad_state_before_ipc(tmp_path, monkeypatch):
    model = tmp_path / "models.joblib"
    model.write_bytes(b"test model")
    python = tmp_path / "python"
    python.write_text("fixture")
    process = FakeProcess([json.dumps({"ready": True, "model_sha256": "fixture",
                             "monitoring_metadata": MONITORING_METADATA}) + "\n"])
    monkeypatch.setattr("thesis_vla.gp.client.subprocess.Popen", lambda *a, **k: process)
    client = GPInferenceClient(model, python_executable=python)
    try:
        with np.testing.assert_raises_regex(ValueError, "finite normalized 8D state"):
            client.predict(np.zeros(7))
        assert process.stdin.writes == []
    finally:
        client.close()


def test_isolated_gp_client_detects_worker_exit(tmp_path, monkeypatch):
    model=tmp_path/"models.joblib";model.write_bytes(b"test model")
    python=tmp_path/"python";python.write_text("fixture")
    ready={"ready":True,"model_sha256":"fixture","monitoring_metadata":MONITORING_METADATA}
    process=FakeProcess([json.dumps(ready)+"\n"])
    monkeypatch.setattr("thesis_vla.gp.client.subprocess.Popen",lambda *a,**k:process)
    client=GPInferenceClient(model,python_executable=python)
    try:
        process.returncode=7
        try:
            client.predict(np.zeros(8))
        except RuntimeError as error:
            assert "worker exited with code 7" in str(error)
        else:
            raise AssertionError("expected worker-exit error")
    finally:
        client.close()
