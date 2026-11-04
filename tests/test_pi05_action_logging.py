import numpy as np

from thesis_vla.pi05 import Pi05PolicyAdapter


def test_runner_resolves_default_and_explicit_cuda_indices(monkeypatch):
    import torch
    from scripts.evaluate_libero_pi05 import _resolve_cuda_device
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert _resolve_cuda_device("cuda") == torch.device("cuda:1")
    assert _resolve_cuda_device("cuda:2") == torch.device("cuda:2")
    assert _resolve_cuda_device("cpu") is None


class ChunkPolicy:
    def predict_action_chunk(self, batch):
        result = np.zeros((1, 3, 7), dtype=np.float32)
        result[:, 1, 0] = 0.75
        return result
    def reset(self):
        pass


def test_proposal_preserves_raw_prediction_and_postprocessed_execution():
    policy = Pi05PolicyAdapter(
        ChunkPolicy(), lambda observation, task: {},
        postprocess_action=lambda chunk: np.asarray(chunk) * 2,
        task="test", execute_steps=2,
    )
    first = policy.act({})
    second = policy.act({})
    assert first.predicted_action[0] == 0
    assert first.executed_action[0] == 0
    assert second.predicted_action[0] == 0.75
    assert second.executed_action[0] == 1.0

def test_pretrained_loader_refuses_missing_local_checkpoint_without_downloading():
    import pytest

    with pytest.raises(FileNotFoundError, match="local checkpoint does not exist"):
        Pi05PolicyAdapter.from_pretrained(
            "/tmp/no-such-pi05-checkpoint", task="fixture", local_files_only=True
        )


def test_pretrained_loader_propagates_incompatible_weights(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    import torch
    from safetensors.torch import save_file
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05 import modeling_pi05

    config = SimpleNamespace(device="cuda", dtype=torch.float32)
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda *a, **kw: config)

    class TinyPolicy(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            assert cfg.device in {"cpu", "meta"} and not cfg.compile_model
            self.config = cfg
            self.weight = torch.nn.Parameter(torch.zeros(2))
        def _convert_openpi_state_dict(self, state):
            return state

    monkeypatch.setattr(modeling_pi05, "PI05Policy", TinyPolicy)
    save_file({"wrong_key": torch.ones(2)}, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        Pi05PolicyAdapter.from_pretrained(tmp_path, task="fixture", device="cpu")

    # A compatible checkpoint must actually replace initialized parameters.
    save_file({"weight": torch.tensor([0.25, 0.75])}, tmp_path / "model.safetensors")
    monkeypatch.setattr(Pi05PolicyAdapter, "from_lerobot_policy", lambda policy, **kw: policy)
    loaded = Pi05PolicyAdapter.from_pretrained(tmp_path, task="fixture", device="cpu")
    torch.testing.assert_close(loaded.weight, torch.tensor([0.25, 0.75]))
    assert not loaded.training


def test_meta_initialized_loader_assigns_checkpoint_weights_without_random_model_copy(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    from safetensors.torch import save_file
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05 import modeling_pi05

    config = SimpleNamespace(device="cuda", dtype=torch.float32)
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda *a, **kw: config)

    class TinyPolicy(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            assert cfg.device == "meta" and not cfg.compile_model
            self.config = cfg
            self.weight = torch.nn.Parameter(torch.zeros(2))
        def _convert_openpi_state_dict(self, state):
            return state

    monkeypatch.setattr(modeling_pi05, "PI05Policy", TinyPolicy)
    monkeypatch.setattr(Pi05PolicyAdapter, "from_lerobot_policy", lambda policy, **kw: policy)
    save_file({"weight": torch.tensor([0.25, 0.75])}, tmp_path / "model.safetensors")
    loaded = Pi05PolicyAdapter.from_pretrained(
        tmp_path, task="fixture", device="cpu", meta_init=True
    )
    assert loaded.weight.device.type == "cpu" and not loaded.weight.is_meta
    torch.testing.assert_close(loaded.weight, torch.tensor([0.25, 0.75]))



def test_passive_gp_is_chunk_boundary_only_and_does_not_change_vla_actions():
    class FakeGP:
        model_sha256 = "fixture-hash"
        last_worker_prediction_seconds = 0.001
        last_round_trip_seconds = 0.002

        def __init__(self):
            self.states = []

        def predict(self, state):
            self.states.append(np.asarray(state).copy())
            return np.full(7, 99.0), np.full(7, 0.5)

        def monitoring_scores(self, state, action, gp_mean, gp_std):
            return {"gp_uncertainty_standardized_rms": 0.9,
                    "gp_vla_disagreement_standardized_rms": 1.1,
                    "nearest_training_state_distance": 1.5}

    def make_adapter(passive_gp=None, monitoring_mode="passive_monitoring", thresholds=None):
        capture = {}
        def prepare(observation, task):
            capture["vla_normalized"] = np.asarray(observation["state"], dtype=np.float32).copy()
            return {}
        adapter = Pi05PolicyAdapter(ChunkPolicy(), prepare, task="fixture", execute_steps=2,
                                    passive_gp=passive_gp, monitoring_mode=monitoring_mode,
                                    monitoring_thresholds=thresholds)
        adapter.prediction_state_capture = capture
        return adapter

    baseline = make_adapter()
    gp = FakeGP()
    monitored = make_adapter(gp)
    observations = [{"state": np.arange(8, dtype=np.float32) + i} for i in range(3)]
    baseline_proposals = [baseline.act(obs) for obs in observations]
    monitored_proposals = [monitored.act(obs) for obs in observations]

    assert len(gp.states) == 2  # one fresh chunk at steps 0 and 2
    np.testing.assert_array_equal(gp.states[0], observations[0]["state"])
    np.testing.assert_array_equal(gp.states[1], observations[2]["state"])
    for plain, watched in zip(baseline_proposals, monitored_proposals, strict=True):
        np.testing.assert_array_equal(plain.predicted_action, watched.predicted_action)
        np.testing.assert_array_equal(plain.executed_action, watched.executed_action)
    assert monitored_proposals[0].passive_gp_prediction["model_sha256"] == "fixture-hash"
    assert monitored_proposals[0].passive_gp_prediction["input_state_vla_normalized"] == observations[0]["state"].tolist()
    assert monitored_proposals[1].passive_gp_prediction is None
    assert monitored_proposals[0].assistance_request["requested"] is False
    assert monitored_proposals[0].passive_gp_prediction["monitoring_scores"]["gp_uncertainty_standardized_rms"] == 0.9
    assert monitored_proposals[2].passive_gp_prediction["input_state_vla_normalized"] == observations[2]["state"].tolist()

    triggered = make_adapter(gp, monitoring_mode="gp_uncertainty", thresholds={"gp_uncertainty": 0.9})
    triggered_proposals = [triggered.act(obs) for obs in observations]
    assert triggered_proposals[0].assistance_request["requested"] is True
    for plain, watched in zip(baseline_proposals, triggered_proposals, strict=True):
        np.testing.assert_array_equal(plain.predicted_action, watched.predicted_action)
        np.testing.assert_array_equal(plain.executed_action, watched.executed_action)
