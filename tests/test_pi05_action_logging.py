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
            assert cfg.device == "cpu" and not cfg.compile_model
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
