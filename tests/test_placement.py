from types import SimpleNamespace
import numpy as np
import pytest
import torch
from thesis_vla.placement import (HEAD_PATHS, place_action_policy,
    verify_single_gpu_action_contract, verify_single_gpu_placement)


class SmallPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.paligemma_with_expert = torch.nn.Module()
        for name in ("paligemma", "gemma_expert"):
            block = torch.nn.Module()
            # Match the real policy's first-parameter trap.
            block.lm_head = torch.nn.Linear(8, 16, bias=False).to(torch.bfloat16)
            block.backbone = torch.nn.Linear(8, 8).to(torch.bfloat16)
            block.norm = torch.nn.LayerNorm(8)  # retain mixed precision
            setattr(self.model.paligemma_with_expert, name, block)
    def forward(self, x):
        for block in self.model.paligemma_with_expert.children():
            x = block.norm(block.backbone(x).float()).to(torch.bfloat16)
        return x


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
def test_placement_preserves_values_dtypes_and_computation(device):
    if device.startswith("cuda") and (not torch.cuda.is_available() or torch.cuda.device_count() <= int(device.split(":")[1])):
        pytest.skip(f"requires visible {device}")
    torch.manual_seed(71)
    p = SmallPolicy().eval()
    original = {k: v.clone() for k, v in p.state_dict().items()}
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    # Reference uses identical kernels/device; only inactive head placement differs.
    reference = SmallPolicy().eval().to(device)
    reference.load_state_dict(original)
    expected = reference(x.to(device))
    placement = place_action_policy(p, device, verify_integrity=True)
    assert placement["weight_integrity_sha256_before_after"]
    actual = p(x.to(device))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert p.state_dict().keys() == original.keys()
    for name, value in p.state_dict().items():
        assert value.dtype == original[name].dtype
        torch.testing.assert_close(value.cpu(), original[name], rtol=0, atol=0)
        assert value.device.type == ("cpu" if any(name.startswith(h + ".") for h in HEAD_PATHS) else torch.device(device).type)
    with pytest.raises(RuntimeError, match="head was called"):
        p.get_submodule(HEAD_PATHS[0])(x)


def test_shared_head_storage_and_training_are_rejected():
    p = SmallPolicy()
    with pytest.raises(ValueError, match="inference-only"):
        place_action_policy(p, "cpu")
    p.eval()
    block = p.model.paligemma_with_expert.paligemma
    block.lm_head.weight = torch.nn.Parameter(block.backbone.weight.detach())
    with pytest.raises(ValueError, match="shares active"):
        place_action_policy(p, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_prediction_gate_compares_full_chunk_and_restores_heads():
    from thesis_vla.placement import verify_prediction_gate
    from thesis_vla.pi05 import Pi05PolicyAdapter
    p = SmallPolicy().eval()
    place_action_policy(p, "cuda:1")
    p.reset = lambda: None
    p.predict_action_chunk = lambda batch: p(torch.randn(2, 8, device="cuda:1", dtype=torch.bfloat16))[:, :7].float()[None]
    adapter = Pi05PolicyAdapter(p, lambda obs, task: {}, task="test")
    result = verify_prediction_gate(adapter, {}, seed=71, reference_head_device="cuda:0")
    assert result["passed"] and result["full_chunk_shape"] == [1, 2, 7]
    assert all(t.device.type == "cpu" for h in HEAD_PATHS for t in p.get_submodule(h).parameters())


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_pinned_image_preprocessing_is_bitwise_equal_with_cpu_first_parameter():
    from types import MethodType
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    p = SmallPolicy().eval()
    p.config = SimpleNamespace(image_features={"image": None, "image2": None, "empty": None}, image_resolution=(224, 224))
    p._preprocess_images = MethodType(PI05Policy._preprocess_images, p)
    # Baseline runs the unmodified pinned processor with a GPU first parameter.
    p.to("cuda:1")
    batch = {k: torch.linspace(0, 1, 3*256*256, device="cuda:1").reshape(1,3,256,256) for k in ("image", "image2")}
    expected = p._preprocess_images(batch)
    p.to("cpu")
    place_action_policy(p, "cuda:1")
    assert next(p.parameters()).device.type == "cpu"
    actual = p._preprocess_images(batch)
    for values, baseline in zip(actual, expected, strict=True):
        for value, reference in zip(values, baseline, strict=True):
            assert value.device == torch.device("cuda:1")
            torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() != 1,
                    reason="requires the one-GPU BEAST configuration")
def test_single_gpu_placement_and_action_contract():
    torch.manual_seed(71)
    policy = SmallPolicy().eval()
    placement = place_action_policy(policy, "cuda:0", verify_integrity=True)
    validated = verify_single_gpu_placement(policy, "cuda:0")
    assert placement["weight_integrity_sha256_before_after"]
    assert validated["visible_cuda_devices"] == 1
    assert all(p.device.type == "cpu" for h in HEAD_PATHS for p in policy.get_submodule(h).parameters())

    class ChunkPolicy:
        def predict_action_chunk(self, batch):
            values = np.zeros((1, 50, 7), dtype=np.float32)
            values[:, 0, 0] = 0.25
            return values
        def reset(self):
            pass

    from thesis_vla.pi05 import Pi05PolicyAdapter
    adapter = Pi05PolicyAdapter(ChunkPolicy(), lambda obs, task: {}, task="test")
    adapter.prediction_state_capture = {"vla_normalized": np.zeros(8, dtype=np.float32)}
    observation = {"pixels": {
        "image": np.zeros((256, 256, 3), dtype=np.uint8),
        "image2": np.zeros((256, 256, 3), dtype=np.uint8),
    }}
    result = verify_single_gpu_action_contract(adapter, observation)
    assert result["passed"]
    assert result["predicted_chunk_shape"] == [1, 50, 7]
    assert result["checkpoint_normalized_state_shape"] == [8]



def test_meta_pi05_nonpersistent_buffers_are_reconstructed_without_touching_weights():
    from thesis_vla.placement import materialize_pi05_meta_buffers

    class Position(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_positions = 4
            with torch.device("meta"):
                self.register_buffer("position_ids", torch.empty((1, 4), dtype=torch.long), persistent=False)

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scalar_embed_scale = 2.5
            with torch.device("meta"):
                self.register_buffer("embed_scale", torch.empty(()), persistent=False)

    class Rotary(torch.nn.Module):
        rope_type = "default"
        config = object()
        attention_scaling = None

        def __init__(self):
            super().__init__()
            with torch.device("meta"):
                self.register_buffer("inv_freq", torch.empty((2,)), persistent=False)
                self.register_buffer("original_inv_freq", torch.empty((2,)), persistent=False)

        @staticmethod
        def compute_default_rope_parameters(config, device=None):
            return torch.tensor([1.0, 0.25], device=device), 1.0

    class MetaBufferPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = Position()
            self.embedding = Embedding()
            self.rotary = Rotary()
            self.weight = torch.nn.Parameter(torch.tensor([3.0]))

    policy = MetaBufferPolicy()
    materialized = materialize_pi05_meta_buffers(policy)
    assert set(materialized) == {
        "vision.position_ids",
        "embedding.embed_scale",
        "rotary.inv_freq",
        "rotary.original_inv_freq",
    }
    assert not any(buffer.is_meta for buffer in policy.buffers())
    torch.testing.assert_close(policy.vision.position_ids, torch.tensor([[0, 1, 2, 3]]))
    torch.testing.assert_close(policy.embedding.embed_scale, torch.tensor(2.5))
    torch.testing.assert_close(policy.rotary.inv_freq, torch.tensor([1.0, 0.25]))
    torch.testing.assert_close(policy.rotary.original_inv_freq, policy.rotary.inv_freq)
    torch.testing.assert_close(policy.weight, torch.tensor([3.0]))
    assert policy.rotary.attention_scaling == 1.0


def test_meta_buffer_reconstruction_fails_closed_on_unknown_buffer():
    from thesis_vla.placement import materialize_pi05_meta_buffers

    class Unknown(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.device("meta"):
                self.register_buffer("unknown", torch.empty((1,)), persistent=False)

    with pytest.raises(RuntimeError, match="Cannot safely reconstruct"):
        materialize_pi05_meta_buffers(Unknown())
