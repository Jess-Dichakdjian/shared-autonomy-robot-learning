from types import SimpleNamespace
import pytest
import torch
from thesis_vla.placement import HEAD_PATHS, place_action_policy


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


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_placement_preserves_values_dtypes_and_computation(device):
    if device.startswith("cuda") and (not torch.cuda.is_available() or torch.cuda.device_count() < 2):
        pytest.skip("requires two CUDA devices")
    torch.manual_seed(71)
    p = SmallPolicy().eval()
    original = {k: v.clone() for k, v in p.state_dict().items()}
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    # Reference uses identical kernels/device; only inactive head placement differs.
    reference = SmallPolicy().eval().to(device)
    reference.load_state_dict(original)
    expected = reference(x.to(device))
    place_action_policy(p, device)
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
