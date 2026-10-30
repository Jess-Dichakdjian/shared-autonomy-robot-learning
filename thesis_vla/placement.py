"""Inference-only placement of pi0.5's unused vocabulary output heads.

The pinned action sampler calls the transformer backbones, never these heads.
Weights and modules are retained. Guards reject any future path that uses them.
"""
from collections import Counter
from types import MethodType, SimpleNamespace
import torch

HEAD_PATHS = (
    "model.paligemma_with_expert.paligemma.lm_head",
    "model.paligemma_with_expert.gemma_expert.lm_head",
)


def _reject_head_call(module, args):
    raise RuntimeError("CPU-resident vocabulary head was called; placement is invalid for this inference path")


def place_action_policy(policy, device):
    """Keep unused, unshared heads on CPU; move every other tensor to device.

    Must run after strict CPU loading and eval(), before any whole-model GPU move.
    No casting, deleting, tying or replacing parameters is performed.
    """
    if policy.training:
        raise ValueError("Selective placement is inference-only; call eval() first")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    heads = {path: policy.get_submodule(path) for path in HEAD_PATHS}
    excluded = {id(p) for h in heads.values() for p in h.parameters()}
    # Check both parameter aliases and distinct Parameters sharing storage.
    head_storage = {p.untyped_storage().data_ptr() for h in heads.values() for p in h.parameters()}
    for name, p in policy.named_parameters(remove_duplicate=False):
        in_head = any(name.startswith(path + ".") for path in HEAD_PATHS)
        if not in_head and (id(p) in excluded or p.untyped_storage().data_ptr() in head_storage):
            raise ValueError(f"Vocabulary head shares active parameter storage: {name}")
    if any(p.device.type != "cpu" for p in policy.parameters()):
        raise ValueError("Placement requires a strictly loaded CPU policy")

    def move(module, path=""):
        if path in heads:
            module.to("cpu")
            module.register_forward_pre_hook(_reject_head_call)
        elif not any(h.startswith(path + ".") for h in heads) and path:
            module.to(device)
        else:
            module._apply(lambda t: t.to(device), recurse=False)
            for name, child in module.named_children():
                move(child, f"{path}.{name}" if path else name)
    move(policy)
    active_parameter = next(p for p in policy.parameters() if id(p) not in excluded)
    if active_parameter.device != device:
        raise RuntimeError("Active parameter is not on the inference device")
    if hasattr(policy, "_preprocess_images"):
        # Execute the exact pinned function with its original config. Its only
        # parameter lookup chooses an image device; an unused CPU head must not
        # decide that device. Pixel transforms, masks and camera order are intact.
        original = policy._preprocess_images.__func__
        def preprocess_images(self, batch):
            view = SimpleNamespace(config=self.config, parameters=lambda: iter((active_parameter,)))
            return original(view, batch)
        policy._preprocess_images = MethodType(preprocess_images, policy)
    return placement_report(policy)


def placement_report(policy):
    counts = Counter()
    for p in policy.parameters():
        counts[f"{p.device}/{p.dtype}"] += p.numel() * p.element_size()
    return {"parameter_bytes_by_device_dtype": dict(counts), "cpu_head_paths": list(HEAD_PATHS)}


def verify_prediction_gate(adapter, observation, *, seed, reference_head_device):
    """Real action first, then exact same-device comparison with heads on GPU 0.

    Active computation, inputs and RNG seed remain identical; only unused
    guarded head storage changes. This is not a full-CPU numerical reference.
    """
    import hashlib
    import time
    import numpy as np

    heads = [adapter.policy.get_submodule(path) for path in HEAD_PATHS]
    def fingerprints():
        return [(str(p.dtype), hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                for h in heads for p in h.parameters()]
    before = fingerprints()
    def predict():
        adapter.reset()
        torch.manual_seed(seed)
        started = time.perf_counter()
        with torch.inference_mode():
            proposal = adapter.act(observation)
        return proposal, time.perf_counter() - started
    first, first_seconds = predict()
    first_chunk = adapter.last_predicted_chunk.copy()
    try:
        for h in heads:
            h.to(reference_head_device)
        reference, reference_seconds = predict()
        reference_chunk = adapter.last_predicted_chunk.copy()
    finally:
        for h in heads:
            h.to("cpu")
        with torch.cuda.device(reference_head_device):
            torch.cuda.empty_cache()
    np.testing.assert_array_equal(first.predicted_action, reference.predicted_action)
    np.testing.assert_array_equal(first_chunk, reference_chunk)
    np.testing.assert_array_equal(first.executed_action, reference.executed_action)
    if before != fingerprints():
        raise RuntimeError("Head weights/dtypes changed during placement comparison")
    adapter.reset()
    return {"passed": True, "seed": seed, "predicted_action": first.predicted_action.tolist(),
            "executed_action": first.executed_action.tolist(), "first_prediction_seconds": first_seconds,
            "reference_prediction_seconds": reference_seconds, "max_absolute_action_difference": 0.0,
            "full_chunk_shape": list(first_chunk.shape), "predicted_chunk": first_chunk.tolist(),
            "comparison": "CPU heads versus other-GPU heads; active computation identical; bitwise equal full chunk and executed action",
            "head_fingerprints_preserved": True}
