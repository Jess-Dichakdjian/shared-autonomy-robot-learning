"""Inference-only placement of pi0.5's unused vocabulary output heads.

The pinned action sampler calls the transformer backbones, never these heads.
Weights and modules are retained. Guards reject any future path that uses them.
"""
from collections import Counter
import hashlib
from types import MethodType, SimpleNamespace
import torch

HEAD_PATHS = (
    "model.paligemma_with_expert.paligemma.lm_head",
    "model.paligemma_with_expert.gemma_expert.lm_head",
)


def _state_fingerprint(policy):
    """Hash state tensors incrementally without keeping a second model copy."""
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        value = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _reject_head_call(module, args):
    raise RuntimeError("CPU-resident vocabulary head was called; placement is invalid for this inference path")


def materialize_pi05_meta_buffers(policy):
    """Recreate known non-persistent pi0.5 buffers after meta parameter loading.

    Strict load_state_dict(assign=True) restores every checkpoint tensor but
    leaves constructor-only buffers on meta. Rebuild only deterministic buffers
    on CPU; unknown meta buffers fail closed rather than losing weights.
    """
    materialized = []
    for name, buffer in list(policy.named_buffers()):
        if not buffer.is_meta:
            continue
        module_path, _, buffer_name = name.rpartition(".")
        module = policy.get_submodule(module_path) if module_path else policy
        if buffer_name == "position_ids" and hasattr(module, "num_positions"):
            value = torch.arange(module.num_positions, device="cpu", dtype=buffer.dtype).expand(1, -1)
        elif buffer_name == "embed_scale" and hasattr(module, "scalar_embed_scale"):
            value = torch.tensor(module.scalar_embed_scale, device="cpu", dtype=buffer.dtype)
        elif buffer_name in {"inv_freq", "original_inv_freq"} and hasattr(module, "compute_default_rope_parameters"):
            if getattr(module, "rope_type", "default") == "default":
                rope_init = module.compute_default_rope_parameters
            else:
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
                rope_init = ROPE_INIT_FUNCTIONS[module.rope_type]
            inv_freq, attention_scaling = rope_init(module.config, device=torch.device("cpu"))
            module.attention_scaling = attention_scaling
            value = inv_freq.to(dtype=buffer.dtype)
            if buffer_name == "original_inv_freq":
                value = value.clone()
        else:
            raise RuntimeError(f"Cannot safely reconstruct non-persistent meta buffer: {name}")
        module._buffers[buffer_name] = value
        materialized.append(name)
    remaining = [name for name, buffer in policy.named_buffers() if buffer.is_meta]
    if remaining:
        raise RuntimeError(f"Meta buffers remain after pi0.5 initialization: {remaining}")
    return materialized


def place_action_policy(policy, device, *, verify_integrity=False):
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
    before_fingerprint = _state_fingerprint(policy) if verify_integrity else None

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
    report = placement_report(policy)
    if verify_integrity:
        after_fingerprint = _state_fingerprint(policy)
        if before_fingerprint != after_fingerprint:
            raise RuntimeError("Model state changed during single-device placement")
        report["weight_integrity_sha256_before_after"] = before_fingerprint
        report["weight_integrity_comparison"] = "exact state_dict byte fingerprint before/after device placement"
    return report


def verify_single_gpu_placement(policy, device="cuda:0"):
    """Check single-GPU active tensors and CPU residency of guarded output heads."""
    device = torch.device(device)
    if device.type != "cuda" or (device.index or 0) != 0:
        raise ValueError("single-GPU validation requires cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("single-GPU validation requires exactly one visible CUDA device")
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        in_head = any(name.startswith(path + ".") for path in HEAD_PATHS)
        expected = torch.device("cpu") if in_head else device
        if parameter.is_meta or parameter.device != expected:
            raise RuntimeError(f"parameter placement mismatch for {name}: expected {expected}, got {parameter.device}")
    report = placement_report(policy)
    report["single_gpu_device"] = str(device)
    report["visible_cuda_devices"] = torch.cuda.device_count()
    report["two_gpu_numerical_comparison"] = "not run; single-GPU action and placement contracts checked instead"
    return report


def verify_single_gpu_action_contract(adapter, observation, *, expected_chunk_steps=50):
    """Run one inference and check cameras, normalized state, chunk, and environment action."""
    import time
    import numpy as np
    pixels = observation.get("pixels", {}) if isinstance(observation, dict) else {}
    cameras = {key: np.asarray(pixels[key]) for key in ("image", "image2") if key in pixels}
    if set(cameras) != {"image", "image2"}:
        raise ValueError(f"expected camera inputs image/image2, got {sorted(cameras)}")
    if any(image.shape != (256, 256, 3) for image in cameras.values()):
        raise ValueError(f"expected two 256x256 RGB cameras, got { {k: v.shape for k, v in cameras.items()} }")
    if getattr(adapter, "execute_steps", None) != 10:
        raise ValueError("the validated controller chunk must execute exactly ten actions")
    started = time.perf_counter()
    with torch.inference_mode():
        proposal = adapter.act(observation)
    elapsed = time.perf_counter() - started
    chunk = np.asarray(adapter.last_predicted_chunk)
    state = getattr(adapter, "prediction_state_capture", {}).get("vla_normalized")
    if state is None or np.asarray(state).shape != (8,):
        raise ValueError(f"expected checkpoint-normalized 8D state, got {None if state is None else np.asarray(state).shape}")
    if chunk.shape != (1, expected_chunk_steps, 7) or not np.isfinite(chunk).all():
        raise ValueError(f"expected finite action chunk (1, {expected_chunk_steps}, 7), got {chunk.shape}")
    predicted, executed = np.asarray(proposal.predicted_action), np.asarray(proposal.executed_action)
    if predicted.shape != (7,) or executed.shape != (7,) or not np.isfinite(predicted).all() or not np.isfinite(executed).all():
        raise ValueError("predicted and executed actions must be finite seven-vectors")
    if np.any(executed < -1.0) or np.any(executed > 1.0):
        raise ValueError("postprocessed environment action is outside [-1, 1]")
    return {
        "passed": True,
        "camera_shapes": {name: list(image.shape) for name, image in cameras.items()},
        "checkpoint_normalized_state_shape": list(np.asarray(state).shape),
        "predicted_chunk_shape": list(chunk.shape),
        "execute_steps": adapter.execute_steps,
        "predicted_chunk_min": float(chunk.min()),
        "predicted_chunk_max": float(chunk.max()),
        "predicted_action": predicted.tolist(),
        "executed_action": executed.tolist(),
        "executed_action_bounds": [float(executed.min()), float(executed.max())],
        "inference_seconds": elapsed,
        "two_gpu_numerical_comparison": "not run on this single-GPU host",
    }


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
