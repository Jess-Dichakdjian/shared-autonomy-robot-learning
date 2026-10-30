# UCL workstation evaluation setup

Host: `LX00060258`, user `jessica`, inspected 30 October 2026.
Canonical checkout: `/home/jessica/projects/shared-autonomy-robot-learning`.

## Reproducibility

- Python 3.12 environment: `~/.venvs/shared-autonomy-robot-learning-pi05`.
- LeRobot source: `146d8474932c8b566717eb2738ba459337ab71c4` (0.6.2).
- CUDA wheels selected: PyTorch 2.10.0 and torchvision 0.25.0, CUDA 12.6.
- Checkpoint: `lerobot/pi05_libero_finetuned_v044`, revision
  `7152f13e34c6c115d808aea0e5d08aca74eac14b`.
- LIBERO assets: `lerobot/libero-assets` dataset, revision
  `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.
- Gated PaliGemma tokenizer access verified on this host; cached tokenizer
  revision `35e4f46485b4d07967e7e9935bc3786aad50687c`. No PaliGemma
  model weights are needed or downloaded.
- Initial hardware: two RTX 3080 GPUs, 10 GiB each; GPU 1 selected for its
  greater free memory. Approximately 112 GiB available host RAM and 2.8 TiB
  free disk at inspection. Separate GPUs do not pool VRAM in this evaluator.

Use `CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 MUJOCO_GL=egl` and logical
`--device cuda` for the first run. NVIDIA EGL was validated; OSMesa is not
required on this host. No system packages or drivers were changed.

The pinned checkpoint specifies BF16, 50-action chunks, ten inference steps,
two cameras plus one empty camera, eight state values, seven action values,
and MEAN_STD state/action normalization. Preserve the checkpoint's processor
files and normalization tensors. The adapter disables compilation and gradient
checkpointing for the initial inference attempt; these are execution settings,
not learned parameters. Precision overrides are applied during construction so
the official mixed-precision parameter rules remain intact.

## Evaluator safeguards

The pinned upstream PI05 loader catches weight-loading errors and may return
an uninitialized model. The thesis adapter instead constructs on CPU, applies
the same upstream key conversion, and loads local safetensors strictly before
moving to the requested device. Errors propagate. Checkpoint files are never
downloaded by the evaluation runner.

Episode seeds now initialize Python, NumPy and PyTorch as well as LIBERO.
This controls policy sampling, but does not promise bitwise reproducibility
across different GPUs or software versions.

The real runner writes `gpu_memory.json` with peak PyTorch allocated/reserved
bytes, including a failed load inside its evaluation try/finally. External
`nvidia-smi` monitoring is also needed because rendering and other processes
consume memory outside PyTorch's allocator. The regular trajectory, metrics,
latency and side-by-side video outputs remain available.

## Evidence

Verified on this workstation:

- CUDA FP32 matrix result agrees with CPU; BF16 GPU matrix multiplication is
  finite and BF16 support is reported. PyTorch is `2.10.0+cu126`.
- `pip check`: no broken requirements. LeRobot installation metadata confirms
  source commit `146d8474932c8b566717eb2738ba459337ab71c4`.
- Unit suite: 13 passed, 2 opt-in integration tests skipped.
- Authenticated tokenizer/processor integration: 1 passed.
- LIBERO task-0 EGL smoke: both `(1, 256, 256, 3)` cameras and an action step
  passed. Direct EGL probes identify both physical devices as NVIDIA.
- LIBERO logging/video integration under EGL: 1 passed, three pre-existing
  robosuite logger deprecation warnings.
- Separate five-step mock rollout, seed 123, execute_steps 2: five JSONL
  trajectory rows, state/action fields, metrics, and decoded 512x256 camera
  video verified. Success 0.0 and return 0.0; this is mock evidence only.
- Saved v044 checkpoint processors/statistics also passed a real LIBERO
  observation through CUDA tokenization/preprocessing and mock inference.
- Weight-free CUDA/EGL preflight passed, with 9.38 GiB free CUDA memory and
  more than 3 TB free disk reported at that check. This is not proof of model
  inference capacity.

The fully resolved environment is captured in `requirements-pi05-ucl-lock.txt`.
Install CUDA wheels from the official cu126 index first, then use the pinned
research requirements with `-c constraints-pi05-ucl.txt`; alternatively replay
the lock with the cu126 index available. Video logging requires the explicitly
listed `imageio[ffmpeg]` dependency.

Setup evidence is stored locally under `outputs/ucl_setup_2026-10-30/` and is
ignored by Git. Checkpoints, caches and raw rollout outputs must not be pushed.
The existing versioned mock experiment remains distinct from the attempted
pretrained run described below.

## Pretrained attempt: weights loaded, no prediction

All checkpoint files were downloaded to
`/home/jessica/models/pi05_libero_finetuned_v044`. File sizes and every LFS
SHA-256 matched the pinned revision. The 7,473,096,344-byte model file has hash
`877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae`.

An 812-tensor metadata-only audit passed strict key/shape loading with the
official key conversion. It estimated 8.71 GiB of mixed-precision parameters.
The real attempt then strictly loaded weights on CPU and moved the policy to
GPU 1. The evaluator reset LIBERO and entered the first action request, but
CUDA OOM occurred inside the saved device processor while transferring the
first observation. The failed allocation requested 2 MiB with 1.94 MiB free.

Measured peaks:

| Measurement | Value |
|---|---:|
| PyTorch allocated | 9,356,805,120 bytes (8.714 GiB) |
| PyTorch reserved | 9,594,470,400 bytes (8.936 GiB) |
| Sampled total physical GPU usage | 9,874 MiB |
| Total GPU use after process exit | 49 MiB |

Total GPU usage includes rendering, driver/context memory and desktop use;
it is sampled every 500 ms, whereas allocator peaks are recorded by PyTorch.
The attempt exited, and the monitor was stopped. No action prediction,
evaluated environment action, episode outcome, inference latency, metrics or
video was produced. The empty step/episode files are failure artifacts, not a
trajectory. Do not assign success 0.0 to this attempt: success is unmeasured.
Model load time was not persisted by this failed attempt; the runner now saves
`run_config.json` before evaluating so later first-step failures retain it.

The runner's initial index-free `cuda` handling was corrected before the
weight-backed attempt and is covered by a regression test. The default device
resolves to the current logical CUDA index, preserving GPU visibility mapping.

The actual attempt used this command (stdout/stderr retained in the setup
evidence directory):

```bash
source ~/.venvs/shared-autonomy-robot-learning-pi05/bin/activate
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 MUJOCO_GL=egl \
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=. \
python scripts/evaluate_libero_pi05.py \
  --checkpoint /home/jessica/models/pi05_libero_finetuned_v044 \
  --suite libero_spatial --task-id 0 --episodes 1 --seed 0 \
  --execute-steps 10 --device cuda --dtype auto --gl egl \
  --output outputs/ucl_pi05_spatial_task0_first_2026-10-30
```

No shortened real episode or additional model variant was substituted. The
observed failure establishes insufficient headroom for this configuration,
not a universal claim that every possible 10 GiB deployment must fail.

## Alternatives investigated, not yet validated

1. **CPU offloading:** Accelerate documents `cpu_offload` and
   `load_checkpoint_and_dispatch` for retaining weights on CPU and moving
   modules to the execution GPU as needed. This can retain the same weights
   and normalization, at the cost of transfers. The pinned pi0.5 implementation
   has no offload/device-map configuration field. Its image preprocessing
   derives the input device from the first parameter, and its custom sampling
   and joint attention call submodules outside ordinary parent forwards.
   A generic offload hook may leave that first parameter on `meta`, so a tested
   execution-device adapter and careful hook placement are required. Offloading
   has not been installed or executed, and is not a proven drop-in CLI option.
2. **A GPU with more free VRAM:** preserves the current evaluator without a new
   device-placement path. Required capacity must be measured with the same
   checkpoint and full inference; the 80 GB training recipe is not an inference
   minimum. Access to another university GPU would require authorization.
3. **Full CPU inference:** the runner supports `--device cpu`, and this host has
   ample RAM. Small BF16 matmul, layer norm and attention primitives passed on
   the Intel Core i9-10920X. It lacks native AVX-512 BF16, so throughput is
   uncertain. No full-model CPU load/inference or rollout was attempted.
4. **Memory-placement/allocator experiments:** PyTorch's OOM message suggests
   expandable segments when fragmentation contributes. This could reduce some
   reserved-memory overhead, but does not establish room for forward activations.
   No retry or successful inference with such settings is claimed. Moving
   rendering to the other GPU likewise requires a separate measured attempt.

Quantization, changing checkpoint dtype rules, removing cameras, changing
normalization, reducing the inference procedure, and fine-tuning were not used.
The two GPU memory pools were not combined.

Sources: [Accelerate CPU offload API](https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.cpu_offload),
[Accelerate big-model inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling),
and the [pinned pi0.5 implementation](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/policies/pi05/modeling_pi05.py).

Reference: [official PyTorch CUDA wheel commands](https://pytorch.org/get-started/previous-versions/)
and [pinned LeRobot source](https://github.com/huggingface/lerobot/tree/146d8474932c8b566717eb2738ba459337ab71c4).
