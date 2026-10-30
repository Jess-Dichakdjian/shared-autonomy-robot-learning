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
- Unit suite: 12 passed, 2 opt-in integration tests skipped.
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
The existing versioned mock experiment remains distinct from any future real
pretrained result. Checkpoint download and real-inference results are pending.

Reference: [official PyTorch CUDA wheel commands](https://pytorch.org/get-started/previous-versions/)
and [pinned LeRobot source](https://github.com/huggingface/lerobot/tree/146d8474932c8b566717eb2738ba459337ab71c4).
