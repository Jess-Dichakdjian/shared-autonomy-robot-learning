# Real pretrained pi0.5 inference on the UCL workstation

Verified on LX00060258 on 30 October 2026. This supersedes the initial
full-GPU OOM as the current deployment status; that failure remains documented
in [ucl_workstation_setup.md](ucl_workstation_setup.md).

## Smallest working placement

- Physical GPU 0: NVIDIA EGL rendering.
- Physical GPU 1: all active pi0.5 inference parameters and computation.
- System RAM: the two unused vocabulary output heads, retained with their
  original weights, shapes and BF16 dtype (1,579,941,888 bytes / 1.471 GiB).
- Active GPU parameters occupy 7.240 GiB, retaining the official FP32/BF16 mix.

The pinned action sampler calls the underlying language models directly;
it does not evaluate either `paligemma.lm_head` or `gemma_expert.lm_head`.
Keeping these modules in RAM changes placement only. No layers are removed,
weights tied, tensors quantized, normalization replaced, cameras removed,
inference steps reduced, or weights fine-tuned. CPU RAM is ample on this host.

The placement checks reject shared storage between offloaded heads and active
parameters. Forward guards fail if either head is called. The real policy's
first registered parameter is an unused head, so the image processor cannot
infer its execution device from that parameter. A narrow adapter executes the
**exact pinned image-preprocessing function** with the original config and an
active parameter supplying its device. Pixel transforms and masks are unchanged.
Tests compare this path bitwise with the original GPU preprocessing.

This is static CPU residency for unused modules, not dynamic offloading of
active transformer layers. It needs no additional dependency or model surgery.

## Investigation and validation before model inference

The two RTX 3080s connect through PCIe/host bridges, with no NVLink reported.
Moving real LIBERO EGL rendering from GPU 1 to GPU 0 recovered about 727 MiB on
GPU 1. Both camera SHA-256 hashes were identical across those render placements.
That is useful headroom, but a rendering-only full-model retry was not needed:
combining separation with CPU head residency recovers another 1.471 GiB.

The lightweight suite passed **18 tests**, with two standard opt-in integrations
skipped. New tests ran on both available GPUs and cover:

- Exact numerical agreement before/after unused-head placement on a small mixed
  precision policy; all state-dict keys, weights and dtypes remain identical.
- Training-mode and shared-storage rejection, and unexpected head-call failure.
- Full-chunk comparison and CPU restoration in the prediction gate.
- The real pinned image function: exact resized pixel values, devices, masks
  and empty-camera behavior with a CPU first parameter.

A placement guard caught the first-parameter device dependency before any
prediction. The explicit device shim and its regression test resolved it.
The original all-on-GPU OOM configuration was not retried.

## Real inference and episode evidence

The same pinned checkpoint was strictly loaded with its saved processors and
normalization. No network access was needed. The first real `(1, 50, 7)` action
chunk completed in 0.561 seconds. Before any evaluated action, the gate repeated
prediction on the same observation and RNG seed with the unused heads stored
temporarily on GPU 0. The **entire chunk and postprocessed first action were
bitwise equal**, with maximum difference 0.0. Head weight/dtype fingerprints
matched after restoring them to CPU. Head-call guards remained active.

This comparison isolates unused-head placement while active GPU kernels remain
identical. It is not a numerical comparison to an unavailable original all-on-one-
GPU forward, nor proof of bitwise reproducibility on every platform.

| Measurement | Observed result |
|---|---:|
| Suite/task | LIBERO-Spatial / 0 |
| Seed | 0 |
| Gate saved initial-state index | 0 |
| Evaluated episode saved initial-state index | 1 |
| Outcome | success, terminated, not truncated |
| Return / evaluated steps | 1.0 / 78 |
| Chunk predictions in episode | 8 |
| Mean / median chunk latency | 0.292 / 0.289 s |
| Maximum episode chunk latency | 0.308 s |
| Policy load time | 100.53 s |
| GPU 1 peak PyTorch allocated / reserved | 8.063 / 8.473 GiB |
| Episode-only sampled total GPU 1 / GPU 0 usage | 9,035 / 1,288 MiB |
| Total GPU 0 peak including temporary-head comparison | 2,798 MiB |
| Video | 79 RGB frames, 512x256, both cameras |

Latencies measure chunk-generating `act` calls, including preparation and action
postprocessing; queued-action calls are excluded. Allocator peaks cover loading,
the gate and episode. Total GPU memory samples are taken every 500 ms and include
rendering, driver contexts and desktop use. They are sampled peaks, not exact
continuous maxima. The earlier guard attempt is retained separately.

LIBERO increments `init_state_id` on each reset. The gate consumes the first
reset, and the episode consumes the second. Their initial observations and first
actions therefore differ; the numerical-consistency test compares the two gate
predictions on the *same* observation. Reproducing this command includes both
resets. One successful episode is feasibility evidence, not a multi-seed success
rate or final thesis benchmark.

Verified local artifacts (ignored by Git):

- `outputs/ucl_pi05_cpu_heads_2026-10-30/`: `run_config.json`,
  `prediction_gate.json` including the full predicted chunk, `steps.jsonl`,
  `episodes.jsonl`, `metrics.json`, `gpu_memory.json`, `episode_000.mp4`.
- `outputs/ucl_placement_2026-10-30/`: test logs, EGL deltas/camera hashes,
  both-GPU samples, run log, guard evidence and `verified_result.json`.

The trajectory contains finite raw predictions, executed seven-value actions
within the existing bounds, robot states, seeds, chunk indices and latencies.
The success/termination fields, all 78 rows and all 79 video frames were checked.
The final frame was visually inspected. Video SHA-256:
`09fc972a74b696ac91a149024d1016224ee2841057b93ef690c6f4c343557122`.

## Reproduction

Retain the environment/checkpoint/tokenizer/asset pins and hashes from the
workstation setup document. No dependency versions changed for this work.

```bash
source ~/.venvs/shared-autonomy-robot-learning-pi05/bin/activate
CUDA_VISIBLE_DEVICES=0,1 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=. \
python scripts/evaluate_libero_pi05.py \
  --checkpoint /home/jessica/models/pi05_libero_finetuned_v044 \
  --suite libero_spatial --task-id 0 --episodes 1 --seed 0 \
  --execute-steps 10 --device cuda:1 --dtype auto --gl egl \
  --cpu-output-heads --prediction-gate --output outputs/NEW_UNIQUE_RUN_DIRECTORY
```

Expose both GPUs: robosuite requires the EGL device to be among visible devices.
Use a new output directory to preserve previous evidence. Record both GPUs with
`nvidia-smi -i 0,1 --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv -lms 500`.
The runner records all visible CUDA allocator peaks and the extra gate reset.

## Other partitioning options

Moving the vision tower and projector to GPU 0 would require explicit image and
feature transfers around the pinned `embed_image` boundary, preserving autocast
and FP32 storage. Splitting paired transformer layers would also require careful
cache, rotary-position and activation placement. Neither is necessary for the
working configuration and neither was implemented. Generic Accelerate hooks are
not a drop-in replacement for the pinned custom sampling/attention path.

Sources: [pinned pi0.5 code](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/policies/pi05/modeling_pi05.py),
[PyTorch device semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html),
[Accelerate CPU-offload API](https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.cpu_offload).
