# Thesis implementation status

**Last updated:** 30 October 2026
**Canonical repository:** shared-autonomy-robot-learning
**Active branch:** codex/ucl-pi05-evaluation

## Current status

The UCL workstation now has a validated CUDA/EGL environment and the pinned,
hash-verified checkpoint. Strict pretrained weight loading succeeded on GPU 1
(RTX 3080, 10 GiB), but the first observation transfer ran out of VRAM before
an action prediction. No real-policy episode or inference latency is established.
The process exited and GPU memory was released. See ucl_workstation_setup.md
for the measured memory peaks and deployment alternatives.

The previous Lenovo RAM limitation remains separate historical context.
Current UCL checks: 13 unit tests passed, two opt-in integrations passed
separately, CUDA FP32/BF16 and EGL smoke passed, and a five-step mock rollout
produced verified trajectory, metrics and video artifacts.

## Verified implementation

| Area | Verified evidence | Limits |
|---|---|---|
| LIBERO | Spatial task 0 reset, OSMesa rendering, two 256x256 cameras and a seven-value relative OSC step | Simulation integration only |
| Observation contract | Two HWC uint8 camera inputs, LeRobot image processors, eight-value state and tokenized instruction | Processor integration used mock policy inference |
| Policy bridge | LeRobot 0.6.2 pi0.5 processors, action postprocessing, chunk queue/reset and seven-value action validation | Actual weight-backed policy was not loaded on Lenovo |
| Evaluator | Seeded episodes, action/state/config JSONL, metrics and synchronized camera video | Only a short mock rollout is recorded |
| Mock result | LIBERO Spatial task 0, seed 123, three steps, OSMesa, mock zero-action policy; output copied to experiments/libero_mock_policy_smoke_2026-10-17/ | Intentionally capped; success 0.0, return 0.0; not a full episode or learned result |
| Tests | UCL default suite 13 passed / 2 skipped; LIBERO EGL and authenticated processor integrations each passed explicitly | No real pretrained prediction established |
| UCL checkpoint preparation | Pinned v044 files cached outside Git and SHA-256 verified; strict CPU load and GPU transfer succeeded | First observation exhausted VRAM before prediction; peak allocated 8.71 GiB, reserved 8.94 GiB, sampled total GPU 9,874 MiB |

## Lenovo CPU feasibility

Windows reported 15.32 GiB physical RAM and 1.68 GiB available at inspection.
WSL had 7.42 GiB total, 6.81 GiB available and 2 GiB unused swap. The weights
are 7,473,096,344 bytes (about 6.96 GiB) before runtime memory. The CPU preflight
uses 1.5x the file size (10.44 GiB) as a conservative safety heuristic and
correctly stops before loading. Small BF16 matrix multiply, layer normalization
and attention operations work on the AMD Ryzen AI 7 445, but these checks do
not validate the full model.

No WSL memory setting was changed. A historical successful run on the user's
HP OMEN RTX 2060 is user-reported, but its original video/results were not found
in accessible project or Windows folders. Do not combine that report with the
new mock-pipeline evidence.

## Prior MuJoCo hover work

Earlier experiments demonstrated a scripted open-gripper hover above a red cube
and exercised model-independent logging. The 20-seed hover batch (seeds 51-70)
had 20/20 scripted hover successes and a mean final target error near 5.56 mm.
These are debugging-baseline results, not object manipulation, VLA, uncertainty
or human-intervention results. See docs/hover-pilot-results.md.

## Not implemented or not established

- A real pretrained pi0.5 action on a LIBERO observation.
- Full-episode or multi-seed learned-policy success rates.
- GP uncertainty estimates or uncertainty-triggered intervention.
- XR interface, human correction collection or learning from corrections.
- Fine-tuning or policy improvement from corrective demonstrations.

## Next steps

1. Establish a memory-feasible deployment of the same checkpoint: investigate
   CPU offloading with validated device handling, or a GPU with more memory.
   Then run one complete LIBERO Spatial task 0 episode and inspect latency,
   memory, metrics and video. Full CPU inference is another untested option.
2. If that run is stable, expand the real-policy baseline to reproducible seeds
   and report uncertainty intervals and failure cases.
3. Only after establishing the learned baseline, implement and evaluate the
   uncertainty/intervention/correction stages.
