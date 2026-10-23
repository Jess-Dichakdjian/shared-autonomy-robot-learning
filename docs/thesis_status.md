# Thesis implementation status

**Last updated:** 23 October 2026
**Canonical repository:** shared-autonomy-robot-learning
**Active branch:** feat/pi05-libero-foundation

## Current status

The LIBERO evaluation loop and LeRobot pi0.5 observation/action processing have
been validated with simulation and mock inference. A pretrained checkpoint is
cached on the Lenovo, but actual model loading and learned-policy inference
remain untested because the current WSL memory allocation is too small.

## Verified implementation

| Area | Verified evidence | Limits |
|---|---|---|
| LIBERO | Spatial task 0 reset, OSMesa rendering, two 256x256 cameras and a seven-value relative OSC step | Simulation integration only |
| Observation contract | Two HWC uint8 camera inputs, LeRobot image processors, eight-value state and tokenized instruction | Processor integration used mock policy inference |
| Policy bridge | LeRobot 0.6.2 pi0.5 processors, action postprocessing, chunk queue/reset and seven-value action validation | Actual weight-backed policy was not loaded on Lenovo |
| Evaluator | Seeded episodes, action/state/config JSONL, metrics and synchronized camera video | Only a short mock rollout is recorded |
| Mock result | LIBERO Spatial task 0, seed 123, three steps, OSMesa, mock zero-action policy; output copied to experiments/libero_mock_policy_smoke_2026-10-17/ | Intentionally capped; success 0.0, return 0.0; not a full episode or learned result |
| Tests | Default suite 10 passed / 2 skipped; LIBERO simulation and authenticated processor integration tests have passed when enabled | No real pretrained inference test |
| Checkpoint preparation | Pinned v044 checkpoint cached outside Git; revision recorded in docs/vla_foundation.md | Weights not loaded; no inference latency or memory peak measured |

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

1. Run the pinned checkpoint on a suitable CUDA Linux host. Begin with one
   LIBERO Spatial task 0 episode and inspect latency, memory, metrics and video.
2. If that run is stable, expand the real-policy baseline to reproducible seeds
   and report uncertainty intervals and failure cases.
3. Only after establishing the learned baseline, implement and evaluate the
   uncertainty/intervention/correction stages.
