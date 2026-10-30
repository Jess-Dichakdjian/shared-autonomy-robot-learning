# Thesis implementation status

**Last updated:** 30 October 2026
**Canonical repository:** shared-autonomy-robot-learning
**Active branch:** codex/ucl-pi05-evaluation

## Current status

Real pretrained inference and one successful LIBERO-Spatial task-0 episode are
verified on UCL. The pinned, hash-verified checkpoint runs with EGL on GPU 0,
active inference on GPU 1, and unused vocabulary output heads retained in RAM.
Checkpoint weights, mixed precision and normalization are unchanged. A full
action chunk passed a bitwise consistency gate before the successful 78-step
episode. See pi05_memory_placement.md for commands and measured evidence.

The previous Lenovo RAM limitation remains separate historical context.
Current UCL checks: 18 unit/placement tests passed, two opt-in integrations passed
separately, CUDA FP32/BF16 and EGL smoke passed, and a five-step mock rollout
produced verified trajectory, metrics and video artifacts.

## Verified implementation

| Area | Verified evidence | Limits |
|---|---|---|
| LIBERO | Spatial task 0 EGL rendering, two 256x256 cameras and relative OSC actions during a complete episode | One task/episode only |
| Observation contract | Eight-value state, two cameras plus checkpoint empty camera, saved tokenization/normalization; device shim preserves original pixels/masks bitwise | Tested pinned configuration only |
| Policy bridge | Real predictions, strict loading, CPU head placement guards and full-chunk numerical consistency | Original all-on-one-GPU forward cannot fit for direct reference |
| Evaluator | Successful 78-step episode, state/action/config JSONL, metrics, chunk latency, memory and synchronized video | Seed 0; gate state 0 and episode saved initial state 1 |
| Mock result | LIBERO Spatial task 0, seed 123, three steps, OSMesa, mock zero-action policy; output copied to experiments/libero_mock_policy_smoke_2026-10-17/ | Intentionally capped; success 0.0, return 0.0; not a full episode or learned result |
| Tests | UCL default suite 18 passed / 2 skipped; earlier EGL and authenticated processor integrations each passed explicitly | No multi-seed benchmark |
| UCL real pretrained result | Success, return 1.0, 78 steps; first prediction 0.561 s; mean chunk latency 0.292 s; sampled episode GPU 1 peak 9,035 MiB | One feasibility result; initial full-GPU OOM remains historical evidence |

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

- Multi-seed learned-policy success rates and final thesis benchmarks.
- GP uncertainty estimates or uncertainty-triggered intervention.
- XR interface, human correction collection or learning from corrections.
- Fine-tuning or policy improvement from corrective demonstrations.

## Next steps

1. Preserve the verified deployment and define an explicit evaluation protocol
   covering seeds, saved initial states and the prediction gate's extra reset.
2. Expand the real-policy baseline only as separately authorized, reporting
   uncertainty intervals and failure cases over multiple episodes.
3. Only after establishing the learned baseline, implement and evaluate the
   uncertainty/intervention/correction stages.
