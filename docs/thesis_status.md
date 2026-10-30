# Thesis implementation status

**Last updated:** 30 October 2026
**Canonical repository:** shared-autonomy-robot-learning
**Active branch:** codex/ucl-pi05-evaluation

## Current status

The next milestone is now complete: 20 real pretrained task-0 episodes, with
20 successes and no failures (95% Wilson interval 83.89%–100%). Explicit saved
initial-state indices and seeds cover 0–19. There are 161 GP-ready chunk-start
pairs and 1,536 step records, with validated videos, timing and memory evidence.
The benchmark and monitor have exited; no episodes remain in this batch.
See pi05_spatial20_results.md and pi05_multi_episode_protocol.md. GP training
has not started. Final suite: 22 passed / 3 skipped; explicit integrations passed.

The earlier feasibility milestone verified real inference and one successful
LIBERO-Spatial task-0 episode. The same pinned, hash-verified checkpoint runs
with EGL on GPU 0,
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
| Evaluator | 20 successful episodes; state/action/config JSONL, chunk-start GP export, metrics, latency, memory and synchronized video | Single task, explicit seeds/states 0–19; no failure examples |
| Mock result | LIBERO Spatial task 0, seed 123, three steps, OSMesa, mock zero-action policy; output copied to experiments/libero_mock_policy_smoke_2026-10-17/ | Intentionally capped; success 0.0, return 0.0; not a full episode or learned result |
| Tests | Final suite 22 passed / 3 skipped; explicit simulator tests 2 passed and processor integration 1 passed | GPU placement tests use the available two-GPU host |
| UCL real pretrained result | 20/20 successes; mean length 76.8; 161 prediction points; mean chunk latency 0.299 s; sampled GPU 1 peak 9,005 MiB | Fixed single-task batch; initial OOM and first feasibility result remain historical |

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

- Broader multi-task benchmarks, failure-case uncertainty validation and final thesis evaluation.
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
