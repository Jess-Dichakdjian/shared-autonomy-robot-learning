# Thesis evidence index

**Updated:** 30 October 2026
**Canonical repository:** shared-autonomy-robot-learning
**Active branch at capture:** feat/pi05-libero-foundation

This index distinguishes the current simulation and processor evidence from the separately reported historical pretrained-model run.

## Real pretrained UCL result - 30 October 2026

See [pi05_memory_placement.md](pi05_memory_placement.md). Unused vocabulary
heads remain in RAM, active inference runs on GPU 1, and EGL renders on GPU 0.
The checkpoint, precision and saved normalization are unchanged.

- 18 unit/placement tests passed, 2 standard opt-in tests skipped. The earlier
  explicit processor and simulator integrations remain separate passing evidence.
- Camera hashes match across EGL device placement; renderer costs about 727 MiB.
- First real prediction: 0.561 s. Full `(1,50,7)` chunk and executed first action
  were bitwise equal with unused heads on CPU versus temporarily on GPU 0.
- One complete episode: success, return 1.0, 78 steps, eight chunk predictions,
  seed 0. Gate used saved initial state 0; episode used saved initial state 1.
- Mean/median chunk latency: 0.292/0.289 s. Episode sampled peak total GPU 1/GPU 0
  memory: 9,035/1,288 MiB. PyTorch GPU 1 peak allocated/reserved: 8.063/8.473 GiB.
- All trajectory rows and all 79 dual-camera video frames were validated.

Local raw evidence is ignored by Git at `outputs/ucl_pi05_cpu_heads_2026-10-30/`
and `outputs/ucl_placement_2026-10-30/`. This is one real feasibility result,
not a multi-seed success-rate estimate or shared-autonomy result.

## Initial UCL workstation verification - 30 October 2026

See [ucl_workstation_setup.md](ucl_workstation_setup.md) for the environment,
commands, pinned revisions, measured memory and offloading investigation.

- Python 3.12, CUDA PyTorch 2.10.0+cu126, exact LeRobot source pin verified;
  package consistency and small GPU FP32/BF16 checks passed.
- Unit suite: 13 passed / 2 skipped. Each opt-in integration passed explicitly.
- LIBERO Spatial task 0 passed NVIDIA EGL dual-camera/action smoke checks.
- Five-step mock rollout, seed 123: state/action JSONL, metrics and decoded
  512x256 video verified; success 0.0 and return 0.0 are mock-only evidence.
- The v044 checkpoint was downloaded to persistent user storage and all LFS
  hashes were verified against the pinned revision.
- Strict pretrained weight loading succeeded. CUDA OOM occurred transferring
  the first observation, before prediction or any evaluated environment step.
  There is no real-policy success score, latency, trajectory or video.
- Peak PyTorch allocated/reserved memory: 9,356,805,120 / 9,594,470,400 bytes.
  Sampled total physical GPU memory peaked at 9,874 MiB, including rendering.
  GPU use returned to 49 MiB after process exit.

Local evidence (intentionally ignored by Git):

- `outputs/ucl_setup_2026-10-30/`: dependency lock/logs, tests, preflight,
  checkpoint identity/integrity, shape audit, real-attempt traceback and GPU
  samples/summary, plus CPU primitive probe.
- `outputs/ucl_mock_spatial_task0_2026-10-30/`: successful mock pipeline artifacts.
- `outputs/ucl_pi05_spatial_task0_first_2026-10-30/`: failed real attempt's GPU
  memory record and empty trajectory/episode files. Empty files are not a
  rollout result. No metrics or video were produced.

## Curated, versioned experiment

The small reproducible mock run is versioned at:

- experiments/libero_mock_policy_smoke_2026-10-17/README.md
- experiments/libero_mock_policy_smoke_2026-10-17/run_config.json
- experiments/libero_mock_policy_smoke_2026-10-17/steps.jsonl
- experiments/libero_mock_policy_smoke_2026-10-17/episodes.jsonl
- experiments/libero_mock_policy_smoke_2026-10-17/metrics.json
- experiments/libero_mock_policy_smoke_2026-10-17/episode_000.mp4

It used LIBERO Spatial task 0, seed 123, OSMesa, and the mock_zero_chunk policy for three steps. The task was not successful (success rate 0.0; return 0.0). This demonstrates the simulator/evaluator/logging/video path only. It is not a pretrained pi0.5 result.

## Windows evidence copy

The separate Windows folder is C:/Users/jessy/Documents/Thesis_Evidence. It contains a copied mock rollout, a small supervisor-demo copy, a historical-run recovery note, and fresh test logs under Verification/. The copies do not replace or modify the repository artifacts.

Fresh verification on 23 October 2026:

- Unit suite: 10 passed, 2 skipped.
- LIBERO OSMesa integration: 1 passed; three robosuite deprecation warnings.
- Authenticated LeRobot pi0.5 processor integration: 1 passed.
- No checkpoint-backed policy load or pretrained inference occurred.

## Historical HP OMEN run

The user reports an earlier successful pi0.5 run on an HP OMEN with an RTX 2060 and 6 GB VRAM. The original video, metrics, command and logs were not found in accessible Windows or WSL locations. The report is therefore preserved as user-reported historical context, not independently verified evidence. No unrelated Ego2Robot videos or new mock video are presented as that result.

The pinned checkpoint is cached at ~/models/pi05_libero_finetuned_v044 on the Lenovo outside Git, but the weights remain unloaded. The Lenovo CPU preflight stopped before model loading because available WSL RAM was below the conservative headroom heuristic. See docs/thesis_status.md and docs/vla_foundation.md.
