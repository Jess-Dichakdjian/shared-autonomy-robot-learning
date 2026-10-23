# Thesis evidence index

**Updated:** 23 October 2026
**Canonical repository:** shared-autonomy-robot-learning
**Active branch at capture:** feat/pi05-libero-foundation

This index distinguishes the current simulation and processor evidence from the separately reported historical pretrained-model run.

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
