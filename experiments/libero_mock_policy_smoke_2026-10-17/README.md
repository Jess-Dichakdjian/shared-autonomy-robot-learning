# LIBERO Spatial task 0 mock-policy pipeline check

This is a copy of the generated run in outputs/mock_libero_smoke/, retained
here as a small, shareable reproducibility artifact. The original output folder
was not moved or changed.

- Date: 17 October 2026
- Suite/task: LIBERO Spatial, task 0
- Seed: 123
- Policy: mock_zero_chunk (not a learned or pretrained policy)
- Renderer: OSMesa
- Duration: 3 environment steps, intentionally capped; this is not a full episode
- Observations: synchronized 256x256 agent-view and wrist-camera RGB images
- Logged: predicted/executed actions, robot state, rewards, episode metadata,
  configuration, seed and video
- Outcome: not successful; observed success rate 0.0 and return 0.0

episode_000.mp4 demonstrates the simulator and recording path only. It must not
be described as pretrained pi0.5 behavior. The JSONL trajectory and metrics
files preserve the associated states, actions and configuration. See
docs/vla_foundation.md for the policy/action contract and remaining work.
