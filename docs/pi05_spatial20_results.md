# LIBERO-Spatial task 0: 20 real pretrained episodes

Verified on LX00060258, 30 October 2026, using the unchanged pinned pi0.5
checkpoint and validated GPU/CPU placement. See
[the protocol and resume instructions](pi05_multi_episode_protocol.md).

| Result | Verified value |
|---|---:|
| Completed episodes | 20 |
| Successful / failed | 20 / 0 |
| Success rate | 100% |
| 95% Wilson interval | 83.89%–100% |
| Saved initial states / seeds | indices 0–19 / seeds 0–19 |
| Distinct saved-state hashes | 20 |
| Evaluated control steps | 1,536 |
| Episode length: mean / range | 76.8 / 74–85 steps |
| Termination reasons | success: 20; timeouts: 0 |
| Fresh chunk predictions / GP pairs | 161 |
| Chunk latency: mean / median / p95 / max | 0.299 / 0.294 / 0.314 / 0.397 s |
| Sampled episode GPU 1 / GPU 0 peak total use | 9,005 / 1,289 MiB |
| Minimum observed free GPU 1 memory | 871 MiB |
| GPU 1 maximum allocator allocated / reserved | 8.063 / 8.473 GiB |

The five-episode pilot completed and passed trajectory, video and GP-export
validation before automatic continuation. Those episodes were not repeated.
The gate reset was overridden by explicit state selection, so evaluated states
are 0–19, rather than the shifted 1–20 from an implicit reset counter.

All complete-episode metrics, trajectories and videos passed validation. The
20 saved-state hashes differ. Representative success video:
`outputs/ucl_pi05_spatial20_2026-10-30/episodes/episode_017/episode_000.mp4`.
There is no representative failure video because no evaluated episode failed.
Per-episode lengths, seeds, states and termination reasons are in the local
`episode_summary.csv`; all raw evidence remains outside Git.

## Dataset and transfer

Persistent experiment root:
`/home/jessica/projects/shared-autonomy-robot-learning/outputs/ucl_pi05_spatial20_2026-10-30`.

- `dataset/gp_chunk_starts.npz`: 161 aligned pairs, `X (161,8)`,
  `X_physical (161,8)`, `Y (161,7)`, `Y_executed (161,7)`,
  complete raw chunks `Y_chunk (161,50,7)`, and identity/timing/outcome arrays.
- `dataset/gp_chunk_starts.csv` and `.jsonl`: portable alternatives.
- `dataset/full_step_trajectories.jsonl`: all 1,536 steps with original
  prediction-state provenance for queued actions and current/next robot state.
- `dataset/schema.json`, `dataset/sha256.json`, `final_audit.json`:
  representations and validation evidence.
- `gp_dataset_bundle.tar.gz`: 2,256,475-byte Lenovo transfer bundle, including
  data, schema, checksums, experiment plan and progress; excludes videos/weights.

Primary `X` is the exact checkpoint-normalized state at each fresh prediction.
`X_physical` retains its pre-normalizer counterpart. `Y` is the raw normalized
horizon-zero VLA action; `Y_executed` is the corresponding applied controller
action. Later chunk actions are not paired with later states as fresh targets.
Independent reconstruction of `X` using the checkpoint's mean/std statistics
had maximum absolute error **0.0**. All dataset hashes and NPZ shapes verified.

Transfer bundle SHA-256:
`c4ef96e62cd72bcde473150feffed443fc45dddc37e0333d176c8622268bfeef`.
The adjacent `.sha256` file can be checked after copying. Use the SCP command
in the protocol document from an authorized Lenovo connection.

## Shutdown and return to UCL

When shutdown was requested, the first process check observed 18 completed
episodes. SIGINT was sent to the identified benchmark and GPU monitor; by the
time it was handled, all 20 completion markers had been written. No new run
was launched after the request. A KeyboardInterrupt during logging cleanup is
retained in the raw run log. Final revalidation confirmed all 20 episodes and
the dataset were complete; no partial episode directories remained. Evaluation
and monitoring processes exited, and GPU memory was released.

There are **no remaining episodes to resume** in this 20-episode batch. On return,
inspect existing results without loading the policy or executing episodes:

```bash
cd /home/jessica/projects/shared-autonomy-robot-learning
source /home/jessica/.venvs/shared-autonomy-robot-learning-pi05/bin/activate
python scripts/report_libero_benchmark.py outputs/ucl_pi05_spatial20_2026-10-30
```

For a genuinely interrupted future run, repeat the exact benchmark command in
the protocol document: validated completion markers are verified and skipped,
and only incomplete/unstarted episodes are evaluated. The current completed
batch is detected before model loading and cannot run extra episodes through
that same command. A different experiment requires a new output path/plan.

## Checks and limitations

Final lightweight suite: 22 passed, 3 standard opt-in tests skipped. Before the
benchmark, the real simulator tests passed 2/2 and the cached real processor test
passed 1/1, including explicit saved-state selection and exact state capture.
Recovery, protocol mismatch, evidence corruption, queue reset, and erroneous
later-state pairing are covered by tests. No dependencies changed.

The [Wilson interval](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
describes this small fixed single-task batch; it is not a cross-task guarantee.
All episodes succeeded, so this dataset cannot by itself validate uncertainty
as a discriminator of task failures. State-only GP inputs omit visual context
and trajectories are correlated. Next: define episode-held-out splits and a
state-to-raw-action GP regression baseline, then assess predictive error and
interval coverage. No GP training, fine-tuning or Ego2Robot changes occurred.
