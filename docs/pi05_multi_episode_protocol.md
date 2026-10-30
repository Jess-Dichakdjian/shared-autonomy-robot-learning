# Sequential pretrained evaluation and GP dataset protocol

This extends the validated [memory placement](pi05_memory_placement.md) without
changing checkpoint weights, mixed precision, normalization, cameras, action
postprocessing, ten-step execution chunks or GPU placement. EGL uses physical
GPU 0, active inference uses GPU 1, and unused vocabulary heads remain in RAM.
No model training or GP fitting is part of this milestone.

## Evaluation plan

The experiment plans 20 sequential episodes of LIBERO-Spatial task 0, with a
five-episode technical pilot. Episode `i` uses policy/simulator seed `i` and
saved initial-state index `i`, for `i=0..19`. The runner explicitly sets the
pinned LIBERO wrapper's saved-state counter before **every** reset, verifies
the requested index, and records the saved-state SHA-256 and initial camera
hashes. All saved-state hashes must differ. Seeds alone are not treated as
evidence of different scenes.

The separate prediction gate resets once, but its implicit index advancement
is overridden by the explicit episode plan. Thus the gate does not shift the
evaluated scenes to indices 1..20. The gate is outside the evaluated sample.
This differs from the earlier one-episode feasibility protocol, where the gate
consumed state 0 and the episode used state 1.

Each episode resets policy temporal state and discards its action queue. The
standard LIBERO-Spatial task limit is 280 control steps. Success, environment
termination and time limit are distinguished; a separate evaluator cap of 280
prevents an unbounded loop. Task failures/timeouts are valid data. Inference,
memory, logging, unknown outcomes or validation errors stop the experiment.

Per episode, validation checks action shape/finiteness/range, state and chunk
alignment, step count and outcome consistency, and every frame of the complete
dual-camera video. At five completed episodes, the GP export is also validated
before continuing to 20. The completed pilot episodes are retained and skipped
on continuation. The original feasibility episode is not part of this batch.

## Recovery and evidence

`experiment.json` is an immutable plan with checkpoint identity, processor
hashes, source hashes, seed/state rules and model placement. `progress.json`
is replaced atomically and fsynced after updates. Per-episode `complete.json`
is written only after validation and hashes the metrics, trajectory and video.
Progress is filesystem metadata, not a policy/optimizer checkpoint.

Restarting the same command verifies completed-file hashes and skips those
episodes. A partially written episode is archived under `interrupted_attempts/`
and restarted from its explicit initial index and seed. Corrupted committed
evidence or a changed experiment protocol is rejected. File locks prevent two
writers. A crash between completion-marker and progress updates is recovered
by scanning validated completion markers. No incomplete attempts enter exports.

Separate directories under `episodes/episode_NNN/` contain steps, episode
metadata, metrics, video and completion metadata. Model sessions/gates are
stored separately. Both-GPU sampling and per-episode PyTorch allocator peaks
are retained. Chunk-generating latency is summarized separately from inexpensive
queued-action retrieval.

## GP representations and pairing

The saved processor supplies the eight-value physical state: end-effector
position xyz, quaternion converted by the official LIBERO processor to axis-
angle xyz, and the two gripper joint positions. Both that vector and the exact
checkpoint-normalized `observation.state` are copied at fresh prediction time.
The unchanged pi0.5 processor also derives state tokens from it.

Primary GP arrays:

- `X`: `(N,8)` exact checkpoint-normalized state at chunk starts.
- `X_physical`: `(N,8)` corresponding pre-normalization state.
- `Y`: `(N,7)` raw normalized pi0.5 action at horizon zero of each fresh chunk.
- `Y_executed`: `(N,7)` corresponding actually applied action after the existing
  unnormalization, LIBERO postprocessing and clipping into relative OSC control.
- `Y_chunk`: `(N,50,7)` complete raw normalized predicted chunks for optional
  future horizon-aware analysis. These are not 50 independent input states.

STATE and ACTION use the checkpoint's saved MEAN_STD statistics. No new
normalization is fitted. Raw `Y` is not an already unnormalized robot command.
Executed OSC values are the existing controller interface, not seven joint
targets or uniformly SI-valued coordinates.

One primary GP row is exported **only at each fresh prediction**, pairing its
state with horizon-zero `Y`. A queued action at a later step retains the original
prediction state, prediction step/time, chunk ID and horizon in the full step
dataset; it is never relabelled as a fresh prediction from the later robot state.
Current/next raw robot states remain in the trajectory for separate analysis.
Episode success/failure is joined after the episode completes.

The NPZ uses numeric arrays and can be loaded with `allow_pickle=False`.
JSONL/CSV alternatives, schema, episode outcomes and checksums accompany it.
Unix wall timestamps are UTC seconds; elapsed and inference times are seconds.

## Commands and transfer

From the canonical repository:

```bash
source ~/.venvs/shared-autonomy-robot-learning-pi05/bin/activate
CUDA_VISIBLE_DEVICES=0,1 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=. \
python scripts/benchmark_libero_pi05.py \
  --output outputs/ucl_pi05_spatial20_2026-10-30 --total 20 --pilot 5 \
  --base-seed 0 --first-state 0
```

Use this exact command to recover interrupted work. Use a **new** output path
for a different plan; the runner rejects protocol changes under an existing path.
`--stop-after 5` is available for an explicit pilot-only run, with the same total
plan retained for later continuation. The authorized experiment continues
automatically without that option.

Evidence-only reporting/export needs no checkpoint load:

```bash
python scripts/export_gp_dataset.py outputs/ucl_pi05_spatial20_2026-10-30
python scripts/report_libero_benchmark.py outputs/ucl_pi05_spatial20_2026-10-30
```

Persistent UCL root:
`/home/jessica/projects/shared-autonomy-robot-learning/outputs/ucl_pi05_spatial20_2026-10-30`.
Portable dataset: `gp_dataset_bundle.tar.gz` under that root. It contains the
GP matrices, complete step dataset, schema, checksums, plan and progress, with
no checkpoint weights or videos. From a Lenovo terminal with authorized SSH
access to the UCL workstation, copy it into the current local directory:

```bash
scp jessica@LX00060258:/home/jessica/projects/shared-autonomy-robot-learning/outputs/ucl_pi05_spatial20_2026-10-30/gp_dataset_bundle.tar.gz .
```

Use the university's reachable hostname/SSH route if the workstation name is
not resolvable from that network. No transfer to Lenovo is performed by this run.

## Interpretation and next GP step

The success interval is a 95% Wilson binomial interval, following the
[NIST proportion-interval formulation](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
This is a small, fixed single-task scene/seed batch, not a random sample of
all deployment settings. The interval does not establish cross-task robustness.

Robot state alone omits the visual scene and task context that condition the
VLA. Similar states can therefore have different action targets. GP predictive
uncertainty on this dataset must not be presented automatically as calibrated
VLA epistemic uncertainty or an OOD detector. Within-episode rows are correlated.

The next implementation step is an **episode-held-out** GP regression baseline:
fix a train/validation split by episode, use `X` and `Y` with explicit action
representation, and assess predictive error, interval coverage and uncertainty
versus input coverage. Any additional preprocessing must be fitted on training
episodes only. No GP model is trained in this milestone.
