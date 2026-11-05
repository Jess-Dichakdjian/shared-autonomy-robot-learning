# pi0.5 + GP Human Correction Integration - BEAST Preflight

**Branch:** `feat/vla-gp-human-correction-recording`
**Base:** `feat/human-teleoperation-correction` at `9bac8e101c44c6e6953040322ca64126e8ad2cd4`
**Status:** The integration and automated tests are implemented. A real pretrained preflight reached a safe human-request state. The genuine human correction and post-correction pi0.5 resumption are intentionally awaiting the operator.

## Architecture

`scripts/teleop_shared_autonomy.py` reuses the established camera cockpit, LIBERO task configuration, LeRobot pi0.5 adapter, GP sidecar, and controller bounds. One `SharedWorker` owns the one LIBERO environment and all simulator calls on `libero-shared-autonomy-owner`. HTTP, status, and browser threads do not step MuJoCo. The autonomy loop submits one action at a time through that owner; busy requests are not queued.

The state machine is:

`HUMAN -> AUTONOMY -> ASSISTANCE_REQUESTED -> HUMAN -> AUTONOMY`

- pi0.5 and the passive GP evaluate the same synchronized observation at fresh chunk starts.
- The existing pretrained adapter keeps two 256x256 RGB views, the checkpoint-normalized 8D state, seven normalized relative OSC actions, 50 predicted actions, and 10 executed actions.
- A validation-frozen GP trigger or a manual operator request takes effect at a chunk boundary. The candidate action is not executed, queued pi0.5 actions are discarded, and `policy.reset()` clears LeRobot temporal state.
- Human control requires explicit acceptance. Existing bounded cockpit controls are enabled only in HUMAN mode.
- Returning control clears temporal state again. pi0.5 then infers from the latest post-correction camera/state observation. The first resumed chunk is marked `fresh_post_correction_inference`.
- Emergency stop clears the policy queue and blocks further control.

The manual request is a request path, not an automatic controller action. GP values remain diagnostic signals; they do not replace pi0.5 actions.

## Demonstration records

Each human command continues to appear in the cockpit's `teleoperation.jsonl`. Human actions additionally produce `human_corrections.jsonl` records with:

- Unique run, episode, and intervention identifiers and an explicit `human_keyboard` or `human_button` source.
- LIBERO task, seed, saved-state index/hash, action step, and normalized 7D relative OSC action.
- Before/after robot states and references to both raw RGB cameras, with SHA-256 hashes for each observation.
- Policy checkpoint identity and human controller ownership.

VLA action records preserve raw normalized predictions separately from postprocessed executed actions, chunk ID/index, inference time, state before/after, raw camera file/hash references, GP mean/std and monitoring score at chunk boundaries, ownership, and simulator outcome. The run configuration records package/model/renderer identity. The service records a dual-camera MP4. Run output is under Git-ignored `outputs/`; it is not committed.

## Frozen model and GP

- pi0.5: `lerobot/pi05_libero_finetuned_v044`, revision `7152f13e34c6c115d808aea0e5d08aca74eac14b`, weights SHA-256 `877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae`.
- Inference used CUDA `cuda:0`, BF16, strict checkpoint loading, and the established CPU-resident unused vocabulary output heads. The model state-integrity/placement check ran; one visible RTX 5070 Ti was confirmed.
- The existing CPU GP sidecar used the validation artifact at `outputs/gp_uncertainty_validation_2026-11-04/models.joblib`, SHA-256 `de2ef0deb70764f585563b73eb09cf249322a869a917b293fd544f60b70a185f`.
- Monitoring mode was `gp_uncertainty` with the documented validation-only 90th-percentile threshold (0.8421057488). No threshold or model was changed.
- WSL2 simulation used OSMesa CPU rasterization. The task reset produced both camera observations. EGL was not used.

## Real BEAST preflight

**Scenario:** LIBERO-Spatial task 1, "Pick up the black bowl next to the ramekin and place it on the plate," seed 1101, saved state 0, SHA-256 `d116842e01703fbea595511db813a3816c8d9f2302b4e979ebec4f15b7a750a2`.

- A real CUDA BF16 tensor operation succeeded. GPU free memory before loading was approximately 10.77 GiB.
- The cached checkpoint loaded strictly; the single-GPU placement check found active parameters on CUDA 0 and unused vocabulary heads on CPU. No checkpoint download occurred. The prior two-GPU numerical comparison was not reproduced on this one-GPU host.
- The live service reset the real LIBERO environment and rendered both camera feeds. The model and GP produced 70 VLA action steps over seven chunk boundaries. All predicted/executed action vectors were finite and seven-dimensional; executed actions were in range. GP predictions had finite seven-dimensional means and positive standard deviations at chunk boundaries.
- Recorded per-call inference latency was 0.106 s mean and 4.283 s maximum, including warm-up. The seven GP calls used the CPU sidecar.
- The policy did not complete the task before the requested handoff. At step 70, the GP uncertainty score was 0.8069, below the unchanged 0.8421 threshold. Therefore the handoff was the manual-request path, not a GP-generated trigger.
- The automated preflight HTTP probe issued three repeated manual-request calls while polling a busy simulator. These were programmatic integration checks, not human inputs. The preflight's recorded requests were programmatic and must not be conflated with the later operator-run handoffs. The source code now makes repeated requests idempotent and records request origin; this running service was loaded before that small safeguard was added.
- At the step-70 programmatic preflight checkpoint, the service was waiting in `ASSISTANCE_REQUESTED`; no human correction had yet been recorded. The service remained live, and pi0.5 later terminated this episode successfully at step 112 with zero human-action steps. This is an autonomous task completion, separate from the subsequent operator-run result below.

Raw evidence directory (ignored by Git):

`outputs/vla_gp_human_correction_task1_state0_seed1101_20261105/`

The preflight MP4 was finalized after graceful shutdown and decoded successfully as 113 frames.

## Automated tests

The new tests cover state transitions, chunk-boundary gating, duplicate-request behavior, GP-trigger preemption, VLA action execution/logging, fresh-inference marking, and correction-record source/camera provenance. The existing policy/teleoperation and GP suites are run in their respective pre-existing isolated Python environments. Final verification: 89 passed and 6 skipped across the pi0.5, teleoperation, LIBERO, shared-autonomy, and GP suites. The 18 GP tests emitted 103 scikit-learn kernel-bound convergence warnings on synthetic fixtures; these are optimizer diagnostics, not test failures. Syntax compilation, CLI help, and whitespace checks also passed.

## Remaining gate

The first end-to-end human corrective handoff run has been captured and audited above. Additional episodes are required before making aggregate success or recovery claims. Investigate the measured 8.5 mm displacement for the 2 mm Fine request before treating Fine as precise placement control.


## Verified BEAST operator run

**Run:** `outputs/vla_gp_human_correction_task1_state0_seed1101_retry1_20261105/`
**Scenario:** LIBERO-Spatial task 1, saved state 0, seed 1101, same pinned checkpoint and OSMesa renderer.

- The operator manually requested the first handoff at step 50. The operator then generated 59 human action steps (steps 51-109; 38 button and 21 keyboard actions) and returned control to pi0.5.
- pi0.5 resumed and acted through step 138. At step 139, the passive GP monitor independently requested another handoff at a chunk boundary (`gp_trigger`). This is the run's GP-triggered intervention; the first request was manual.
- After accepting the GP-triggered handoff, the operator made one additional Fine Forward action at step 140 and returned control. The interface requested 2.0 mm and recorded 8.5 mm measured end-effector displacement, so fine preset displacement can overshoot in this contact state.
- pi0.5 then executed 14 action steps (141-154). The episode outcome at step 154 recorded success=true, terminated=true, truncated=false. The final camera view shows the bowl on the plate.
- The run contains 60 human action records in `human_corrections.jsonl` (39 button and 21 keyboard actions), two handoffs (one manual and one GP-triggered), 10 passive GP chunk predictions, and fresh policy inference after both corrections.
- All 310 unique raw camera PNG references in the teleoperation log and all 124 unique before/after camera PNG references in the corrective records were present and matched their recorded pixel-array SHA-256 values. The finalized dual-camera MP4 decoded successfully as 155 frames. Shutdown GPU report recorded 7,126,723,584 peak allocated bytes and 7,291,797,504 peak reserved bytes on the RTX 5070 Ti Laptop GPU.

This is one successful shared-autonomy episode, not a success-rate estimate. The success followed 60 human action steps and cannot be attributed to pi0.5 alone or to the GP trigger alone. Preserve the manual and GP-triggered handoffs as distinct conditions in analysis. The 8.5 mm displacement for a 2 mm Fine request should be investigated before treating Fine requests as precise motion.

The raw run, corrective records, camera PNGs, MP4, and GPU report remain under Git-ignored `outputs/`; they are not included in the repository commit.
