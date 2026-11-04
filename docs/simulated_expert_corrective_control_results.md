# Simulated Expert Corrective Control: Feasibility Gate Results

Date: 4 November 2026
Branch: `feat/simulated-expert-corrective-control`
Base: `62232532d9740f1dfeea047a50bae130b32a55f1`

## Decision

The expert-feasibility gate did not pass. A small privileged Cartesian pick/place controller could move the robot to the target region, but no bounded trial lifted the bowl. Therefore this branch does **not** implement or claim a working π0.5-to-expert handoff, corrective demonstration, recovery, or assisted-control result. No unreliable provider was attached to the VLA evaluator, and no controlled comparison was run.

This is a task/controller feasibility result, not evidence that simulated expert correction is impossible in LIBERO. The exact contact or grasp failure remains undiagnosed. The fastest valid next route is a human-operated 7D correction through a live simulator view, or a separately validated task-specific expert. The current BEAST experiment used headless OSMesa; a live display and input path must be established before keyboard/gamepad teleoperation can be recorded.

## Preserved setup and baseline

The worktree was created from the verified GP branch commit `62232532d9740f1dfeea047a50bae130b32a55f1` at `/home/jess/projects/shared-autonomy-robot-learning-sim-expert`. The previous GP uncertainty worktree, integration worktree, task-0 benchmark artifacts, pretrained π0.5 weights, and Ego2Robot checkout were not modified.

The frozen π0.5 + GP monitor baseline was rerun for LIBERO-Spatial task 4, saved initial state 4, seed 1204. The saved-state SHA-256 was `eacc257cf75be38a89bdf75e5300cded0aee0ec1d0a2ac4b91d4c4cdb11c763b`. The strict pretrained checkpoint/GPU/action gate passed. The episode **succeeded in 142 steps**, with 15 chunk predictions, mean/p95 chunk latency 0.345/0.410 s, and allocator peak 7,125,786,624 bytes allocated / 7,291,797,504 bytes reserved. Thus the previous 280-step timeout on this saved state did not reproduce in this run.

The validation-frozen GP uncertainty trigger requested assistance at step 120 (score 0.8743, threshold 0.8421). It was request-only; π0.5 kept control and succeeded 22 steps later. This late request in a successful episode is not evidence that a correction was needed. It is also a poor first handoff point for proving recovery. No threshold was retuned against this rollout.

## Expert feasibility investigation

No existing scripted task expert or human demonstration controller was found in the thesis repository. Installed robosuite exposes OSC pose control (`OSC_POSE`) and the LIBERO environment exposes the current task’s `obj_of_interest`, simulator object bodies, and end-effector site. The tested prototype used these current privileged poses to target the task’s object and plate. This would be an **oracle-assisted simulated controller**, not a human demonstration. It did not read future states or teleport the robot.

The prototype issued finite seven-dimensional normalized relative OSC actions within `[-1, 1]`; robosuite’s OSC pose translation scale is 0.05 m per unit. The Panda gripper interface documents -1 as open and +1 as closed. The controller approached, descended, closed, and attempted to lift. A first implementation reopened during lift; that bug was corrected, and later trials held the gripper closed through lift. Even so, the object did not rise by the predefined 3.5 cm grasp-confirmation threshold.

Task-4 attempts used the same saved state and seed. The first 90-step run stalled near the bowl. A retry with a bounded close-on-stall attempt reached a transfer phase but did not move the bowl. A corrected-gripper retry used up to 60 close actions and a 240-step maximum; it still did not lift the bowl. The final bounded retry held close for 120 actions, with a 220-step maximum and stop-after-lift criterion; it also did not lift the bowl. The robot reached an end-effector position near the bowl, but the object remained at approximately z=1.063 m. These runs are feasibility attempts, not successful corrections.

One task-1 tabletop attempt with the initial controller also failed before the gripper-retention bug was fixed. A final bounded task-1 attempt with corrected gripper retention and 120 close actions also failed to lift the object. The controller is therefore not validated even on the simpler tabletop case. We stopped rather than generating more variants or implying successful grasping from end-effector motion alone.

### What was and was not observed

- Confirmed: the simulator can expose current object and end-effector poses; task object/destination names resolve; the robot accepts seven-dimensional OSC actions; videos and step/action traces were written for each attempt.
- Confirmed: the expert moved the end effector to the target vicinity and produced bounded actions.
- Not achieved: object lift, placement, task subgoal, VLA pause, VLA resumption after correction, or end-to-end assistance episode.
- Not diagnosed: whether the missing lift is caused by contact geometry, target height/offset, gripper configuration/actuator dynamics, or the approach path. Do not call it a LIBERO limitation.
- No simulated oracle success, future action replay, or human/XR correction was claimed.

## Evidence locations

All raw attempts are separate and Git-ignored under:

- VLA/GP task-4 baseline: `outputs/simulated_expert_baseline_task4_state4_seed1204/`
- Initial bounded task-4 attempt: `outputs/simulated_expert_feasibility_task4_state4_seed1204/`
- Task-4 gripper-retention retry: `outputs/simulated_expert_feasibility_retry2_task4_state4_seed1204/`
- Task-4 120-action-close retry: `outputs/simulated_expert_feasibility_retry3_task4_state4_seed1204/`
- Initial task-1 attempt: `outputs/simulated_expert_feasibility_task1_state0_seed1101/`
- Corrected task-1 retry: `outputs/simulated_expert_feasibility_retry1_task1_state0_seed1101/`

Each feasibility directory contains `summary.json`, `expert_steps.jsonl`, and `expert_feasibility.mp4`. The baseline directory contains episode trajectory, action/GP logs, metrics, and video. Videos/models/raw outputs are excluded from Git. This report should be committed alone with a checksum manifest; raw evidence remains on BEAST and in the separate evidence archive.

## Requirements before continuing

1. Provide a working live view and keyboard/gamepad input route to the BEAST LIBERO environment (or approve switching this feasibility experiment to a suitable interactive host). Headless OSMesa alone cannot support a genuine human correction.
2. Validate a human correction source independently, with source labels distinct from simulated-oracle actions.
3. At a later live GP request, pause at the chunk boundary, clear queued π0.5 actions, record the correction, refresh the observation, reset policy temporal state, and confirm π0.5 produces a new action from the post-correction camera/state.
4. Do not run a comparison or claim benefit until at least one actual correction and post-correction VLA resumption are verified.

## Status

| Requirement | Status |
|---|---|
| Frozen π0.5 + live GP baseline on target state | Verified; success, 142 steps |
| Reproduce prior task-4 timeout | Not reproduced |
| Valid expert source | Prototype only; no lift, not validated |
| Assistance request | Verified at step 120; request only |
| Safe handoff / stale chunk invalidation | Not implemented in this branch |
| Actual corrective control | Not achieved |
| Correction trajectory dataset | Not produced; feasibility action traces are not demonstrations |
| π0.5 resumption after correction | Not tested |
| Controlled comparison | Not feasible at this gate |
| Keyboard/gamepad / XR | Not implemented; live input/display path required |

No new code tests were run in this branch because no controller or evaluator implementation was added. The preceding GP/VLA test results remain those documented in the base report.
