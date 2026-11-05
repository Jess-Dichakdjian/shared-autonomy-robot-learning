# LIBERO human teleoperation cockpit and validation

## Scope

The human-only LIBERO-Spatial interface is maintained on `feat/human-teleoperation-correction`. It is a research control instrument for human corrective demonstrations. No VLA, GP model, assistance trigger, XR device, or autonomous handoff is active. The interface does not change the pretrained policy, simulator physics, OSC controller, or canonical camera observations.

Scenario: LIBERO-Spatial task 1, “Pick up the black bowl next to the ramekin and place it on the plate,” saved initial state 0, seed 1101. The pinned state SHA-256 is `d116842e01703fbea595511db813a3816c8d9f2302b4e979ebec4f15b7a750a2`. The private BEAST server binds only to `127.0.0.1:8765` and is reached through an SSH local tunnel.

## Cockpit layout

The page now uses a compact desktop grid: a roughly 65% dual-camera workspace and 35% human-control workspace, with a slim full-width simulation status bar. The scene and wrist feeds remain square and side by side. Movement, gripper, pause/resume and emergency-stop controls sit beside them. Wrist rotation controls and detailed state/recording diagnostics are collapsed by default. Narrow screens switch to a vertically stacked, scrollable layout.

The status bar reports HUMAN/PAUSED state, task, episode, step, recording, connection and assistance status. It states that no VLA is connected; π0.5 or GP is not presented as active. Compact feedback shows end-effector position, gripper aperture/status and requested versus measured movement.

## Display orientation and control mapping

The Scene camera (`agentview`) stays in its original orientation. The Wrist camera (`robot0_eye_in_hand`) defaults to a human-facing 180-degree CSS rotation and has Original/180° toggles. The source arrays and image bytes are not rotated, mirrored, cropped or resampled before canonical PNG storage or future policy input. The square source aspect ratio is preserved with `object-fit: contain`; CSS rotation is presentation-only.

Control references remain Scene, Wrist and World. Live MuJoCo camera-to-world extrinsics are recalculated after every simulator step, so a moved/rotated wrist camera changes subsequent wrist-relative mapping. When the wrist display is rotated 180 degrees and Wrist is the selected control reference, the planar Forward/Backward and Left/Right axes are each negated once to match what the operator sees. Scene and World controls ignore the wrist display setting, and Up/Down always use world ±Z. The selected reference and wrist display angle are visibly labelled.

The active control reference and wrist display angle are written into configuration metadata, operator-view change records and every action-step record, together with camera extrinsics. This lets later analysis reconstruct the view and reference used for each action. Canonical raw frame hashes and arrays remain independent of the display choice.

## Controls and safety

- W/S: Forward/Backward; A/D: Left/Right; E/Q: Up/Down.
- V/G/H: Open/Close/Hold gripper.
- 1/2, 3/4, 5/6: positive/negative world axis-angle rotation components.
- Fine / Normal / Coarse request 2 / 5 / 30 mm. A request is a target; measured displacement is reported separately.
- A movement command is bounded to 7 / 10 / 18 simulator steps and 2.2 seconds. Held input repeats one segment at a time, with a 0.75-second heartbeat lease and five-second hard limit. Release, blur, lost heartbeat, command failure, pause or emergency stop cancels movement; commands are not queued.
- Open/Close run at most 20 simulator control steps and stop when the aperture reaches its limit or stalls. Hold sends a single step.

The server-side action limits, stale-command/epoch checks, controller-owner thread and emergency-stop transitions are retained. The browser shows the actual HUMAN or PAUSED owner state. Gripper motion is not evidence of a grasp; a grasp is verified only by seeing the object lift and remain held.

## Automated validation on BEAST

The teleoperation unit suite passed: **27 passed**. It covers camera-basis projection, Original/180° wrist signs, world and scene invariance, remapping after changed camera extrinsics, no double inversion, view metadata, 7D action/state/camera-hash records, byte-exact canonical PNG output, heartbeat expiry and emergency-stop cancellation.

The LIBERO/OSMesa calibration self-test passed on the dedicated simulator-owner thread, using a temporary synthetic run. It checked 11 initial world/scene/wrist directions, both wrist display orientations, and wrist-relative motion after an actual 1° end-effector rotation changed the live camera extrinsics. Measured direction-alignment cosine across the checks was at least 0.9824. In the orientation-specific checks:

- Wrist display Original, Forward: measured approximately `[+1.530, -0.008, -0.252]` mm, cosine 0.9867.
- Wrist display 180°, Forward: measured approximately `[-1.453, -0.017, +0.275]` mm, cosine 0.9825.
- After tool rotation, Wrist display 180°, Right: measured approximately `[-0.046, +1.680, -0.003]` mm, cosine 0.9997. The live camera-right vector changed from approximately `[-0.00005, -1.00000, 0]` to `[+0.00195, -1.00000, 0]`; the movement used the updated extrinsics.

Fine/Normal/Coarse travel remained ordered and measured approximately 1.563 / 6.538 / 29.514 mm for requests of 2 / 5 / 30 mm. Normal still overshoots by about 1.54 mm in this pinned state, so use Fine near an object. Gripper aperture proxy changed from approximately 0.07999 open to 0.00558 closed and back to 0.07601 open. This was actuator validation, not an object grasp or human demonstration. The temporary integration log checked 146 action-step records for seven-dimensional actions, operator-view metadata and hashes for both raw camera frames.

## Viewport verification

The running browser was visually and programmatically checked at both target sizes after loading live camera frames:

| Browser viewport | Scene feed | Wrist feed | Control workspace | Document / scroll |
| --- | --- | --- | --- | --- |
| 1366 × 768 | 425 × 425 px | 425 × 425 px | x=889, y=62, 470 × 699 px; Emergency Stop visible | 1366 × 768; `scrollY=0` |
| 1920 × 1080 | 601 × 601 px | 601 × 601 px | x=1249, y=74, 661 × 996 px; Emergency Stop visible | 1920 × 1080; `scrollY=0` |

Both screenshots showed the two feeds and primary controls simultaneously. The viewport override used for the checks was reset afterward.

## Recordings and preservation

Earlier human recordings remain untouched in:

- `outputs/human_teleop_task1_state0_seed1101/`
- `outputs/human_teleop_task1_state0_seed1101_retry1/` (135 JSONL records and its video)
- `outputs/human_teleop_task1_state0_seed1101_retry2/`

The current service writes to the separate `outputs/human_teleop_task1_state0_seed1101_retry3/`. Runtime outputs, raw frames, video and logs are ignored by Git. No human inputs were fabricated, and automated checks use temporary output folders.

## Next manual test

Keep or open the Lenovo SSH tunnel:

```powershell
ssh -N -L 8765:127.0.0.1:8765 friend-gpu
```

Then refresh `http://127.0.0.1:8765`. Confirm both images are clear, the status says HUMAN and the Wrist view says 180° display. Start the Fine calibration checklist and briefly test W/S/A/D/E/Q in open space. Compare actual movement with the selected control reference; repeat with Wrist selected only after the Scene test is clear. Toggle the Wrist view to Original and back to 180° while no motion is active to see the presentation change. Test the gripper only after confirming controls, and count a grasp only if the bowl visibly rises and remains held. Report camera clarity, direction feedback, gripper behavior and whether a real lift was verified.

Do not start VLA handoff, GP retraining, XR development or additional model experiments until the user has completed this interface check.
