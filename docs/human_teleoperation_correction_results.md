# Human teleoperation correction interface: Gates A–D

## Scope and status

This change improves the existing human-only LIBERO teleoperation instrument on BEAST, on branch `feat/human-teleoperation-correction`. The VLA policy and its observations/actions are not changed. No pretrained policy, GP, XR device, intervention trigger, or learned correction is active. Automated movements reported here are controller validation, not human demonstrations or task-success evidence.

The test scenario is LIBERO-Spatial task 1: “pick up the black bowl next to the ramekin and place it on the plate,” saved initial state 0, seed 1101. The saved-state SHA-256 is `d116842e01703fbea595511db813a3816c8d9f2302b4e979ebec4f15b7a750a2`. Rendering uses the already-installed OSMesa path and a dedicated simulation-owner thread.

## Camera orientation and coordinates

Camera directions are computed from live MuJoCo local-to-world extrinsic rotations, then projected onto the tabletop plane. For each camera, local +X is image-right and local +Y is image-up. At the tested initial state, scene image-right aligns approximately with world +Y and image-up with world −X; wrist image-right aligns approximately with world −Y and image-up with world +X. The views are physically oriented in opposite yaw directions; neither canonical image is mirrored or rotated by the interface.

The control reference selector offers Scene camera, Wrist camera, and World. World is +X forward, +Y right, +Z up. Scene/wrist forward and left/right use the selected camera's extrinsic tabletop basis. Height commands always use world ±Z. A camera-axis overlay shows the current projected image-right and image-up directions in world coordinates. These transformations affect teleoperation commands only. Raw simulator pixels remain the canonical observations and are written as lossless PNGs; the browser's JPEG preview and MP4 are display/evidence formats only.

## Controls and bounded motion

- W/S: forward/backward in the selected frame; A/D: left/right; E/Q: up/down.
- 1/2, 3/4, 5/6: positive/negative world axis-angle wrist rotation on axes 0, 1, 2.
- V/G/H: open/close/hold gripper; Pause, Resume, and Emergency Stop have explicit controls.
- Fine requests 2 mm, Normal 5 mm, and Coarse 30 mm. Requests are Cartesian targets; measured motion can differ because of controller response, contacts, and stopping tolerance.
- A command is bounded to at most 7/10/18 simulator control steps for Fine/Normal/Coarse and 2.2 seconds. Fine/Normal/Coarse tolerances are 0.35/0.55/1.2 mm. The underlying OSC per-axis translation limit remains 50 mm per step, and the interface never queues commands.
- Holding a direction repeats one segment every 250 ms. Each held input has a 0.75-second heartbeat lease and a five-second hard limit. Key/button release, browser blur, failed command/heartbeat, or lease expiry stops the sequence; server-side lease checks stop an in-progress multi-step command between simulator steps.
- Open/close each run up to 20 control steps, stopping once the measured gripper aperture reaches its threshold or ceases changing; Hold sends one step.

The fixed Robosuite OSC_POSE controller is delta-based with normalized inputs in [-1, 1], ±0.05 m translation per axis, and ±0.5 rad rotation-vector components. Panda gripper input −1 opens, +1 closes, and 0 holds.

## Validation results

The teleoperation unit suite passed: `20 passed` on BEAST using the pinned Python 3.12 environment.

The real LIBERO/OSMesa self-test ran from the pinned task/state on the dedicated `libero-simulator-owner` thread. It made 11 automated direction checks (World forward/right/up, Scene forward/backward/right/left, Wrist forward/backward/right/left); measured-direction alignment cosines ranged from 0.98247 to 0.99997. These checks validate the extrinsic mapping at this state; they are not human inputs.

Measured end-effector travel in the same self-test was:

| Preset | Requested | Measured |
| --- | ---: | ---: |
| Fine | 2 mm | 1.576 mm |
| Normal | 5 mm | 6.536 mm |
| Coarse | 30 mm | 29.500 mm |

Fine < Normal < Coarse held. Normal overshot its request by about 1.54 mm (31%); the UI and log explicitly distinguish requested from measured displacement. These figures are one automated calibration from the fixed saved state, not an accuracy guarantee across poses or contacts.

The gripper aperture proxy changed from 0.079973 open to 0.005579 closed and back to 0.076006 open over the bounded multi-step checks. This validates actuator response only; no object was grasped or lifted in the self-test.

A previous manual browser run with 135 user-input records and its video is preserved in `outputs/human_teleop_task1_state0_seed1101_retry1/`. It is not overwritten by this server restart, and it is not claimed as a successful grasp/task. The earlier camera-corruption attempt is also preserved in `outputs/human_teleop_task1_state0_seed1101/`.

## Research logging

A new manual attempt uses a fresh output directory, `outputs/human_teleop_task1_state0_seed1101_retry2/`; the server refuses to start if that directory already exists. Each action record contains timestamps, task/run/seed/state provenance, keyboard/button input source, requested command, actual 7D executed OSC action, controller substep, before/after robot state, measured end-effector delta, reward and termination/success flags, plus the live camera extrinsics. Raw synchronized scene/wrist camera frames are saved losslessly with hashes. The MP4 is a side-by-side viewing copy. Runtime recordings, server logs and PID files are ignored by Git and must not be committed.

## Manual validation required

The server is bound only to BEAST loopback at `127.0.0.1:8765`; access is through an SSH local tunnel from the Lenovo:

```powershell
ssh -N -L 8765:127.0.0.1:8765 friend-gpu
```

Keep the tunnel terminal open and refresh `http://127.0.0.1:8765`. Start with Fine, Scene-camera reference, and the on-screen calibration checklist. Test each direction in open space, inspect the measured vector and both images, then test gripper open/close. Only approach the bowl if those checks look right. A grasp is verified only if the bowl visibly rises and stays held. Pause or use Emergency Stop whenever control is unclear. Close the browser and end the SSH tunnel with Ctrl+C when finished.

The next dependency is the user's manual validation of the refreshed controls and frames. After that, inspect the new human-action records/video and verify a real grasp before implementing the separate VLA → human correction → VLA resumption sequence.
