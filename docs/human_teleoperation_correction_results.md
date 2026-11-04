# Human teleoperation correction — Gates A–C

## Status and scope

Gates A–C are implemented on BEAST's existing pinned Python 3.12 environment.
The user's first browser attempt sent 16 human commands, then both camera frames
became corrupted. Those commands and the video are preserved, but the visual
failure means no grasp or task outcome is validated. No VLA, GP model, learned
policy, or expert controller was used, and no task success is claimed. The
thread-affinity fix below is ready for a fresh manual-control attempt.

The interface runs LIBERO-Spatial task 1, “pick up the black bowl next to the
ramekin and place it on the plate,” from saved initial state 0 with seed 1101.
The verified saved-state SHA-256 is
`d116842e01703fbea595511db813a3816c8d9f2302b4e979ebec4f15b7a750a2`.
Rendering uses the already-installed headless OSMesa path. Simulation physics
and model dependencies were not changed.

## Verified OSC and gripper convention

The pinned Robosuite `OSC_POSE` config uses `control_delta=true`, normalized
inputs in `[-1,1]`, translation outputs of ±0.05 m per world-frame axis, and
rotation-vector outputs of ±0.5 rad per axis. Relative translation is added to
the current end-effector world position. The interface sends one action per
key press; its translation slider bounds each increment to 1–20 mm and its
rotation slider bounds each increment to 1–8 degrees. The gripper is controlled
separately: Panda `-1` means open and `+1` means close. A gripper press advances
one simulation control step; repeat deliberately with fresh observation between
presses rather than assuming one command completes a grasp.

Keys: W/S move ±world X; D/A move ±world Y; R/F move ±world Z. I/K, L/J and
U/M rotate about the three configured axis-angle components. O opens and C
closes the gripper. Space or the Pause button pauses command acceptance; Resume
is explicit. Emergency Stop disables the session until its server process is
restarted. Each request carries an epoch and increasing sequence number;
concurrent commands are rejected without queueing. A lost connection does not
cause client retries or background stepping. The server binds only to BEAST's
`127.0.0.1`, never a LAN or Tailscale interface.

## Automated checks

- 17 unit tests pass for all key-axis signs, the OSC scaling, bounds, invalid
  inputs, and Panda gripper signs.
- A real OSMesa LIBERO self-test reset the fixed task/state, applied three
  automated W commands, measured end-effector world X from −0.208967 m to
  −0.206362 m (positive displacement 0.002605 m), and reset back to the same
  pinned state. This is a controller mapping check, **not human input**.
- Both camera arrays were confirmed as 256×256 RGB and their initial-state
  hashes matched the fixed scenario.
- The first browser attempt exposed an OSMesa thread-affinity bug: camera
  rendering performed from HTTP request threads returned corrupted frames.
  The server now creates, steps and closes MuJoCo on one dedicated owner
  thread; concurrent HTTP requests are rejected rather than queued. Regression
  self-test on that exact owner thread passed after three mapped W actions and
  reset: front-frame standard deviation 50.76/range 239; wrist-frame standard
  deviation 64.02/range 245. The first attempt's trajectory and video are
  preserved in the original run folder; the corrupted video is not evidence
  of a valid visual demonstration.
- The server offers a browser status endpoint and records each submitted human
  command with before/after robot state, mapped/executed action, task status,
  ownership and synchronized front/wrist video frames. Runtime logs and video
  are written under ignored `outputs/` and must not be committed.
- This worktree has no VLA handoff or resumption behavior yet. Those remain
  intentionally deferred until the user has personally validated manual
  control.

## Gate D — Lenovo instructions

The BEAST server is intended to run at `http://127.0.0.1:8765` and accept
connections only over an SSH local forward. From a Lenovo PowerShell window:

```powershell
ssh -N -L 8765:127.0.0.1:8765 friend-gpu
```

Leave that PowerShell window open, then open `http://127.0.0.1:8765` in the
Lenovo browser. Use one key press at a time and wait for fresh camera frames
before the next. Start with 5 mm translation increments. Test the axes in open
space, then move toward the bowl, open/close incrementally, and attempt a lift.
Check visually that the bowl itself rises and remains held; the interface's
task-success flag is separate from a grasp observation. Use Pause to stop
commands, Resume only when ready, or Emergency Stop to disable control.

When finished, close the browser and stop the tunnel with Ctrl+C in the
PowerShell window. The simulation server can be stopped cleanly with:

```powershell
ssh friend-gpu "cd /home/jess/projects/shared-autonomy-robot-learning-human-teleop && test -f outputs/human_teleop_task1_state0_seed1101_retry1/server.pid && xargs -r kill -TERM < outputs/human_teleop_task1_state0_seed1101_retry1/server.pid"
```

Runtime trajectory and camera video are saved in the ignored
`outputs/human_teleop_task1_state0_seed1101_retry1/` directory; the simulator
server log is `outputs/teleop_server_retry1.log`. The first attempt remains
preserved under `outputs/human_teleop_task1_state0_seed1101/`. Do not remove
either folder before we review the human-operated test. After manual control
works, the next step is
to verify the recorded trajectory and only then implement the separate
VLA-to-human-to-VLA handoff.
