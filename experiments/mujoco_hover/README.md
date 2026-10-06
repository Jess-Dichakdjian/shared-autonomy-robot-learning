# Scripted MuJoCo hover baseline

This folder contains the small scripted controller and image/state logger used for an initial reach-and-hover baseline. The scripts were developed against the separate UCL `VLA_Benchmark` MuJoCo scene; that upstream repository and its scene assets are not copied into this thesis repository.

## Policy interface

`policy_interface.py` defines a model-neutral observation and action contract. The current scripted differential-IK controller is wrapped as a `ScriptedPolicy`. `VLAAdapter` is a placeholder with the same observation-to-action role: it accepts the observation and instruction, then either calls an optional scripted fallback or returns a no-op that holds the observed joints and gripper. The marked section in `VLAAdapter.predict` is where selected GR00T/openpi input processing, inference, and action conversion will be connected later. It does not call a model, and neither model has been installed.

The canonical action is seven arm joint-position targets in radians plus one Panda finger-position target in metres. `ActionLimits` clamps targets to the MuJoCo actuator ranges and limits each command change relative to the previous command. `VLAAdapter` produces that same `RobotAction` format. Any future model-specific output (including action chunks, if applicable) must be converted to this format and passed through `ActionLimits` before it reaches the simulator.

To use the placeholder from Python, construct `VLAAdapter()` and call `predict(observation, instruction="...")`; it returns the observed pose as a no-op. To route through a scripted controller instead, pass it as `VLAAdapter(fallback_policy=scripted_policy)`. The current hover episode still uses `ScriptedPolicy` directly. The stub has not been connected to episode execution, and it does not invoke or require GR00T/openpi.

Each logged frame stores the instruction, seed, robot-state observation, `requested_action`, `clamped_action`, and current episode outcome. The synchronized static and wrist camera images are saved as JPEGs and referenced from the frame record. The final episode outcome is recorded in `manifest.json`.

## Policy interface smoke test

On 2026-10-06, the scripted hover policy was run through this interface with seed 71. The first attempt timed out because the safety filter limited commands relative to measured joint position. The filter was changed to limit target changes relative to the previous commanded target, preserving the baseline's cumulative joint-target updates. The next run with the same seed succeeded with a final hover error of 5.63 mm.

The successful episode saved 62 observations and 62 synchronized image pairs. Each frame includes the instruction, seed, observation, `requested_action`, and `clamped_action`; the manifest records the final instruction, seed, and outcome (`success`). This confirms the scripted policy/interface/filter/logger path. It does not test VLA inference, uncertainty estimation, or human correction.

On 2026-10-06, a separate seed 72 smoke test selected `VLAAdapter` in `scripted_fallback_no_model` mode. It succeeded with a 5.62 mm final hover error and saved 63 synchronized observation/image pairs. All frames recorded `requested_action`, `clamped_action`, and the selected policy. The manifest records `model_inference_called: false`. The gripper remained open at 0.04 m, and adjacent logged clamped joint targets changed by no more than 0.035 rad over each 0.1 s logging interval. This checks adapter selection, fallback, logging, and safety filtering only; no GR00T/openpi inference ran.

## Task

Move the Panda's open gripper to a point 10 cm above the red cube and hold it there. The cube is sampled inside the tested tabletop rectangle using a repeatable random seed. The controller does not touch, close around, or grasp the cube. A run succeeds when the tool-center point stays within 2 cm of its target for 0.5 simulated seconds, before the 10 simulated-second timeout.

## Run one episode

1. Set up the Python environment used by the local `VLA_Benchmark` MuJoCo project.
2. In PowerShell, change directory to that repository's `mujoco` folder. The script finds `franka_emika_panda/scene.xml` there automatically; use `--scene` to pass a different XML path.
3. Replace the example paths below with your local paths, then run:

```powershell
$python = 'C:\path\to\your\python.exe'
$thesisRepo = 'C:\path\to\shared-autonomy-robot-learning'
& $python "$thesisRepo\experiments\mujoco_hover\scripted_reach.py" --headless --randomize-cube --seed 71
```

The episode is saved under `recordings` in the current MuJoCo working directory. Its manifest records the seed and exact sampled position. Reusing a seed reproduces that placement; use a different seed for a new placement. Camera images and synchronized state/action data are saved with the episode.

## Recorded batch

`seed_51_70.csv` contains one summary row for each of the 20 unique seeds run on 2026-10-06. `../../docs/hover-pilot-results.md` explains the results and limitations. The full image and state/action episode folders remain in the local `VLA_Benchmark/mujoco/recordings` directory and are not committed here.
