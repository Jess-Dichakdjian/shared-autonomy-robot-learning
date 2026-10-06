# Scripted MuJoCo hover baseline

This folder contains the small scripted controller and image/state logger used for an initial reach-and-hover baseline. The scripts were developed against the separate UCL `VLA_Benchmark` MuJoCo scene; that upstream repository and its scene assets are not copied into this thesis repository.

## Task

Move the Panda's open gripper to a point 10 cm above the red cube and hold it there. The cube is sampled inside the tested tabletop rectangle using a repeatable random seed. The controller does not touch, close around, or grasp the cube. A run succeeds when the tool-center point stays within 2 cm of its target for 0.5 simulated seconds, before the 10 simulated-second timeout.

## Run one episode

1. Set up the Python environment used by the local `VLA_Benchmark` MuJoCo project.
2. In PowerShell, change directory to that repository's `mujoco` folder. The script expects to find `franka_emika_panda/scene.xml` relative to the current directory.
3. Replace the example paths below with your local paths, then run:

```powershell
$python = 'C:\path\to\your\python.exe'
$thesisRepo = 'C:\path\to\shared-autonomy-robot-learning'
& $python "$thesisRepo\experiments\mujoco_hover\scripted_reach.py" --headless --randomize-cube --seed 71
```

The episode is saved under `recordings` in the current MuJoCo working directory. Its manifest records the seed and exact sampled position. Reusing a seed reproduces that placement; use a different seed for a new placement. Camera images and synchronized state/action data are saved with the episode.

## Recorded batch

`seed_51_70.csv` contains one summary row for each of the 20 unique seeds run on 2026-10-06. `../../docs/hover-pilot-results.md` explains the results and limitations. The full image and state/action episode folders remain in the local `VLA_Benchmark/mujoco/recordings` directory and are not committed here.
