# Scripted hover pilot: seeds 51–70

**Run date:** 2026-10-06  
**Result:** 20/20 successful runs (100% in this pilot)  
**Final target error:** average 5.56 mm; range 5.33–5.72 mm

## What was run

The Panda arm moved its open gripper to a target 10 cm above the red cube and held that position. For each run, the red cube was placed using a different, recorded random seed. The sampled tabletop bounds were X = 0.45–0.55 m and Y = −0.23 to −0.17 m. Each run had a 10-second simulated timeout. Success required the tool-center point to remain within 2 cm of its target for 0.5 simulated seconds.

Every seed from 51 through 70 succeeded. Each run saved matching numbers of static-camera and wrist-camera frames (61–64 of each), along with a manifest and synchronized state/action log. The per-seed values are in [seed_51_70.csv](../experiments/mujoco_hover/seed_51_70.csv); the scripts and instructions are in [xperiments/mujoco_hover](../experiments/mujoco_hover/README.md).

## How to interpret this

This is a useful baseline check: the scripted arm motion reached the hover target for these 20 cube placements. The result applies only to this controller, scene, and tested tabletop range. It is not evidence that the robot can touch or grasp the cube, and it does not evaluate a VLA, diffusion policy, Gaussian Process, XR interface, or human corrections. Those components have not been evaluated in this experiment.

## Data and provenance

The MuJoCo scene came from the existing UCL VLA_Benchmark repository (scene commit 708bfc8c67ecdd232e3130736055cf0856f1685d; MuJoCo 3.15.0). The controller and logger are the thesis experiment code in xperiments/mujoco_hover. Full episode image/state logs are retained locally under VLA_Benchmark/mujoco/recordings; this repository stores the compact CSV summary instead of those large recordings.
