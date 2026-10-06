# Thesis implementation status

**Last updated:** 6 October 2026  
**Repository:** `shared-autonomy-robot-learning`

## Current status in one sentence

A Panda robot can make a scripted, open-gripper move to a hover point above a red cube in the existing MuJoCo benchmark, and the runs are logged. The learned-policy and human-correction parts of the thesis pipeline have not been implemented in this work.

## Completed so far

| Completed step | What it contributes to the thesis |
|---|---|
| The thesis repo records the research question and intended shared-autonomy pipeline. Its README reports that the literature review and system design are complete. | Gives the implementation a research goal and a planned end-to-end structure. The literature review itself is not code in this repository. |
| The existing UCL `VLA_Benchmark` MuJoCo scene was inspected and run with a Franka Panda, table, cube objects, and cameras. | Provides a simulation baseline to build on. The scene and benchmark are existing UCL infrastructure, not an original thesis contribution. |
| A scripted controller was added for moving the Panda's open gripper to a point 10 cm above the red cube and holding there. | Confirms that basic robot motion and a measurable success condition can work in the simulator. It is a debugging baseline. |
| A small tabletop position range and repeatable random seeds were added for the red cube. | Makes it possible to repeat a placement and compare controller runs across different placements. |
| Episode logging was added for static and wrist camera images, robot state/action records, and a run manifest. | Creates a first record of what the simulated robot saw and did, which is groundwork for later policy evaluation and data handling. |
| The initial randomized hover pilot used five unique seeds (46–50): all five succeeded. Seed 46 was also repeated once at the same position. A further 20 unique seeds (51–70) all succeeded. | Gives an early repeatability check for the scripted hover baseline. The 20-run batch had a mean final target error of about 5.56 mm, with a 5.33–5.72 mm range. These results cover hover only and are not final thesis results. |
| The hover scripts, instructions, per-seed CSV for seeds 51–70, and results note were added to this thesis repository. | Keeps the new experiment code and its compact summary alongside the thesis project. The full image and state/action episodes remain in the local `VLA_Benchmark/mujoco/recordings` folder. |

## Baseline versus thesis contribution

The current robot behaviour is a **scripted hover baseline**. It moves above the cube, keeps its gripper open, and does not touch or grasp the object. The 100% result for seeds 51–70 means only that this scripted controller met the hover success rule for those 20 sampled placements.

This baseline helps verify the simulator, motion control, seeding, success measurement, and logging. It is not a VLA result, a diffusion-policy result, or evidence that uncertainty-aware shared autonomy works. The thesis contribution is still intended to come from implementing and evaluating the learned-policy, uncertainty, and human-correction parts of the pipeline.

The Franka model, MuJoCo scene, and benchmark infrastructure come from the existing UCL `VLA_Benchmark` project. They should be described as prior infrastructure, not claimed as new thesis work.

## Not done yet

- **VLA inference:** no VLA has been run to produce robot actions in this simulation.
- **GR00T/openpi selection:** neither has been selected as the thesis VLA, installed for this project, or connected to the simulator.
- **Diffusion/flow policy execution:** no diffusion or flow policy is currently driving this logged hover pipeline. The existing README mentions earlier small-scale diffusion-policy work, but this repository does not contain a verified policy execution for the current task.
- **Uncertainty layer:** no live uncertainty estimate or uncertainty-triggered pause has been implemented.
- **ProMP/GP:** neither Probabilistic Movement Primitives nor Gaussian Process code has been implemented or evaluated here. Which method belongs in the final design remains undecided.
- **Human correction:** no operator correction has been requested, recorded, or fed back into a policy or uncertainty model.
- **Unity/XR:** no Unity or XR interface is connected to the simulator.
- **Target manipulation task:** the Square task named in the thesis plan has not been implemented and evaluated in this work; the cube hover is only a simple baseline.
- **End-to-end evaluation:** there are no results yet for a learned policy, uncertainty estimate, intervention, grasp, or full shared-autonomy pipeline.

## Current minimum viable thesis pipeline

The minimum end-to-end system still to be built is:

1. Run the planned Square manipulation task in MuJoCo with repeatable episode resets and a clear success/failure rule.
2. Send camera images, the task instruction, and the robot state to one selected learned policy.
3. Convert its output into bounded robot actions and execute them in simulation.
4. Estimate uncertainty during execution and pause autonomy when uncertainty is high.
5. Let a human provide a correction, record that correction, and resume or recover the task.
6. Evaluate task success, failures, uncertainty-triggered interventions, and the effect of corrections over repeatable runs.

At present, only the simulator groundwork, scripted hover motion, and basic logging from the first part of this list exist. The rest is a plan, not implemented functionality.

## Next three coding tasks

1. **Add a repeatable Square-task episode wrapper.** Reuse the thesis's planned Square task and existing benchmark assets; define reset, seed, episode timeout, and success/failure reporting. Keep the hover baseline unchanged.
2. **Define one observation/action record format for that task.** Record timestamped camera images, instruction, robot state, action, seed, and outcome in each episode so policy data and evaluation use the same fields.
3. **Add a model-independent policy interface and action checks.** Make a stub policy pass observations through the interface and validate/clamp proposed actions to safe simulator limits. Do not install or run GR00T, openpi, diffusion, or flow policies as part of this task.

After these three tasks, choose the exact policy/checkpoint based on the available model inputs, action format, and compute. VLA inference should start only after that interface is documented and the simulator can safely execute validated actions.
