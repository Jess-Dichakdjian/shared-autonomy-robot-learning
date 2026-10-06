# Thesis implementation status

**Last updated:** 6 October 2026  
**Repository:** shared-autonomy-robot-learning

## Current status in one sentence

A Panda robot can make a scripted, open-gripper move to a hover point above a red cube in the existing MuJoCo benchmark, and the runs are logged. The learned-policy and human-correction parts of the thesis pipeline have not been implemented in this work.

## Completed so far

| Completed step | What it contributes to the thesis |
|---|---|
| The thesis repo records the research question and intended shared-autonomy pipeline. Its README reports that the literature review and system design are complete. | Gives the implementation a research goal and a planned end-to-end structure. The literature review itself is not code in this repository. |
| The existing UCL VLA_Benchmark MuJoCo scene was inspected and run with a Franka Panda, table, cube objects, and cameras. | Provides a simulation baseline to build on. The scene and benchmark are existing UCL infrastructure, not an original thesis contribution. |
| A scripted controller was added for moving the Panda's open gripper to a point 10 cm above the red cube and holding there. | Confirms that basic robot motion and a measurable success condition can work in the simulator. It is a debugging baseline. |
| A small tabletop position range and repeatable random seeds were added for the red cube. | Makes it possible to repeat a placement and compare controller runs across different placements. |
| Episode logging was added for static and wrist camera images, robot state/action records, and a run manifest. | Creates a first record of what the simulated robot saw and did, which is groundwork for later policy evaluation and data handling. |
| The initial randomized hover pilot used five unique seeds (46–50): all five succeeded. Seed 46 was also repeated once at the same position. A further 20 unique seeds (51–70) all succeeded. | Gives an early repeatability check for the scripted hover baseline. The 20-run batch had a mean final target error of about 5.56 mm, with a 5.33–5.72 mm range. These results cover hover only and are not final thesis results. |
| The hover scripts, instructions, per-seed CSV for seeds 51–70, and results note were added to this thesis repository. | Keeps the new experiment code and its compact summary alongside the thesis project. The full image and state/action episodes remain in the local VLA_Benchmark/mujoco/recordings folder. |

## Baseline versus thesis contribution

The current robot behaviour is a **scripted hover baseline**. It moves above the cube, keeps its gripper open, and does not touch or grasp the object. The 100% result for seeds 51–70 means only that this scripted controller met the hover success rule for those 20 sampled placements.

This baseline helps verify the simulator, motion control, seeding, success measurement, and logging. It is not a VLA result, a diffusion-policy result, or evidence that uncertainty-aware shared autonomy works. The thesis contribution is still intended to come from implementing and evaluating the learned-policy, uncertainty, and human-correction parts of the pipeline.

The Franka model, MuJoCo scene, and benchmark infrastructure come from the existing UCL VLA_Benchmark project. They should be described as prior infrastructure, not claimed as new thesis work.

## Not done yet

- **VLA inference:** no VLA has been run to produce robot actions in this simulation.
- **GR00T/openpi selection:** neither has been selected as the thesis VLA, installed for this project, or connected to the simulator.
- **Diffusion/flow policy execution:** no diffusion or flow policy is currently driving this logged hover pipeline. The existing README mentions earlier small-scale diffusion-policy work, but this repository does not contain a verified policy execution for the current task.
- **Uncertainty layer:** no live uncertainty estimate or uncertainty-triggered pause has been implemented.
- **ProMP/GP:** neither Probabilistic Movement Primitives nor Gaussian Process code has been implemented or evaluated here. Which method belongs in the final design remains undecided.
- **Human correction:** no operator correction has been requested, recorded, or fed back into a policy or uncertainty model.
- **Unity/XR:** no Unity or XR interface is connected to the simulator.
- **Cube contact/manipulation:** contact with the red cube and grasping/lifting it have not been implemented or evaluated. The current cube task is only hover.
- **End-to-end evaluation:** there are no results yet for a learned policy, uncertainty estimate, intervention, grasp, or full shared-autonomy pipeline.

## Current minimum viable thesis pipeline

The immediate task progression is:

1. **Minimum task — reach/hover:** move the open gripper to the target point above the red cube. A scripted baseline for this stage already exists and has been run with varied seeds.
2. **Next task — contact/touch:** extend the approach so the robot moves down and makes controlled contact with the cube. Define a contact success condition and stop safely after contact.
3. **Stretch task — grasp/lift:** close the gripper around the cube and lift it, if time and reliable contact control allow.

Square insertion remains an optional later task. It is not the immediate implementation target unless Valerio explicitly requires it.

For the thesis pipeline, use the simplest task stage that supports the research question and can be evaluated reliably. Then send the camera image, task instruction, and robot state to one selected learned policy; validate and execute its actions; add uncertainty-triggered pausing and recorded human correction; and evaluate task outcomes and interventions across repeatable episodes. The task progression above can remain at hover or contact if that is sufficient for a valid thesis and supervisor agrees.

At present, only simulator groundwork, scripted hover motion, and basic logging for the minimum reach/hover task exist. Contact, grasp/lift, learned-policy execution, uncertainty, human correction, and end-to-end evaluation remain future work.

## Next three coding tasks

1. **Make the existing red-cube hover a repeatable episode in the thesis repo.** Define reset, seed, timeout, target tolerance, hold time, and success/failure output so the minimum task can be rerun and evaluated consistently.
2. **Implement a separate controlled cube-contact stage.** Start from a safe hover, descend toward the cube, detect contact, and stop; report contact success/failure. Keep the established hover task available as the baseline.
3. **Prepare one model-independent observation/action interface for these stages.** Record camera image, instruction, robot state, action, seed, and outcome, and validate/clamp proposed actions to simulator limits. Use a stub first; do not install or run GR00T, openpi, diffusion, or flow policies as part of these tasks.

After these three tasks, choose the policy/checkpoint based on available inputs, action format, and compute. Treat grasp/lift as stretch work and Square insertion as optional future work. Start VLA inference only after the interface is documented and the simulator can safely execute validated actions.
