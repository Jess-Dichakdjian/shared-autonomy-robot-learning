# VLA integration: current blockers

**Checked:** 6 October 2026  
**Purpose:** Record what is ready in the thesis work and what must be clarified before connecting a real VLA. No experiment or installation was performed for this inspection.

## What works now

- The existing UCL `VLA_Benchmark` MuJoCo scene opens and runs with the Franka Panda, table, cube, and cameras.
- The scripted hover baseline reaches the target above the red cube. The seed 71 interface smoke test and seed 72 adapter smoke test succeeded.
- The thesis repository has a model-independent policy interface with a common action format: seven arm joint-position targets and one gripper-finger-position target.
- `VLAAdapter` accepts an observation and instruction. Its no-op mode holds the observed pose, and its scripted-fallback mode calls the existing scripted policy. The adapter fallback was exercised in the seed 72 run; the action safety filter and logger remained active.

These checks establish the simulation and policy plumbing only. They do **not** show that a VLA can produce actions for this task.

## What is missing or uncertain

### Model source folders

In the local Windows `VLA_Benchmark` checkout, the README points to `VLA_Benchmark/Isaac_GR00T`, but that directory is absent. The checkout contains `VLA_Repos/Isaac-GR00T` and `VLA_Repos/openpi` as small files containing the absolute Linux paths `/home/yuxin/Isaac-GR00T` and `/home/yuxin/openpi`. They appear to be path or symlink references to folders on another machine; the actual source folders are not present in this checkout and those referenced Linux paths are not usable as local Windows source folders.

The README path `Isaac_GR00T` also differs in spelling and location from the `VLA_Repos/Isaac-GR00T` path reference. This needs to be reconciled with the working machine/source location.

### Server entry point and model files

The `VLA_Benchmark/README.md` instructs the user to run `python vla_server.py` from `VLA_Benchmark/Isaac_GR00T` and says to modify the model path in that script. The referenced directory and `vla_server.py` are missing from the current checkout, so the README's server command cannot be followed here. No checkpoint or model path is available in this checkout either. There is also a protocol detail to confirm: the README calls this an HTTPS server, while `VLAClient` defaults to `http://localhost:8000`.

### Client input and action output contract

The MuJoCo client's `VLAs/communications.py` posts a JPEG image and instruction to `/predict`; it can also send end-effector position, end-effector quaternion, and joint positions. It reads a JSON `action` field and converts it to a NumPy array. The server's required image size/normalization, state requirements, and response schema beyond that `action` field are not established.

The `mujoco/vla_simulation.py` comment describes a returned action sequence shaped roughly `(prediction_horizon, 9)`. It selects `new_action[-1]`, although the adjacent comment says to take the “first action.” `mujoco/panda_robot.py` can map a 9-value vector to eight MuJoCo controls by passing through seven arm values and averaging the final two finger positions into one gripper command. However, the exact model contract remains unconfirmed: the units and meaning of the seven arm values, the gripper convention, the prediction horizon, and which timestep of the sequence should be applied need confirmation. The client docstring itself is ambiguous about whether its returned actions are joint velocities or positions.

The thesis adapter's canonical action is seven **joint-position targets** plus one gripper target in metres. A real model adapter must explicitly convert its confirmed output into this canonical format and pass it through the safety filter before execution.

## Questions for Valerio

Before installing or connecting a model, ask Valerio to confirm:

1. **Which model should the thesis use:** GR00T or openpi?
2. **Where is the working source folder?** Please provide the machine and exact path, and clarify how the `Isaac_GR00T` and `VLA_Repos/Isaac-GR00T` names relate.
3. **Where is the checkpoint/model path?** Include any access or configuration needed to read it.
4. **Which machine and GPU should run inference?** Include the expected operating system and whether inference runs locally or remotely from MuJoCo.
5. **How should the server be started?** Provide the environment/setup steps, working directory, entry-point command, and endpoint/port.
6. **What is the exact input/output contract?** Specify image preprocessing and resolution, instruction format, optional state fields, response schema, prediction horizon, action dimensions, position-versus-velocity meaning, units, gripper convention, and which sequence timestep the client should apply.

## Ready status

The simulation, scripted hover, policy interface, and non-model adapter fallback are available. Real GR00T/openpi inference is **not tested**. Resolve the source, checkpoint, machine/server, and input/output questions above before implementing or running a real model backend.
