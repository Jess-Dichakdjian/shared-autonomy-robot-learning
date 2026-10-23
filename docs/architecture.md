# System architecture

## Current evaluation path

LIBERO Spatial task 0
  -> synchronized agent-view and wrist RGB cameras plus robot state
  -> LeRobot LIBERO observation processing
  -> task instruction tokenization and pi0.5 policy interface
  -> predicted action chunk (batch, time, 7)
  -> LeRobot policy and environment postprocessors
  -> validated seven-value relative OSC action
  -> LIBERO simulation
  -> trajectory, metrics, configuration, seed and video logging

The verified path currently substitutes a mock policy for learned inference.
The real LeRobot pi0.5 tokenizer and preprocessors/postprocessors have been
exercised against synthetic LIBERO observations, but pretrained policy weights
have not been loaded on the Lenovo.

## Observation and action contract

- Cameras: agent view and wrist view, each RGB, 256x256, synchronized per step.
- State: LeRobot LIBERO processor maps end-effector position (3), quaternion
  converted to axis-angle (3), and two gripper positions into an eight-value
  observation.state.
- Language: the LIBERO task instruction is passed through LeRobot's pi0.5
  tokenizer/preprocessor.
- Policy action: pi0.5 emits a chunk with shape (1, T, 7).
- Environment action: each chunk item passes through policy and LIBERO
  postprocessors, then finite/range validation and clipping into the
  seven-dimensional relative OSC interface.
- Lifecycle: reset clears queued actions and calls the policy reset hook.
  Evaluation runs in torch.inference_mode() and records action latency.

The earlier MuJoCo hover pilot uses a separate seven-joint-target plus gripper
distance interface. Those action spaces have different semantics and are not
interchanged.

## Planned extension points

Uncertainty estimation, intervention decisions and XR correction are not yet
implemented. They can be added around the validated evaluator after a real
pretrained baseline is established, while preserving the evaluator's logging
and task contracts.
