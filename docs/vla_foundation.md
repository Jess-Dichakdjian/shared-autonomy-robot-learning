# VLA foundation: LIBERO and LeRobot

This branch adds the thesis integration under thesis_vla/. The existing MuJoCo
hover pilot remains independent prior work. Its seven joint targets plus
gripper distance are not interchangeable with LIBERO's seven-dimensional
relative OSC action in [-1, 1].

## Current implementation

- The LIBERO config selects libero_spatial task 0, two 256x256 RGB cameras,
  pixels_agent_pos observations, 20 Hz, initialized states, hard resets, and
  relative control. The factory creates one synchronous LeRobot LIBERO vector
  environment.
- WSL robosuite EGL rendering fails because the user cannot access /dev/dri and
  robosuite requires an EGL device-display extension. A direct MuJoCo EGL scene
  renders through a software fallback, but that fallback does not satisfy
  robosuite. The verified environment backend is OSMesa; it avoids changing
  render-device groups or security settings. Software rendering may limit
  throughput.
- LiberoVectorAdapter unwraps one-environment vector results and validates,
  finite-checks, and clips seven-dimensional actions to the environment range.
- Pi05PolicyAdapter loads a local checkpoint on demand, defaults to local-only
  loading, and wires LeRobot observation conversion, LIBERO camera/state
  processing, policy preprocessing, and both action postprocessors. It requests
  a chunk, executes at most ten steps by default, and records chunk id/index.
  clear_action_queue discards stale actions and calls policy.reset() to flush
  policy temporal inference memory.
- No policy checkpoint, demonstration dataset, or learned inference result is
  included in this foundation.

## Observation/action contract

LeRobot LiberoEnv supplies two uint8 HWC camera images and nested robot state.
LeRobot preprocess_observation maps the cameras to observation.images.image
and observation.images.image2, adds a batch dimension, converts to float CHW
in [0,1], and returns tensors. LeRobot's LiberoProcessorStep flips both image
axes and maps end-effector position (3), quaternion to axis-angle (3), and
gripper qpos (2) to an eight-value observation.state. The pi0.5 preprocessor
then tokenizes task text and applies the checkpoint's input transforms. The
policy action postprocessor and LIBERO environment postprocessor run on each
chunk step before actions reach the environment.

LIBERO's action is seven-dimensional relative OSC; the hover interface
describes seven joint-position targets plus a gripper distance. They have
different controllers and units and stay separate.

## Environment setup

The isolated Python 3.12 venv contains LeRobot 0.6.2 source, CPU PyTorch,
LIBERO, robosuite and pytest. The egl-probe build requires make and a C/C++
toolchain; with current CMake, set CMAKE_POLICY_VERSION_MINIMUM=3.5 for the
install command. OSMesa is provided by Ubuntu's libosmesa6 runtime package.

The official LIBERO geometry/texture assets are cached at
~/.cache/libero/assets. No demonstrations or model weights were downloaded.
To rerun the checkpoint-free smoke test from the canonical repository:

    MUJOCO_GL=osmesa python scripts/libero_smoke.py --gl osmesa --suite libero_spatial --task-id 0

## Verified versus pending

Verified: pinned LeRobot 0.6.2 dependency installation; official LIBERO config
instantiation; exact image conversion, 180-degree image flip, and eight-value
state processing; eight passing automated tests; 128x128 EGL rendering
with software fallback; complete libero_spatial task 0 OSMesa reset, two
256x256 camera checks, and one seven-value no-op relative OSC step.

Pending: loading a real pi0.5 checkpoint and running inference. No benchmark
success score or real-policy performance is claimed.


## Evaluation pipeline (checkpoint-free mock)

`thesis_vla.evaluation.run_evaluation` runs seeded episodes against the one-env
`LiberoVectorAdapter`, resets policy state and queued chunk actions at each
episode, and writes:

- `steps.jsonl`: seed/task/config, current and next raw robot state, raw
  predicted action, postprocessed/clipped executed action, chunk id/index,
  reward, terminal flags and observed success.
- `episodes.jsonl`: seed, outcome, return, length, reset info, config and video path.
- `metrics.json`: observed success rate (null if the environment reports no
  success field), mean/std return, mean length and episode-level results.
- `episode_NNN.mp4`: synchronized agent-view and wrist-camera images side by
  side, from initial observation through final step.

Run a short mock-policy integration rollout (no checkpoint is loaded):

    source ~/.venvs/shared-autonomy-robot-learning-pi05/bin/activate
    cd ~/projects/shared-autonomy-robot-learning
    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/evaluate_libero_mock.py \
      --suite libero_spatial --task-id 0 --episodes 1 --seed 123 \
      --max-steps 3 --execute-steps 2 --output outputs/mock_libero_smoke

Test this repository with `PYTHONPATH=. pytest -q`. The project has no
packaging metadata yet, so setting the root import path is required.

Verified on 2026-10-17: 10 unit tests pass. A real OSMesa LIBERO spatial
task-0 rollout completed three mock actions and saved metrics, JSONL trajectories,
and a playable MP4. Its reported success rate was 0.0 under the intentionally
uninformed mock policy; this is a pipeline check, not a learned-policy score.

## Pretrained pi0.5 deployment assessment

The current WSL environment has a CPU-only PyTorch build; `torch.cuda.is_available()`
is false, although WSL exposes `/dev/dxg`. No CUDA inference performance has
been measured. LeRobot's current π0.5 documentation identifies the model as a
large VLA, gives an 80 GB GPU configuration for LIBERO finetuning, and documents
a LIBERO-specific checkpoint. This does not establish 80 GB as an inference
minimum. For a practical first pretrained evaluation, use a Linux workstation
or cloud GPU with LeRobot's CUDA build, run the simulator and evaluator on that
same machine if possible, and record its GPU, dtype, latency and versions.
Remote inference is technically possible, but adds a policy-server/network
boundary that this LIBERO evaluator has not implemented or validated.

LeRobot's π0.5 processor construction tried to fetch the gated
`google/paligemma-3b-pt-224` tokenizer and received HTTP 401 in this environment.
Actual inference therefore also needs authorized Hugging Face access to that
tokenizer and a local pretrained checkpoint (or access to the appropriate Hub
checkpoint). The mock path intentionally uses LeRobot's actual LIBERO
observation and environment action pipelines, without requesting that tokenizer
or downloading model weights. The existing `from_pretrained` adapter remains
the swap-in path once model access and compute are available.


For the automated real-simulator integration test (requires the cached assets and
OSMesa), run:

    RUN_LIBERO_INTEGRATION=1 MUJOCO_GL=osmesa PYTHONPATH=. pytest -q tests/test_libero_integration.py

The normal test command skips this simulator-dependent case, so unit tests remain
usable on machines without the LIBERO rendering stack.


## Real pretrained evaluation runner

The CUDA-host entry point is `scripts/evaluate_libero_pi05.py`. It calls the same
`run_evaluation` loop and logger as the mock path, requires a local checkpoint
directory, sets the policy and processor device, and never downloads checkpoint
weights:

    source ~/.venvs/shared-autonomy-robot-learning-pi05/bin/activate
    cd ~/projects/shared-autonomy-robot-learning
    hf auth login
    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/evaluate_libero_pi05.py \
      --checkpoint /local/path/pi05_libero_finetuned_v044 \
      --suite libero_spatial --task-id 0 --episodes 10 --seed 0 \
      --execute-steps 10 --device cuda --dtype auto \
      --output outputs/pi05_libero_spatial

The Hugging Face `lerobot/pi05_libero_finetuned_v044` checkpoint is 7.47 GB
on disk. No weights were downloaded during this work. The user reports a prior
successful π0.5 run on the HP OMEN RTX 2060 with 6 GB VRAM; that is useful
historical evidence for the GPU route, but this WSL machine has not reproduced
it. The runner provides explicit dtype selection if the host/checkpoint needs
it; validate the chosen dtype and memory use on that GPU before a full benchmark.
A CUDA Linux machine can run simulator, model and evaluator together, avoiding
an unimplemented remote-inference transport. LeRobot offers a remote policy
server for deployments where the GPU is on another computer, but that is a
separate path and is not needed for this simulation evaluation.
