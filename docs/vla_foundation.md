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

Hugging Face authentication was verified in WSL and only google/paligemma-3b-pt-224/config.json was explicitly downloaded. Building LeRobot's π0.5 processors then initialized the real GemmaTokenizer and policy/environment pipelines. The tokenizer cache contains vocabulary/config assets (including a 17.5 MB tokenizer.json); no policy weight file was downloaded. An opt-in test passes synthetic LIBERO-compatible observations through the real π0.5 tokenizer/preprocessor and postprocessors with mock inference, retaining the eight-value state and seven-value action contracts. The reusable Pi05PolicyAdapter.from_lerobot_policy constructor was exercised with a mock inference object and the real LeRobot processor pipelines; the actual checkpoint-backed policy remains uninitialized because its 7.47 GB weights are not present.


For the automated real-simulator integration test (requires the cached assets and
OSMesa), run:

    RUN_LIBERO_INTEGRATION=1 MUJOCO_GL=osmesa PYTHONPATH=. pytest -q tests/test_libero_integration.py

The normal test command skips this simulator-dependent case, so unit tests remain
usable on machines without the LIBERO rendering stack.


## Real pretrained evaluation runner

The CUDA-host entry point is `scripts/evaluate_libero_pi05.py`. Its default is
one episode so the first real run is constrained by default. It calls the same
`run_evaluation` loop and logger as the mock path, requires a local checkpoint
directory, sets the policy and processor device, and never downloads checkpoint
weights:

    source ~/.venvs/shared-autonomy-robot-learning-pi05/bin/activate
    cd ~/projects/shared-autonomy-robot-learning
    hf auth login
    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/evaluate_libero_pi05.py \
      --checkpoint /local/path/pi05_libero_finetuned_v044 \
      --suite libero_spatial --task-id 0 --episodes 1 --seed 0 \
      --execute-steps 10 --device cuda --dtype auto \
      --output outputs/pi05_libero_spatial

Checkpoint identity and revision: the user's historical run used
`lerobot/pi05_libero_finetuned`. Hugging Face currently redirects that ID to
`lerobot/pi05_libero_finetuned_v044`, so it is an alias, not evidence of a
separate current checkpoint. The v044 main revision checked on 2026-10-17 is
`7152f13e34c6c115d808aea0e5d08aca74eac14b`; its `model.safetensors` is
7.47 GB, with processor/config files alongside it. For reproducibility the
procedure below pins this full revision. The report should still record both
the historical alias and the resolved canonical ID/revision. No weights were
downloaded during this work. The user reports a prior
successful π0.5 run on the HP OMEN RTX 2060 with 6 GB VRAM; that is useful
historical evidence for the GPU route, but this WSL machine has not reproduced
it. The runner provides explicit dtype selection if the host/checkpoint needs
it; validate the chosen dtype and memory use on that GPU before a full benchmark.
A CUDA Linux machine can run simulator, model and evaluator together, avoiding
an unimplemented remote-inference transport. LeRobot offers a remote policy
server for deployments where the GPU is on another computer, but that is a
separate path and is not needed for this simulation evaluation.


After authenticating the tokenizer, run the actual π0.5 processor integration
without model weights with:

    RUN_PI05_PROCESSOR_INTEGRATION=1 PYTHONPATH=. pytest -q tests/test_pi05_processor_integration.py

The default suite reports 10 passed and skips both opt-in integrations. The real OSMesa LIBERO integration and authenticated π0.5 processor integration each pass when run explicitly in this configured Ubuntu environment.


## CUDA Linux host: reproducible first real run

This project pins the LeRobot source commit
`146d8474932c8b566717eb2738ba459337ab71c4`; the configured and tested
environment reports LeRobot 0.6.2. Keep that exact source pin for this
evaluation. The Lenovo's existing environment is CPU-only and must not receive
the checkpoint. Install this environment on the target CUDA Linux host, using
the CUDA-enabled PyTorch build appropriate for that host and the repository's
requirements file. Do not replace it with a CPU-only Torch wheel.

Before downloading weights, run the weight-free preflight. It verifies that
PyTorch can initialize CUDA, reports GPU/free VRAM, checks conservative disk
headroom against the pinned checkpoint size, and launches the existing LIBERO
Spatial task-0 renderer smoke test with OSMesa. It does not fetch model files.
The 80 GB figure in LeRobot docs is a training recipe, not an inference
minimum. Public documentation does not specify a guaranteed inference VRAM
minimum; the user's historical RTX 2060 6 GB result is evidence from another
machine, not a guarantee for this exact revision. Record the preflight's GPU
and free-memory output before proceeding.

`lerobot/pi05_libero_finetuned` resolves to
`lerobot/pi05_libero_finetuned_v044` on the Hub. The v044 revision pinned
above contains a 7.47 GB model file. The command reserves at least twice that
size at the target model directory's filesystem as download/cache headroom.
The separate gated Google tokenizer requires the account to have accepted
`google/paligemma-3b-pt-224` access and be authenticated on the CUDA host.

On the CUDA Linux host, use Python 3.12 and the same pinned repository
checkout. Install the LIBERO OSMesa system libraries if they are not already
present, then create an isolated environment. Install a CUDA-enabled PyTorch
build selected for the host's NVIDIA driver using PyTorch's official
installation selector before installing the repository requirements; verify
torch.cuda.is_available() before proceeding. The repository requirements
pin LeRobot's source commit, its LIBERO/pi extras, pytest and CMake.

```bash
sudo apt-get update
sudo apt-get install -y build-essential libosmesa6
python3.12 -m venv ~/.venvs/shared-autonomy-robot-learning-pi05
source ~/.venvs/shared-autonomy-robot-learning-pi05/bin/activate
python -m pip install --upgrade pip
# First install CUDA-enabled PyTorch selected for this host's driver.
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pip install -r requirements-pi05-cuda-linux.txt
python -c 'import torch, lerobot; print(torch.__version__, lerobot.__version__, torch.cuda.is_available())'
python scripts/preflight_pi05_cuda.py --checkpoint-dir "$HOME/models/pi05_libero_finetuned_v044" --gl osmesa
hf auth whoami
hf download lerobot/pi05_libero_finetuned_v044 --revision 7152f13e34c6c115d808aea0e5d08aca74eac14b --local-dir "$HOME/models/pi05_libero_finetuned_v044"
MUJOCO_GL=osmesa PYTHONPATH=. python scripts/evaluate_libero_pi05.py --checkpoint "$HOME/models/pi05_libero_finetuned_v044" --suite libero_spatial --task-id 0 --episodes 1 --seed 0 --execute-steps 10 --device cuda --dtype auto --output outputs/pi05_libero_spatial_task0_first
```

The runner loads a local directory only, then calls the established pi0.5
adapter, shared evaluation loop and episode logger. A successful run should
produce metrics, predicted/executed actions, state/config/seed records, and a
side-by-side camera video. Inspect those artifacts and peak GPU memory before
increasing episode count. A later multi-episode benchmark must specify its
episode count explicitly and use distinct/reproducibly recorded seeds.

The target CUDA Linux machine and its user/session access have not yet been
provided to this Codex session. The checkpoint is now cached at ~/models/pi05_libero_finetuned_v044 on the
Lenovo. Its safetensors header reports 690 BF16 and 122 F32 tensors (7,473,096,344
bytes); weights remain unloaded. The CPU preflight stops before renderer/model
loading because WSL has 6.81 GiB available against its conservative 10.44 GiB
headroom heuristic. No actual checkpoint load or real-policy inference is
claimed complete.


### Lenovo CPU feasibility findings (2026-10-17)

Windows reports 15.32 GiB physical RAM, 1.68 GiB available (89% memory load),
and 4.32 GiB pagefile available. WSL reports 7.42 GiB total, 6.81 GiB
available at the CPU preflight, and 2 GiB swap with no swap used. WSL's root
filesystem has 948 GiB free; Windows C: has 828 GiB free. The CPU is an AMD
Ryzen AI 7 445 (6 cores / 12 logical CPUs) and exposes AVX2, AVX-512 BF16 and
AVX-VNNI. PyTorch is 2.11.0+cpu with six intra-op threads. Small BF16 matmul,
layer-norm and scaled-dot-product-attention operations succeeded; this verifies
those primitives only, not the complete Pi05 graph.

The weights occupy about 6.96 GiB in memory before Python, processor and
inference workspace. WSL's entire 7.42 GiB limit leaves too little safety
margin, and Windows host memory is already under pressure. The CPU preflight
uses 1.5x the checkpoint file size (10.44 GiB available) as a conservative
load-headroom heuristic, not an upstream Pi05 requirement. It correctly
refuses the Lenovo before model load or renderer startup. No WSL memory setting
was changed. The preflight can be checked with:

```bash
python scripts/preflight_pi05_cuda.py --device cpu --checkpoint-dir "$HOME/models/pi05_libero_finetuned_v044"
```

The real runner already accepts `--device cpu`; its automatic dtype retains the
checkpoint's BF16 configuration. It now sets evaluation mode and the shared
evaluator wraps policy actions in `torch.inference_mode()`. Step logs record
inference latency; metrics record process peak RSS and the runner records load
time and post-load peak RSS. The authenticated processor integration passed
on CPU with mock inference. Those checks do not establish actual model load or
inference viability on this Lenovo. Do not increase WSL memory or use swap to
force the load: Windows has only 15.32 GiB total RAM and is currently highly
loaded.
