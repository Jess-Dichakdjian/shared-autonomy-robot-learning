# BEAST real pi0.5 and GP integration

Date: 2026-11-04
Branch: `feat/beast-vla-gp-integration`
Base: `feat/offline-gp-baseline` at `0ba83d4f974ac82ba10a84f9c03d25b584ad42be`

## Verified execution environment

BEAST ran Ubuntu 22.04 under WSL2 with one NVIDIA GeForce RTX 5070 Ti Laptop GPU (12,227 MiB, compute capability 12.0). The isolated VLA environment used Python 3.12.16, PyTorch 2.10.0+cu128, CUDA runtime 12.8, LeRobot 0.6.2 at source commit `146d8474932c8b566717eb2738ba459337ab71c4`, Transformers 5.5.4, MuJoCo 3.8.1 and hf-libero 0.1.4. FP32 CUDA tensor operations and BF16 matrix multiplication were executed successfully.

LIBERO camera rendering passed with OSMesa and two 256x256 RGB views. EGL hung in the WSL GPU bridge, so the benchmark used OSMesa CPU rasterization. The MuJoCo simulator, task, observations and controller remain the same, but rasterization/rendered pixels can differ from the UCL EGL setup. The two rendering conditions should be reported separately.

## Pinned policy and load checks

The benchmark used `lerobot/pi05_libero_finetuned_v044` at revision `7152f13e34c6c115d808aea0e5d08aca74eac14b`. The checkpoint file SHA-256 was `877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae` (7,473,096,344 bytes). The historical identifier `lerobot/pi05_libero_finetuned` is a separate alias recorded in previous work; it was not used for this run. The PaliGemma processor initialized from `google/paligemma-3b-pt-224` revision `35e4f46485b4d07967e7e9935bc3786aad50687c`.

Strict loading required materializing six deterministic non-persistent buffers after meta initialization. Unknown meta buffers fail closed. Active modules were placed on CUDA, while unused vocabulary output heads stayed on CPU. Weight fingerprint before and after placement matched. The original two-GPU numerical comparison could not be reproduced on this single-GPU host; it was replaced with checkpoint-hash, placement and action-contract checks.

One real inference produced finite shape `(1, 50, 7)`. The contract preserved two RGB cameras, an 8D checkpoint-normalized state, seven raw normalized relative OSC actions, 50 predicted actions and 10-step execution chunks. One initial complete task-0 feasibility episode succeeded (78 steps). The fixed benchmark below is a separate run.

Approximate load time was 11-14 seconds. Process peak RSS after load was 9,793,638,400 bytes. PyTorch peak CUDA allocation/reservation was approximately 7.13/7.29 GB. Active CUDA placement was about 478.9 MB FP32 plus 6.47 GB BF16; CPU BF16 vocabulary heads occupied about 1.58 GB. The benchmark's normal chunk inference latency averaged 0.2813 seconds (median 0.2811, p95 0.2927, max 0.3217 seconds).

## Real BEAST benchmark

A five-episode pilot validated export before resuming the same planned experiment for 15 more episodes. The 20-episode run covered saved initial states/seeds 0-19 for LIBERO-Spatial task 0. All 20 episodes succeeded, totaling 1,625 environment steps and 170 fresh chunk-start predictions/state-action pairs. Mean episode length was 81.25 steps. The descriptive 95% Wilson interval for success was 0.8389-1.0; this is a fixed single-task, single-host run and should not be interpreted as broad generalization evidence.

Evidence is stored outside Git under:

- `outputs/beast_pi05_spatial20_2026-11-04/`: benchmark records, videos, actions, states, metrics, configs and GPU telemetry.
- `outputs/beast_pi05_spatial20_2026-11-04/gp_dataset_bundle.tar.gz`: validated dataset bundle, SHA-256 `d0fe03a61d2a14ac213e6efbcf65391b2e6015e0d4857a890c1bbb1982f6b27b`.
- `outputs/beast_pi05_spatial_task0_2026-11-04-r2/`: the separate initial feasibility episode.

The GP bundle contains `X (170,8)` checkpoint-normalized states, `Y (170,7)` raw normalized horizon-zero predictions, plus physical states, executed controller actions and full 50-action chunks. Episode IDs, seeds and saved initial-state indices are 0-19; all values are finite and `Y == Y_chunk[:,0]`. Outputs and model artifacts are ignored by Git. They are BEAST results and must not be merged with or described as the UCL benchmark.

Telemetry sampled 84 times: maximum GPU temperature 51 C, maximum sampled memory 7,303 MiB and minimum free memory 4,642 MiB; swap remained zero.

## GP training and held-out evaluation

The CPU-only GP environment is `~/.venvs/shared-autonomy-robot-learning-gp-beast` (Python 3.12.16, NumPy, SciPy, scikit-learn, matplotlib and pytest; no Torch). GP fitting used seven independent Matern 5/2 regressors with fixed standardized observation-noise variance `1e-3`, train-only X/Y scaling, episode-held-out partitions (split seed 17), and a Ridge baseline.

The validation episodes were 9, 14, 16 and 17 (36 pairs); test episodes were 3, 4, 5 and 6 (32 pairs); training used 102 pairs from the remaining 12 episodes. Validation selected the configuration before the test partition was evaluated once. Validation MAE/RMSE were 0.3368/0.5484 for GP versus 0.4121/0.5616 for Ridge. GP interval coverage was 83.33% at 90% nominal and 89.68% at 95% nominal.

On the held-out test partition, GP MAE/RMSE were 0.2946/0.4414 versus Ridge 0.4859/0.6498. GP coverage was 83.48%/88.84% at 90%/95% nominal, below nominal. Some optimized length scales reached the upper bound. These are held-out action prediction and interval-coverage results for this dataset, not calibrated failure probabilities or proof of GP epistemic uncertainty. All episodes succeeded, the task and checkpoint are fixed, neighboring chunk starts are correlated, and the 8D state omits visual and language context.

Final GP artifacts and plots are in `outputs/beast_gp_final_2026-11-04/`. The model bundle is local and Git-ignored.

## Passive live-GP integration

The VLA process uses a persistent subprocess in the separate GP CPU environment, so scikit-learn is not imported into the VLA environment. At each fresh pi0.5 chunk boundary, the sidecar receives the exact 8D checkpoint-normalized state and returns seven GP means and standard deviations with timing and provenance. It logs but does not modify VLA predictions or executed controls.

A paired state-20 episode succeeded with monitoring enabled and without monitoring (80 steps each). The passive run emitted eight GP predictions; worker mean latency was about 1.545 ms and client round-trip mean about 1.791 ms. Predicted actions, executed actions, states and chunks matched bitwise over all 80 steps; the videos also had identical SHA-256 `7d840ab95a597431cf572ef0fd6da5cb36b6f8fe267920492bdfa0fdf63b6f6a`. This establishes a no-control-change check for this paired episode only, not general behavioral equivalence.

Evidence is in `outputs/beast_pi05_passive_gp_state20_2026-11-04/` and `outputs/beast_pi05_baseline_state20_2026-11-04/`.

## Reproduction

Run CUDA policy evaluation in the VLA environment using the BEAST-specific script and configuration, and run GP training/evaluation from the GP-only CPU environment. Exact invocation metadata is preserved in each output directory. For the fixed benchmark, the wrapper is:

```bash
source ~/.venvs/shared-autonomy-robot-learning-pi05-beast/bin/activate
python scripts/benchmark_libero_pi05_beast.py --help
```

The GP pipeline accepts the BEAST bundle only with its recorded SHA-256 identity:

```bash
source ~/.venvs/shared-autonomy-robot-learning-gp-beast/bin/activate
python scripts/run_gp_baseline.py --help
```

Do not rerun a benchmark over an existing output directory unless resuming the explicitly recorded plan.

## Scope and next scientific step

No intervention thresholds, autonomous takeover, XR correction, or fine-tuning were implemented. The next scientific step is to transfer and archive the BEAST dataset and reports alongside (but clearly distinct from) UCL results, then design additional tasks/seeds and a protocol that can test whether GP residuals or uncertainty provide useful decision information. The current all-success, one-task dataset cannot test failure detection or intervention utility.
