# GP Uncertainty, Risk Signals, and Assistance Interface: 4 November 2026

## Scope and questions

This experiment asks whether a state-only Gaussian Process (GP) fitted to frozen π0.5 actions can (i) imitate the policy's horizon-zero action, (ii) provide empirically useful predictive intervals or an error-ranking signal, and (iii) identify episodes that end unsuccessfully. A separate engineering question is whether an online monitor can request assistance without changing frozen-policy execution. We do not claim that GP uncertainty is VLA epistemic uncertainty, task-failure probability, or human-intervention utility.

The frozen checkpoint is `lerobot/pi05_libero_finetuned_v044`, revision `7152f13e34c6c115d808aea0e5d08aca74eac14b`, checkpoint SHA-256 `877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae`. It receives two 256×256 RGB cameras, language, and checkpoint-normalized 8D robot state; it predicts 50 seven-dimensional relative OSC actions and executes 10 at a time. BEAST uses one RTX 5070 Ti Laptop GPU; LIBERO rendering used OSMesa because EGL task rendering hangs in this WSL2 setup. The existing 20-episode task-0 evidence was not modified.

## Gate A: what the GP measures

The model is a seven-output collection of independent scalar Gaussian Process regressors, one for each action coordinate. Each maps checkpoint-normalized state `x ∈ R^8` to the corresponding raw checkpoint-normalized horizon-zero π0.5 action `y ∈ R^7`; it is a state-only policy surrogate. The learned policy also conditions on images and language, which the GP does not receive. Distinct visual/language situations can therefore map to similar states and different actions.

Each regressor uses an ARD Matérn-5/2 kernel and standardized training targets. The exported predictive mean is inverse-transformed to the original normalized action coordinates. The reported predictive variance includes latent posterior variance plus fixed homoscedastic observation-noise variance, then is transformed back to action units; standard deviation is its square root. The fixed noise is a modeling assumption, not a measured sensor-noise estimate. For the selected model its standardized-target variance is 0.01. Predictive variance is conditional on the fitted kernel and hyperparameters. It is not, by itself, a calibrated measure of policy knowledge or task risk.

Optimized ARD length scales frequently reached the configured upper bound. This means some feature sensitivities were not resolved within that search range; it cautions against interpreting the fitted kernel as a stable physical relevance analysis. The validation search compared upper bounds 100 and 1000 and fixed noise variances 1e-4, 1e-3 and 1e-2. The selected candidate still emitted 14 bound warnings. Increasing the bound is not automatically a remedy: it changes the search domain and requires new validation data.

At a fresh chunk boundary, the current VLA horizon-zero prediction is available alongside the current state. Thus GP–VLA disagreement can be computed without changing π0.5. Here it is an action-surrogate residual, not policy error: the VLA output is the regression target, not a ground-truth safe action. A nearest-training-state distance is a simple novelty baseline. These signals can rank where to inspect or request assistance, subject to validation; they cannot establish that assistance is needed or useful.

## Gate B: validation selection and calibration

Model choices were made on validation episodes only. Training comprised 102 historical task-0 chunk-start rows from 12 episode IDs plus 24 pilot rows from task-1/state-1 and task-4/state-1. Validation comprised 25 chunk-start rows from task-1/state-0 and task-4/state-0. The historical task-0 test split was not used to select this new model. Candidate selection minimized validation RMSE, then among candidates within 2% selected 90% interval coverage closest to 0.90. Selected parameters: Matérn-5/2 ARD, fixed noise variance 0.01 in standardized-target units, upper length-scale bound 100, two optimizer restarts.

On validation, GP MAE/RMSE were 0.3743/0.5547; Ridge MAE/RMSE were 0.6213/0.7834. Empirical nominal 90%/95% interval coverage was 92.0%/94.3%. A validation-only descriptive multiplier of 0.9099 was calculated from the 90th percentile of absolute standardized residual divided by 1.645. It is not a conformal guarantee, and the chunk samples are temporally correlated. The multiplier did not transport to the held-out episodes (see below), so unscaled intervals remain the primary report and recalibrated figures are shown only as a failed transport check.

## Gate C: frozen pilot and held-out protocol

The task plan was recorded before running these policy episodes in `configs/gp_uncertainty_pilot_plan.json` and `configs/gp_uncertainty_heldout_plan.json`. LIBERO-Spatial task 1 instruction: “pick up the black bowl next to the ramekin and place it on the plate.” Task 4 instruction: “pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate.” Saved simulator states and their hashes are recorded in the held-out plan. Seeds were fixed: pilot 1101–1102; held-out 1202–1206. Each pilot task used states 0 and 1. The held-out set used states 2–6 for each task, max 280 environment steps. State 1 from each task augmented training; state 0 was validation; states 2–6 were held out. No failures were selected or manufactured.

All runs used the frozen pretrained checkpoint and passive GP sidecar: predicted VLA actions continued to control the environment. Both cameras, normalized state, seven-action contract, simulator success, chunk-start outputs, trajectory, and video were validated. Pilot outcomes were 4/4 successes (task 1: 111 and 100 steps; task 4: 121 and 131 steps). Held-out outcomes were 9/10 successes: task 1 was 5/5 (104, 101, 113, 110, 107 steps); task 4 was 4/5 (145, 126, one 280-step timeout, 126, 118 steps). The only adverse outcome was task 4, state index 4, seed 1204. This is a small, task-specific sample, not a broad LIBERO risk benchmark.

## Gate D: action fidelity, uncertainty, and task outcomes

Across 137 held-out chunk-start state/action pairs, GP action-surrogate MAE/RMSE were 0.4746/0.7315; Ridge was 0.7190/0.9563. Task 1 GP MAE/RMSE were 0.4037/0.6256 (Ridge 0.6260/0.7963); task 4 GP were 0.5236/0.7965 (Ridge 0.7834/1.0528). This supports better surrogate fit than this Ridge baseline on these episodes, not better task control: the GP did not control the robot.

The original predictive intervals covered 88.43% and 92.39% of scalar held-out targets at nominal 90% and 95%. The validation-scaled intervals covered only 85.82% and 90.41%; validation scaling therefore did not transfer. Per-dimension 90% coverage ranged from 78.83% to 93.43%. Standardized residual standard deviations ranged from 0.81 to 1.49, indicating dimension-dependent dispersion. These empirical coverage rates are descriptive and correlated within episode.

Uncertainty was defined as `U = RMS_j(std_j / s_y,j)`, with `s_y,j` the training-fitted target scale. Disagreement was `D = RMS_j((mu_GP,j - a_VLA,j) / s_y,j)`. Novelty was `H = min_i ||(x - x_i)/s_x||_2` using training-fitted state scales. On held-out chunk boundaries, episode-cluster bootstrap Spearman correlations with absolute GP action residual were 0.543 (95% interval 0.331–0.638) for U and 0.548 (0.352–0.634) for H, from 10 episode clusters. D's correlation is 1 by construction because it is the same standardized absolute surrogate residual aggregated across dimensions; it is not independent evidence.

Only one of ten held-out episodes timed out. Its maximum episode scores ranked above all nine successes for U, D and H (sample AUROC/AP 1.0), but one positive makes these metrics extremely unstable and not a validation of risk discrimination. At thresholds frozen from the validation 90th percentiles, U requested at 45 boundaries, D at 33 and H at 58. All flagged the one timeout; they also flagged 8, 7 and 8 successful episodes, respectively. A request during a successful episode is not known to be unnecessary because no counterfactual correction was run. Matched-budget random schedules over the 137 boundaries detected the one timeout with probability 1.0, 0.9999 and 1.0 at those respective budgets. The timeout episode had 28 boundaries, versus 11–15 for successes, so uniform boundary sampling is exposure-biased toward longer episodes. With one failure and this random reference, useful task-risk discrimination is **inconclusive**, not verified. No combined U/D score was selected; validation evidence did not justify adding one.

## Gate E: request interface and correction-source audit

`thesis_vla/gp/monitoring.py` separates signal computation, threshold decision, request object, correction-provider protocol, budget, and arbitration. Supported modes are autonomy-only, passive monitoring, uncertainty threshold, disagreement threshold, state-distance threshold, and deterministic matched-budget random requests. Thresholds use frozen validation quantiles; decisions occur only at fresh action-chunk boundaries and do not inspect future observations or episode outcomes. Passive mode leaves the VLA action unchanged. The GP worker remains a separate CPU process. The action arbitrator accepts a correction only when active mode, a valid request, provider and budget are all explicitly supplied; it checks seven finite in-range action values and records provenance. Tests cover threshold equality, non-boundary calls, deterministic scheduling, budget accounting, passive invariance, active selection and invalid provider outputs.

The repository and available simulator assets were inspected for an expert/scripted controller, replayable human corrections, teleoperation demonstrations, or another valid correction source. None was found. A mock provider exists only to test the interface contract. No oracle was invented, no future successful action was replayed, no correction was applied, and no XR integration or intervention experiment was performed. Active selective control is therefore not feasible in this milestone. The monitor is a tested request-only extension point, not a demonstrated shared-autonomy method.

## Limitations and interpretation

- Ten held-out episodes and one timeout cannot establish robust failure detection, threshold utility, or a human-assistance policy.
- The episode-held-out sample is small and task-specific; chunk observations are correlated.
- The GP omits vision and language and models policy actions, not optimal/safe actions or environment dynamics.
- Fixed noise, bound-seeking length scales and failed interval transport limit uncertainty-calibration claims.
- The matched-random reference is descriptive and boundary-exposure-biased; it does not make the single timeout informative.
- No correction source was available, so intervention benefit, cost, recovery, autonomy fraction and false-assistance rates remain unmeasured.
- No claim is made about XR, fine-tuning, policy failure probability, or human intervention utility.

## Reproducibility and artifacts

Code is on branch `feat/gp-uncertainty-selective-assistance`, based on verified BEAST integration commit `82611289112843033fd90164ac7856533666a690`. The CUDA/VLA environment is `~/.venvs/shared-autonomy-robot-learning-pi05-beast` (Python 3.12, single-GPU CUDA/LeRobot stack); the separate CPU GP environment is `~/.venvs/shared-autonomy-robot-learning-gp-beast` (Python 3.12, NumPy/SciPy/scikit-learn). LIBERO used OSMesa for WSL2 camera rendering. Relevant tests: VLA/evaluator suite: 38 passed, 6 skipped; GP/interface suite: 27 passed. GP tests emit expected scikit-learn kernel-bound warnings.

Tracked frozen plans and chosen validation configuration:

- `configs/gp_uncertainty_pilot_plan.json`
- `configs/gp_uncertainty_heldout_plan.json`
- `configs/gp_uncertainty_selected_model.json`

Ignored evidence on BEAST (videos, per-episode logs, models and raw arrays are not committed):

- Pilot: `outputs/gp_uncertainty_pilot_2026-11-04/`
- Held-out rollouts: `outputs/gp_uncertainty_heldout_2026-11-04/`
- Candidate search, fitted artifact, and validation plots: `outputs/gp_uncertainty_validation_2026-11-04/`
- Authoritative held-out metrics, episode/chunk tables, and plots: `outputs/gp_uncertainty_heldout_analysis_final_2026-11-04/`

Re-run the held-out analysis against the preserved ignored outputs with:

```bash
cd /home/jess/projects/shared-autonomy-robot-learning-gp-uncertainty
source ~/.venvs/shared-autonomy-robot-learning-gp-beast/bin/activate
PYTHONPATH=. python scripts/evaluate_gp_risk_signals.py \
  --model outputs/gp_uncertainty_validation_2026-11-04/models.joblib \
  --task1 outputs/gp_uncertainty_heldout_2026-11-04/libero_spatial_task1 \
  --task4 outputs/gp_uncertainty_heldout_2026-11-04/libero_spatial_task4 \
  --threshold-config configs/gp_uncertainty_selected_model.json \
  --output outputs/gp_uncertainty_heldout_analysis_reproduction
```

The original 20-episode evidence and this milestone's run artifacts are ignored by Git. Only source, tests, frozen plans/config and this report are intended for commit. A branch backup/evidence archive should include the report, plans and aggregate CSV/JSON/plots plus SHA-256 manifest, but no checkpoint weights, private credentials, or unnecessary videos.

## Status matrix

| Area | Status | Evidence / boundary |
|---|---|---|
| VLA inference | Verified | Frozen π0.5 produced the 14 pilot and 10 held-out rollouts; historical 20/20 task-0 run remains intact. |
| GP regression | Verified | 137 held-out policy-action pairs; better aggregate MAE/RMSE than Ridge in this sample. |
| GP calibration | Inconclusive | Empirical coverage below nominal overall; validation multiplier failed to transfer. |
| Risk discrimination | Inconclusive | One timeout among 10; random matched-budget schedules also detected it. |
| Selective-assistance trigger | Implemented and tested | Request-only triggers and explicit safe arbitration contract; no intervention was applied. |
| Corrective-action provider | Unavailable | No valid expert, replay, teleoperation or XR correction source found. |
| Controlled intervention comparison | Not feasible | Provider unavailable and risk discrimination inconclusive; no assisted results claimed. |
| XR integration | Intentionally outside this milestone | No XR implementation or evaluation. |
