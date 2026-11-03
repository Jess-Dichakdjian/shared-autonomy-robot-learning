# Offline GP regression baseline

## Dataset contract and scientific target

The input is the numeric GP chunk-start dataset from the UCL pi0.5 evaluation.
The primary regression uses only:

- **X (N, 8):** the exact checkpoint-normalized observation.state captured at a
  fresh pi0.5 action-chunk prediction.
- **Y (N, 7):** the raw checkpoint-normalized pi0.5 prediction at horizon zero
  of that same fresh chunk, in the checkpoint ACTION MEAN_STD representation.
- **episode_id (N,):** the evaluated episode identity for group splitting.

Y_executed is the controller action after checkpoint unnormalization, LIBERO
postprocessing, and clipping. It is not the target in this baseline. X_physical
is not substituted for normalized X. Future queued actions are not new
state/action pairs. The loader reads NPZ with numpy.load(..., allow_pickle=False);
it also reads the NPZ directly from the documented tar bundle without extracting
archive paths. Only numeric schema fields are needed.

## Formulation

Seven independent scalar GaussianProcessRegressor models map eight state
coordinates to the seven action coordinates. The starting configuration uses a
Matern-5/2 kernel, train-only standardization of X and each Y coordinate, and
fixed homoscedastic Gaussian observation noise with variance 1e-4 in standardized
training-target units. RBF is available as an alternative. Hyperparameters are
optimized by scikit-learn's marginal-likelihood optimizer; restart count is
zero by default for a reproducible, light CPU run. Optimized kernel parameters are exported per action. If scikit-learn reports that a length scale or amplitude reached its configured bound, inspect that warning and the exported values before interpreting validation; the bound is a constraint, not evidence that the parameter is identified.

The observation-noise variance is added as alpha to the training covariance.
For reported predictions, predictive observation variance is explicitly
defined as latent-function posterior variance plus that same independent
observation-noise variance. It is transformed back to raw normalized Y units.
Thus reported standard deviations and interval coverage refer to a future noisy
target under this model. They are not automatically calibrated pi0.5
epistemic uncertainty, failure probability, or intervention utility.

The state-only inputs omit camera and language context even though the VLA is
image-conditioned. Visually different scenes can therefore produce different
actions at similar X. The held-out episodes are correlated samples from one
task/setup; 161 chunk-start rows are not 161 independent robot trials.

## Splitting, comparison, and final test

Unique episode IDs are deterministically partitioned before fitting any
preprocessing. Default fractions are 60% train, 20% validation and 20% test
for the 20-episode UCL set (12/4/4 episodes). Fractions use ceiling for small
sets; at least three distinct episodes are required. No rows from an episode
cross partitions.

Input and target StandardScalers are fitted on training episodes only. The GP
and a Ridge regression baseline are fitted on train rows. Validation reports
MAE/RMSE per action coordinate, predictive standard deviations, standardized
residual summaries and empirical central 90%/95% interval coverage. Coverage is
the fraction of held-out scalar targets within GP mean plus/minus the Gaussian
quantile times the predictive standard deviation defined above. It is an
empirical sample statistic, not proof of calibration.

The default command does not evaluate test rows. Inspect the validation
comparison and freeze model/config choices first. Then run the same config on a
new output directory with --final-test for the final test report. The tool
refuses to overwrite a non-empty output directory so earlier evidence is
preserved. It saves split episode IDs, resolved configuration, model bundle,
numeric predictions, metrics and action-wise plots.

## Development environment and tests

Use the isolated Python 3.12 CPU environment; it contains NumPy, SciPy,
scikit-learn, matplotlib, pytest and their small dependencies, with no torch or
CUDA packages. requirements-gp.txt gives supported ranges and
requirements-gp-lock.txt records the tested resolved versions. Tests generate
deterministic, explicitly synthetic fixtures matching X/Y/episode_id. Those
tests validate code behavior only and are not pi0.5 results.

## Running after UCL dataset transfer

Copy gp_dataset_bundle.tar.gz from the workstation using the command in
pi05_multi_episode_protocol.md. Then from the canonical repository root:

```bash
source ~/.venvs/shared-autonomy-robot-learning-gp/bin/activate
python scripts/run_gp_baseline.py \
  --data outputs/ucl_pi05_spatial20_2026-10-30/gp_dataset_bundle.tar.gz --verified-ucl-pi05 \
  --output outputs/gp_validation_2026-11-03

# After reviewing validation and locking choices, use a separate output:
python scripts/run_gp_baseline.py \
  --data outputs/ucl_pi05_spatial20_2026-10-30/gp_dataset_bundle.tar.gz --verified-ucl-pi05 \
  --output outputs/gp_final_2026-11-03 --final-test
```

Outputs per run: metrics.json, splits.json, experiment.json, config.json,
models.joblib, predictions.npz, validation_predicted_vs_target.png,
validation_standardized_residuals.png and, for the final run, equivalent
final-test plots. Do not describe synthetic fixture results as actual VLA
experimental results.
