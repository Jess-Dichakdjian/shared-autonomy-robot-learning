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

## Offline inference API

The data flow at an eligible policy prediction boundary is:

```text
π0.5 observation
  -> saved processor produces checkpoint-normalized observation.state (8 values)
  -> GPInference.predict(state)
  -> (mean, predictive_std), each with 7 raw checkpoint-normalized action values
```

The evaluator-facing contract is NumPy-only:

```python
from thesis_vla.gp import GPInference

predictor = GPInference.load("outputs/gp_final/models.joblib")
mean, std = predictor.predict(normalized_state)       # each shape (7,)
means, stds = predictor.predict(normalized_state_batch)  # each shape (N, 7)
```

The loader accepts the pipeline's models.joblib bundle or a directly serialized
IndependentGaussianProcess. It restores the fitted input and target scalers with
the seven scalar models. Incompatible artifacts, absent files, invalid shapes
and non-finite states raise explicit errors. Keep model artifacts trusted:
joblib uses Python pickle internally and must not load files from untrusted
sources.

Call this interface only when the VLA has a fresh chunk prediction, using the
same checkpoint-normalized eight-value state that produced the π0.5 observation
state. Do not call it for each queued chunk action as if those actions were new
observations. The interface reports a diagnostic prediction alongside the VLA
path; it does not modify policy actions or evaluator behavior.

Returned standard deviations are in raw checkpoint-normalized Y units. Their
variance is conditional on the fitted kernel hyperparameters and equals latent
GP posterior variance plus the configured fixed homoscedastic observation-noise
variance after target inverse-scaling. It does not integrate kernel
hyperparameter uncertainty and is not automatically calibrated uncertainty,
failure probability or intervention utility.

## Methodology review and current limits

The split and train-only scaling avoid within-episode sample leakage. This does
not make chunk starts independent: the UCL batch has 161 pairs from 20 episodes,
one task and a single policy/checkpoint setup. The robot state omits the visual
scene and language that condition π0.5, so state-only inputs can be aliased.
All 20 recorded episodes succeeded; this dataset has no failed episodes with
which to test whether GP spread distinguishes failures. Real validation can
assess held-out action prediction and interval coverage, but cannot support a
failure-probability interpretation from this batch.

The configured noise variance is fixed at 1e-4 in training-standardized target
units. It is a documented initial Gaussian noise assumption, not measured
sensor noise or a calibrated quantity. Optimizer restarts default to zero, so
the marginal-likelihood optimizer uses one initialization and can reach a local
optimum. Bounds warnings are surfaced during fitting; the optimized kernels are
saved in experiment metadata. Review warnings and fitted parameters on the real
episode split before interpreting metrics or changing bounds/noise.

## Minimal synthetic inference example

Run from the repository root. This fits only a deterministic test fixture; it
does not read the UCL dataset and is not a thesis result.

```bash
source ~/.venvs/shared-autonomy-robot-learning-gp/bin/activate
PYTHONPATH=.:tests python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
import joblib

from gp_synthetic_fixture import write_synthetic_gp_npz
from thesis_vla.gp import (
    GPInference, IndependentGaussianProcess, load_gp_dataset,
    measure_cpu_prediction_latency, split_by_episode,
)

with TemporaryDirectory() as directory:
    root = Path(directory)
    fixture = write_synthetic_gp_npz(root / "synthetic.npz")
    data = load_gp_dataset(fixture)
    split = split_by_episode(data.episode_id, seed=17)
    fitted = IndependentGaussianProcess(seed=17).fit(data.X[split.train], data.Y[split.train])
    artifact = root / "models.joblib"
    joblib.dump({"gp": fitted}, artifact)
    predictor = GPInference.load(artifact)
    mean, std = predictor.predict(data.X[split.validation][0])
    print(mean.shape, std.shape)  # (7,), (7,)
    print(measure_cpu_prediction_latency(predictor, data.X[split.validation][0]))
PY
```

Run the interface and baseline tests:

```bash
PYTHONPATH=.:tests python -m pytest -q tests/test_gp_baseline.py tests/test_gp_inference.py
```


## Verified BEAST dataset

The separately collected BEAST dataset, its episode-held-out GP evaluation, artifact paths, metrics, passive-inference comparison, environment conditions and limitations are recorded in [the BEAST VLA-GP integration report](beast_pi05_vla_gp_integration_2026-11-04.md). BEAST and UCL datasets/results must remain separately identified. The BEAST run used OSMesa rendering under WSL2; UCL's validated run used EGL. These rendering conditions are not pixel-identical.
