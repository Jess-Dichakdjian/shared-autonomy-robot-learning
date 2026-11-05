# LIBERO-Spatial controlled perturbation baseline (discovery)

**Status:** descriptive discovery results; not a held-out test or population estimate.
**Date:** 2026-11-05.
**Canonical code branch:** `feat/libero-controlled-perturbation-evaluation` in `shared-autonomy-robot-learning`.
**Raw evidence:** ignored under `outputs/libero_controlled_perturbations_20261105/`; do not add these files to Git.

## Question and protocol

The frozen `lerobot/pi05_libero_finetuned_v044` policy was evaluated on LIBERO-Spatial task 1: “pick up the black bowl next to the ramekin and place it on the plate.” The task goal predicate is `On(akita_black_bowl_1, plate_1)`. All trials use saved initial state 0 (`d116842e…b7a750a2`), seed 1101, two 256×256 RGB cameras, relative OSC actions, 50-action prediction chunks with 10 actions executed per chunk, the pinned checkpoint revision `7152f13e34c6c115d808aea0e5d08aca74eac14b`, and the pinned model digest `877b3ec1…970997ae`. The renderer was OSMesa. The π0.5 model and preprocessing/postprocessing were frozen.

The pre-registered scenario manifest is `configs/libero_task1_scene_scenarios_v1.json` (SHA-256 `bcabf64dc889f04a54d2b9a21535c4c410b074e72b267e6267e2e1ae2711fb58`). It contains 12 discovery, 6 validation, and 6 held-out scenarios. Only discovery policy rollouts are reported here. Discovery physical preflight accepted 11 scenarios and rejected `D08_bowl_y_minus_5cm` because settling produced new bowl–ramekin penetration (maximum recorded depth about 7.7 mm). The six validation and six held-out policy evaluations have not been run; the held-out split remains reserved.

Perturbations translate the identified bowl or plate free-joint in world XY after the same saved-state reset. The object, task instruction, robot, controller, cameras, and checkpoint remain fixed. Each condition receives the same ten no-op settle steps. The task predicate is checked before policy control and must be false. Actual settled translations and camera hashes are recorded. For `D09`, physics settling changed the requested (+3.5, −3.5) cm bowl translation to an observed (+3.90, −2.32) cm; the observed value is reported below.

## Discovery outcomes

| Scenario | Requested scene change | Actual settled change (cm) | Outcome | Steps |
|---|---|---:|---|---:|
| D00 reference | none | bowl (0, 0) | success | 112 |
| D01 bowl X +2 cm | bowl (+2, 0) | (+2, 0) | success | 116 |
| D02 bowl X −2 cm | bowl (−2, 0) | (−2, 0) | success | 112 |
| D03 bowl X +5 cm | bowl (+5, 0) | (+5, 0) | success | 116 |
| D04 bowl X −5 cm | bowl (−5, 0) | (−5, 0) | success | 107 |
| D05 bowl X +8 cm | bowl (+8, 0) | (+8, 0) | success | 105 |
| D06 bowl X −8 cm | bowl (−8, 0) | (−8, 0) | timeout at evaluation limit | 280 |
| D07 bowl Y −2 cm | bowl (0, −2) | (0, −2) | success | 107 |
| D08 bowl Y −5 cm | bowl (0, −5) | — | rejected during physical preflight | — |
| D09 bowl diagonal | bowl (+3.5, −3.5) | (+3.90, −2.32) | success | 102 |
| D10 plate X +2 cm | plate (+2, 0) | (+2, 0) | success | 112 |
| D11 plate X −2 cm | plate (−2, 0) | (−2, 0) | success | 111 |

Among the 11 physically valid discovery scenes, the policy succeeded in 10 and timed out in one. These are 11 condition-level observations from one saved scene state and one seed, not independent randomized trials. They support only the narrow statement that the frozen policy completed these tested scene variants under this setup, while failing to complete D06 within 280 steps. They do not establish broad distributional robustness or a statistically reliable success rate.

A first D03 attempt terminated before action execution because the adapter returned an empty/non-finite action. Repeating the same D03 condition completed successfully. The initial error remains archived under `discovery/interrupted_attempts/`; its cause is undetermined and the repeat does not erase it.

## Passive GP monitoring

The existing GP artifact was loaded read-only from the separate GP worktree, with digest `de2ef0de…70a185f`; the threshold was frozen at `0.8421057488361606`. At fresh action-chunk boundaries, the monitor computed predictive uncertainty, GP–VLA disagreement, and nearest training-state distance. It could display a request-only assistance signal; it had no correction provider and no effect on predicted or executed policy actions.

The uncertainty monitor issued 40 request-only flags across these 11 rollouts: 19 flags occurred in successful rollouts and 21 in D06. D06 first flagged at step 40 and subsequently flagged at 50–120 and 160–270. Its chunk-boundary standardized RMS uncertainty had mean 0.844 and maximum 0.974. Yet the same frozen threshold also flagged in successful D01–D05 and D10 trials. D07, D09, and D11 completed without flags. Thus the monitor surfaced D06 early in this run, but its flags were not specific to task failure. With no matched request-only baseline, no intervention effect, only one D06 observation, and no held-out evaluation, this does not show that the GP predicts failure or that its alerts improve task success. Predictive variance remains a model-dependent action-regression quantity, not a failure probability or intervention utility.

## What the videos show

The D06 recording ends at the 280-step evaluation limit with the task unfinished. Visual inspection of sampled frames confirms an incomplete rollout but does not by itself establish the precise physical cause. The video is preserved locally as `Thesis_Evidence/libero_task1_D06_bowl_x_minus_8cm_timeout.mp4`; it is not tracked in Git. Full per-episode recordings and state/action trajectories remain in the ignored BEAST output tree.

## Answers to the study questions

1. **What moved, and by how much?** Identified bowl and plate free-joints were translated in world XY within the pre-registered range. Actual post-settle deltas are in the table and in each episode's `scene_perturbation.pose_records`; D09 shows why requested offsets alone are insufficient.
2. **How did π0.5 adapt?** Ten valid discovery conditions reached the simulator's task-success predicate; D06 did not reach it by the 280-step limit. The setup changes object poses, so this is a controlled scene-variation check, not new object or language generalization.
3. **Which conditions failed?** D06 (bowl X −8 cm) timed out. D08 was not an evaluation failure; it was rejected before inference for physical overlap. The first D03 inference attempt is a separate, archived runtime error followed by a successful retry.
4. **Was the GP warning useful?** D06 was flagged from step 40, but 19 flags also occurred in successes. This is an early descriptive association in one rollout, not validated predictive utility or causal benefit.
5. **What user-correction scenario follows?** A future validation study could compare a pre-specified request-only policy with matched random-budget requests and a human correction/handoff condition on the validation split. Define outcomes and correction accounting before running it; do not tune on the held-out six.
6. **Is this ready to learn from corrections?** The logger preserves chunk-start normalized state, predicted/executed actions, GP scores, request events, episode outcomes, and video, so it is a useful recording foundation. No human correction was delivered or incorporated here. Correction labels, synchronized operator events, data quality checks, and a separate train/validation protocol still require implementation and evaluation.

## Reproducibility and evidence locations

- Scenario definition: `configs/libero_task1_scene_scenarios_v1.json`.
- Physical discovery preflight: `outputs/libero_controlled_perturbations_20261105/scene_validation_discovery.json`.
- Rollout records and videos: `outputs/libero_controlled_perturbations_20261105/{discovery,pilot_continuation,discovery_remaining}/` (ignored).
- The initial D00–D02 successes are recorded in `discovery/`; D03–D04 and D10 retry/continuation records are in `pilot_continuation/`; D05–D09 and D11 are in `discovery_remaining/`.
- All outputs identify the pinned policy, processor hashes, GPU runtime, GP artifact hash, threshold, manifest hash, and actual perturbation records.
- Do not report the discovery tally as the reserved held-out test result.
