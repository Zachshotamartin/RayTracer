# Diagnosing stalled reconstruction learning

The custom U-Net can stall near its noisy-input identity solution. A controlled
training diagnostic identified radiance conditioning as an actionable contributor.
This is an optimization result, not a qualified replacement renderer or a speedup.
The released `diffuse-pilot-v1.onnx` remains unchanged.

## Diagnosis and correction

The C10/C11 screening checkpoints barely changed noisy RGB, including on eight
inspected training views. On the first inspected view, C10's final decoder had
95.8% zero activations and very small gradients in deeper layers. Output gradients
were present; broad saturation of the output clamp did not explain these cases.

We tested four fixed 64×64 center crops from distinct training layouts, at 4 spp.
The same patches, initial weights, AdamW rate 0.0003, reconstruction/gradient/energy
loss, and 200 optimizer updates were used for each paired comparison. Random
augmentation, fusion, preservation sampling, scheduling and early stopping were
disabled in this diagnostic. Validation and test images were not used to fit the
patches. Reported error is log-radiance MAE within pixels eligible for reconstruction.

The correction in C13 expresses radiance in larger internal units: color enters
as `log1p(16 * RGB)` and variance as `log1p(256 * variance)`. Predictions are divided
by 16 to restore physical HDR units. The physical output ceiling, original feature
buffers, parameter count, support policy and high-sample raw bypass are preserved.
The factor is an explicit model configuration value saved with the checkpoint and
ONNX metadata; it is not a display exposure change or a fitted target-dependent
normalization. Sixteen is a tested setting on this pilot, not a universal optimum.

Leaky ReLU is available as a separate diagnostic control. It removed zero-activation
blocking but did not consistently resolve the unscaled recipe. The conditioned
ReLU and Leaky ReLU controls were similar here, so C13 retains the original ReLU.
Legacy configurations retain radiance scale 1 and their original activation.

The reproducible diagnostic is
[`diagnose_learning.py`](../tools/diagnose_learning.py). It records selected training
patches, seeds, data identity, code hashes, losses, gradient norms and activation
statistics, under a wall-time cap. It does not label fitting a few patches as
generalization. The follow-up controls use frozen source commit
`fbee052997fd0cfb951fabd43a88d620752266c1`.

## Frozen controlled results

These follow-up comparisons use the same frozen implementation for both arms.
Only `model.radiance_scale` differs. Seeds are paired; parameter count is 479,299
in every run. The fixed-patch numbers below are error reduction relative to raw
inputs, after exactly 200 optimizer updates.

| Seed | Original scale 1 | Corrected scale 16 |
| --- | ---: | ---: |
| 42 | 0.54% | 39.03% |
| 43 | 5.84% | 40.97% |

The second check retains the full C11 recipe: all 448 training examples from eight
layouts, random 96×96 crops, edge sampling, exposure/color/spatial augmentation,
independent-noise fusion and measured near-clean pairs. Each arm runs five epochs
(280 updates), with the original 50-epoch cosine schedule. Validation at each
checkpoint uses the same 80 images from four development layouts: five sample
budgets, four views and the first noise stream. Test data stays sealed.

| Seed | Validation SSIM, original → corrected | Edge MAE, original → corrected | Edge-error reduction |
| --- | ---: | ---: | ---: |
| 42 | 0.7370 → 0.8335 | 0.08838 → 0.06440 | 27.1% |
| 43 | 0.7370 → 0.8345 | 0.08851 → 0.06377 | 28.0% |

The new health flag identified both original-recipe runs as making little
improvement over raw inputs; neither conditioned run received that flag. All four
runs stopped intentionally at the five-epoch diagnostic boundary, not because they
completed the planned 50-epoch study. **All four still fail the structural gate**:
a-trous scores 0.9574 SSIM / 0.04656 edge MAE on this selection subset. The correction
addresses early optimization stagnation; it does not yet establish competitive
final quality, preservation at higher sample budgets or broad generalization.

Evidence: [fixed unscaled](detail-development/conditioning-fixed-unscaled.json),
[fixed scaled](detail-development/conditioning-fixed-scaled.json),
[paired full-recipe histories/configurations](detail-development/conditioning-short-recipe.json),
[HDR region summary](detail-development/conditioning-c5-hdr-regions.json),
and [all region records](detail-development/conditioning-c5-hdr-regions-per-image.jsonl.gz).
The complete pilot is a small development collection, and none of these timings
should be used as an uncontended speed comparison.

To repeat the fixed-patch comparison, run `ml/tools/diagnose_learning.py` with the
C11 and C13 configs, the pilot dataset, separate output JSON paths, and
`--steps 200 --rates 0.0003 --seeds 42 43 --activations relu`. For the full-recipe
check, use the saved per-arm configuration and `rtml train --max-new-epochs 5` with
a fresh run directory. Keep the scheduler's 50-epoch setting unchanged.

## Detecting ineffective learning

The opt-in `learning_diagnostics` configuration records raw-input and model loss
on the **same actual training batches**, including augmentation. It also records
prediction changes, gradient norms, decoder zero fractions and the learning rate.
After five epochs with less than 2% reduction from raw-input loss, the C13 recipe
emits `little-improvement-over-input`. This is a diagnostic flag, not an automatic
restart, new stopping policy, or model qualification. `rtml status` exposes it.
CPU tests verify that observation preserves gradients/RNG and exact epoch resume.

## Correcting the HDR interpretation

The older evaluator's `supported_linear_mse` used the schema-1 full-diffuse test,
even for schema-2 models. That disagreed with the current model's center support,
partial-background handling and 128-sample bypass. New evaluations record
`region_metric_schema: 2`, use the actual model policy, and separate conditional
region MSE from each region's contribution to whole-image MSE. Historical reports
are retained with their original semantics. Every baseline uses the same custom
model's input-defined regions; this does not assert that OIDN has that policy.

An audit of all 224 saved C5 validation predictions found:

| Method | Model-region MSE contribution | Fallback-region MSE contribution | Whole-image MSE |
| --- | ---: | ---: | ---: |
| Raw | 0.0022160 | 0.2495483 | 0.2517643 |
| A-trous | 0.0015994 | 0.2481827 | 0.2497822 |
| Guided C5 | 0.0015646 | 0.2495483 | 0.2511129 |

These contributions have a common whole-image denominator and add across regions.
**99.38% of C5's HDR squared error is in its raw-fallback region.** Its conditional
model-region MSE is 0.0019082 versus a-trous 0.0019497, a small development-cohort
improvement. The full-image result still favors a-trous. Four validation layouts
and an unquantified regional reference-noise floor do not establish significance.

This points to two distinct next stages: improve learned reconstruction within its
supported domain, and evaluate a conventional fallback or additional real sampling
for unsupported regions. This change does not enable untrained glass/metal handling.
Independent reference checks from the current frozen generators saved scalar
disagreement only; their regional noise floor cannot be recovered from those
receipts. That qualification remains outstanding.

The audit can be reproduced from the original predictions without rerendering or
inference using [`audit_reconstruction_regions.py`](../tools/audit_reconstruction_regions.py).
It requires matching dataset/checkpoint/evaluation identities and a complete
training or validation split, and refuses a sealed test split.

## Verification and limits

The current renderer and native ONNX path passed 80 Python tests; two tests needing
the optional OpenImageIO environment were skipped. Native export tests exercise
nonzero weights, conditioning, activation, fallback and differing image sizes.
An initial test invocation selected an obsolete local renderer and failed four
renderer-compatibility checks; rerunning with both renderer environment variables
pointing to `build-detail/raytracer` passed. Two additional tests verify that the
ineffective-learning flag reaches progress reports and that summarized HDR region
contributions add to whole-image error. The failed log is retained locally.

Larger-data, resolution, three-seed finalist, sealed-test, temporal/flicker and
uncontended matched-quality timing gates remain necessary. Existing frozen renders
and experiments retain their original recipes and artifacts.
