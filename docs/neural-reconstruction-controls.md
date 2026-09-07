# Versioned reconstruction corrections and bounded controls

This implementation follows the [literature audit](neural-reconstruction-literature-review.md).
It removes two proven representational/integration constraints and makes the guide
and loss hypotheses testable. Existing checkpoint defaults and the released model
remain unchanged. A code correction is not evidence that newly trained weights are
ready for release.

## Model contract

All options are saved in checkpoint configuration, the ONNX sidecar, and ONNX
`rt_model_config` metadata. Native inference executes the exported graph.

| Option | Legacy default | Explicit alternative |
| --- | --- | --- |
| DetailNet `output_head` | `bounded_multiplicative` | `additive_log` for refinement and/or the learned 2× correction |
| DetailNet `radiance_scale` | `1` | Internal radiance conditioning, e.g. `16`, with variance multiplied by its square; requires an additive head |
| DetailNet `blend_policy` | `legacy` | `single`: apply sample attenuation once, after the complete 2× reconstruction |
| Guided `guide_policy` | `center` | `sampled`: use sampled mean normals and depth in fixed filtering penalties |
| Loss `kind` | `log_l1` | `relative_l2`: squared error in linear radiance, divided by `(prediction.detach() + 0.01)^2` |
| Learning diagnostics `warning_metric` | `whole_image_loss` | `model_region_log_mae`: assess improvement only where reconstruction is enabled |

The additive correction is `expm1(log1p(scale * input) + correction) / scale`,
clamped in log space to nonnegative output and the existing physical radiance
ceiling `expm1(12)`. A black input can now become brighter than the legacy
approximately 0.064 limit. This is an additive correction in **log radiance**, not
an unconstrained linear residual. Unsupported pixels and pixels at 128 or more
samples retain exact raw output. At 2×, that raw output is bilinear reconstruction
of the input. These policies remain conservative domain restrictions, not calibrated
confidence or proof of convergence.

`single` leaves spatial blending unchanged. For 2×, the low-resolution prediction
is enlarged and corrected before one final blend with enlarged raw RGB. Sample
strength and support are enlarged by nearest neighbor. This specifies behavior
when neighboring pixels have different sample counts; it does not interpolate a
previously attenuated image. A neutral upscaling head now retains the same
correction as the matching spatial model at uniform sample counts.

The sampled guide control changes **only the deterministic penalty inputs**.
The encoder retains all 27 channels, including center guides. Sampled normals
are the unnormalized average, and depth is the per-sample mean with misses
contributing zero; neither is divided by coverage. Albedo is the existing sampled
buffer. This is a measured footprint control, not an assertion that noisy moment
buffers are optimal. Prefiltering and layered/first-non-delta guides remain separate
experiments. [OIDN auxiliary-buffer guidance](https://www.openimagedenoise.org/documentation.html)

The relative L2 control follows the stopped-denominator objective in section 3.3
of [Noise2Noise](https://proceedings.mlr.press/v80/lehtinen18a/lehtinen18a.pdf).
Targets stay in linear radiance. Its denominator must not receive gradients.
Near zero it can produce large gradients; the existing finite checks and gradient
clipping remain active. It is an experimental alternative, not a universal default.
The guide/loss comparison omits gradient and energy terms in **all** arms so the
loss-kind comparison has one changed factor. Adding those terms would invalidate
an interpretation based on the pure squared-error objective alone.

Learning-health schema 2 records model-region log MAE and linear MSE for prediction
and raw, plus model-region and bypass pixel observations. Region errors use a
pixel-weighted denominator across the observed batches. Whole-image configured
loss is retained. The warning metric is explicit to preserve historical experiment
semantics. Reports observe existing gradients without changing optimization or RNG
state. Counts refer to observations, including repeated/augmented samples, not
unique dataset pixels.

## Bounded study

[Preregistered configuration](../ml/configs/research/literature-controls.yaml)
selects the first source view in each training/validation stratum before any errors
are inspected. For the current pilot this gives four training and four validation
layouts: edges, textures, lighting and clutter. Test views are excluded.

A separate 128×72 cohort is rendered with the **same frozen renderer binary** as the
source collection. It has 4/16-sample inputs, two independent noise streams per
budget, 4,096-sample primary references, and independently seeded 8,192-sample
checks. The new resolution requires freshly rendered inputs as well as references.
All arrays, seeds, scene identities, renderer/source hashes, process receipts and
reference-region disagreement are retained. Original datasets are not edited.
Checksum, scene/split, seed, sample coverage and reference metric checks run before
fitting. Each stage has a 1,500-second cap; rendering also has a 2 GiB write cap and
an 8 GiB free-space guard. Interrupted artifacts remain diagnostic and cannot be
loaded as complete cohorts. Use a new output directory for a new attempt.

Seven recipes run with seeds 42 and 43, 200 AdamW updates each, batch 4, crop 64,
learning rate 0.0003, weight decay 0.0001 and gradient clipping at 1:

- Guided center/sampled penalties × log L1/relative L2 (four recipes).
- Refinement with the legacy head, additive head at scale 1, and additive head with
  internal radiance scale 16 (three recipes).

Each seed uses the same precomputed noisy-example/crop schedule in every recipe.
Within an architecture, initial parameter tensors match. Training uses primary
references only. There is no validation checkpoint search, early stopping,
augmentation, fusion or scheduler in this optimization diagnostic; the final
200-update checkpoint is retained. Validation uses all 16 full images and scores
raw, a-trous and neural against **both** references. Region masks are fixed from the
primary reference or the actual input support policy and reused for every method
and both reference comparisons. The independent check never enters training.

```sh
ml/.venv/bin/python ml/tools/run_research_controls.py render \
  --config ml/configs/research/literature-controls.yaml \
  --data "/Volumes/Zach's SSD/RayTracer/datasets/detail-pilot-v2" \
  --renderer "/Volumes/Zach's SSD/RayTracer/bin/raytracer-data-v2" \
  --output "/Volumes/Zach's SSD/RayTracer/datasets/literature-cohort-v1"
ml/.venv/bin/python ml/tools/run_research_controls.py compare \
  --config ml/configs/research/literature-controls.yaml \
  --data "/Volumes/Zach's SSD/RayTracer/datasets/literature-cohort-v1" \
  --output "/Volumes/Zach's SSD/RayTracer/runs/literature-controls-v1"
```

The [completed results and labeled comparison images](../ml/reports/research-corrections.md)
record all 14 fits. Conditioned refinement improved; the sampled-guide and relative-L2
alternatives did not justify changing defaults in this short study. Large sample counts
alone do not qualify a reference: report regional disagreement and whether each
observed improvement is stable against the independent check. Four validation
layouts, two optimization seeds and 200 updates cannot establish generalization,
training convergence or a runtime advantage. Full-data finalists, independent scene
families, dynamic temporal behavior, unsupported transport, and matched-quality
end-to-end timing remain required before promotion.
