# Literature cross-check of the reconstruction models

Reviewed online on 2026-09-06 against project commit
`cb501c4cbe0ece4865e0be4ce0c05c10759a3252`. This follow-up examines the implementation
after the [checkpoint/evaluation corrections](../ml/reports/qualification-fixes.md).
Primary papers and upstream documentation are linked alongside the relevant claims.

**The task formulation is sound. The present implementation still has specific
limitations that more data or a larger network alone cannot repair.** Two further
implementation problems were reproduced: double attenuation in the 2× path and
the refinement head's remaining missing-signal cap. Additional concerns are
clearly identified as design limitations or conditional research risks below.
These findings do not justify discarding the existing diffuse datasets.

## Evidence and priorities

| Priority | Finding | Status | Action before a larger training commitment |
| --- | --- | --- | --- |
| 1 | The detail model applies sample-dependent blending both before and after its 2× head. | Confirmed behavior; likely unintended double attenuation. | Introduce a versioned correction that blends once, preserving old checkpoint semantics. Verify zero-head scale-1/scale-2 consistency and native export. |
| 1 | `kind: refine` still uses the bounded multiplicative output, although C13's U-Net received an additive head. | Confirmed representational limitation. | Add the same explicit head/conditioning control to refinement, compare matched initialization and support, then run a bounded fit test. |
| 1 | Learned output is discarded in unsupported pixels; their own reconstruction loss provides no parameter gradient. | Confirmed policy limitation. | Treat unsupported materials, mixed visibility and primary emission separately. Validate a fallback or new representation before expanding support. |
| 1 | Log/L1 and target-relative losses lack an unbiased-mean guarantee when references remain noisy. | Conditional statistical risk; current dataset bias not measured. | Measure retained regional reference disagreement, then compare the current objective with a suitable linear-output noisy-target objective on a fixed development subset. |
| 2 | Hard guided weights combine sampled albedo with center-ray normals/depth, while RGB is integrated across pixel samples. | Confirmed footprint difference; contribution to artifacts is a hypothesis. | Compare matched-footprint, prefiltered and layered guides without changing model capacity or sampling. |
| 2 | The default guided model is a small constrained local filter, rather than a reproduction of a published kernel network. | Confirmed architecture limitation, not proof that the model is too small. | Compare receptive field, prior strength and signal decomposition before another width-only sweep. |
| 2 | Fixed sample count controls blending and the 128-spp raw bypass. | Confirmed heuristic; sample count does not establish convergence. | Measure progressive prefixes and risk calibration; retain extra real sampling for unresolved regions. |
| 2 | Global learning-health loss can hide successful learning in the supported region. | Confirmed diagnostic counterexample. | Add support-normalized measurements and effective supported-pixel counts; retain full-image quality reporting. |
| 3 | Procedural diffuse families and camera sequences do not cover arbitrary assets, transport or moving objects. | Confirmed coverage limit. | Keep family/asset/transport/resolution and motion tests separate, and compare three-seed finalists. |
| 3 | Quality metrics do not establish a total rendering speedup. | Confirmed evidence gap. | Compare equal-quality wall time, feature generation, transfers, reconstruction, display updates and peak memory under equivalent resources. |

## What the literature changes about our interpretation

### 1. The guided prior is useful, but our constraints are substantial

[Bako et al. (2017), sections 3–5 and 7](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Bako22.pdf)
uses separate diffuse/specular processing, diffuse albedo demodulation and learned
kernel prediction. Their reported networks use nine convolutional layers, with
100 filters in hidden layers, and 21×21 output kernels. They also found L1 useful
with their reference data. Their results do not establish that every denoiser
must adopt that architecture or that L1 is universally wrong.

Our [guided model](../ml/src/raytracer_ml/models/detail.py) has three 3×3 encoder
convolutions and three sequential nine-tap filtering stages by default. With
dilations 1/2/4, the filtered radiance comes from a radius-seven neighborhood.
This is a statement about which **radiance samples can contribute**, not the
complete receptive field of every predicted weight. Without demodulation,
temporal history or upscaling, positive normalized filtering cannot recover a
nonzero value when every contributing RGB sample is zero.

Its learned logits are capped at ±5, while fixed albedo/normal/depth penalties can
reach 60. Two learned logits can differ by at most 10, limiting the ability to
counteract a mistaken geometry penalty. This constraint can protect edges or
prevent useful filtering; its effect requires an isolated ablation. The default
`demodulate` setting is false, and no diffuse/specular component buffers exist.

### 2. Guide alignment deserves a controlled test

[OIDN's input documentation](https://www.openimagedenoise.org/documentation.html)
requires auxiliary images to use the beauty image's pixel reconstruction filter.
It distinguishes noisy guides from clean or prefiltered guides and warns that
incompatible anti-aliasing can harm edges. It also describes guides beyond perfect
specular hits to improve reflections and transmission.

Our OIDN path correctly exports sampled albedo/normals and declares them noisy.
The concern is the custom guided model: its fixed penalties use center-ray depth
and normals with sampled RGB/albedo. One center surface may not describe the full
pixel footprint near silhouettes or overlapping geometry. Additional center
channels are not inherently invalid neural inputs; hard filtering decisions based
on them need the controlled comparison above.

[Gharbi et al. (2019)](https://groups.csail.mit.edu/graphics/rendernet/)
shows why per-pixel summaries lose information available in individual samples.
[Munkberg and Hasselgren (2020)](https://research.nvidia.com/labs/rtr/publication/munkberg2020layerdenoise/)
addresses complex visibility using sample layers and learned compositing.
Those are useful directions for mixed boundaries, with additional storage and
runtime costs. They do not prove that replacing our guides will fix all corners.

### 3. Noisy references change the loss question

[Noise2Noise, sections 2 and 3.3](https://proceedings.mlr.press/v80/lehtinen18a/lehtinen18a.pdf)
distinguishes mean-seeking L2 from median-seeking L1. Its Monte Carlo discussion
explains bias from nonlinear noisy-target transforms and uses linear predictions
with output-normalized squared error, stopping gradients through the denominator.
Independent seeds alone are insufficient for arbitrary objectives.

Our [loss](../ml/src/raytracer_ml/losses.py) combines log L1, target-normalized linear
L1 and optional gradient/energy terms. This is a reasonable perceptual objective
to test with sufficiently converged references; it has no general guarantee of
recovering expected linear radiance from unconverged references. A synthetic
target distribution `[0, 0, 0, 4]` has mean 1. At prediction 1, our detail loss has
a positive derivative, pushing the estimate downward. That demonstrates the risk,
not its magnitude in the current 1024/2048-spp collections.

Switching every run to plain MSE would be unjustified: large outliers can dominate
optimization. First measure reference uncertainty, then compare objectives and
highlight/energy errors with identical scenes, updates and seeds. The literature
does not prescribe a universally sufficient reference sample count or dataset size.

### 4. Some errors require better signals or more samples

[NRD's integration guidance](https://github.com/NVIDIA-RTX/NRD#noisy-inputs)
separates primary emission from denoised illumination and describes diffuse/specular
signals. It warns about sparse high-energy outliers and inadequate input convergence.
NRD is a conventional denoising integration reference, not a training recipe for
our network; its specific interface requirements are not universal model rules.

Our stored image combines signal types, and unsupported pixels return raw RGB.
The previous [HDR audit](../ml/reports/learning-diagnosis.md) found 99.38% of C5's
whole-image HDR error in that fallback region. Training the current supported
branch for more epochs cannot alter pixels that are explicitly replaced by raw
output. Component separation, boundary representation, emission handling and
additional real samples need independent comparisons. Blindly enabling the model
on glass or blurring primary emission is not a validated correction.

[Hasselgren et al. (2020)](https://research.nvidia.com/labs/rtr/publication/hasselgren2020neuraltemp/)
jointly optimizes temporal sampling and denoising, including extra samples for
disocclusions and specular features. This supports investigating where to spend
rays, rather than assuming a larger image network can resolve every missing event.
Our current fixed 128-spp bypass remains a policy, not a confidence estimate.

### 5. Temporal training now addresses one important mismatch

[Chaitanya et al. (2017)](https://research.nvidia.com/publication/2017-07_interactive-reconstruction-monte-carlo-image-sequences-using-recurrent)
uses recurrent reconstruction and temporal training for low-sample sequences.
Our newer four-frame training rollouts use previous predictions, and validation
runs for 32 frames. This addresses the earlier mismatch between training histories
and deployment histories. The [baseline audit](../ml/reports/qualification-fixes.md)
also measures comparable reference-corrected changes.

The current sequences move cameras and include declared camera/light changes;
they do not move objects. Resetting all history on a scene change is an explicit
limit. The reported 25.8% temporal-error improvement on four development layouts
does not qualify object motion, reflected motion, progressive sample updates or
persistent ghost trails. Those claims need their own data and measurements.

## Reproduced implementation findings

The [probe record](../ml/reports/detail-development/literature-audit-probes.json)
contains the inspected commit, source hashes, inputs and numeric observations.
These are small synthetic math/implementation checks, not trained quality results.
The [probe script](../ml/tools/probe_reconstruction_constraints.py) reproduced all
six probe groups exactly on CPU. Run it against the audited model implementation:

```sh
ml/.venv/bin/python ml/tools/probe_reconstruction_constraints.py --output artifacts/new-literature-probes.json
```

It refuses existing output files. Its assertions intentionally capture the audited
constraints; corrected model variants require a new comparison, not a rewritten
historical result.

| Probe | Observation | Implication |
| --- | --- | --- |
| Refinement model, supported raw RGB 0, head bias 100, 4 spp | Maximum output 0.06389056, matching `0.01 * (exp(2) - 1)` | The additive-head fix for `kind: unet` did not remove the refinement head's ceiling. |
| Matching spatial/2× guided weights and a neutral learned upscaling head | At 64 spp, 2× retains 2/3 of the corresponding spatial correction; at 96 spp, only 1/3 | The spatial filtering correction is attenuated twice. At 96 spp its net factor is 1/9, rather than 1/3 relative to full-strength filtering. |
| Three-stage guided model, all contributing RGB zero; bright sample eight pixels away | Output stays exactly zero; moving that sample to distance seven permits a nonzero output | More training cannot overcome the finite radiance footprint. |
| Loss only on unsupported pixels | Output equals raw, total parameter-gradient magnitude 0 | Adding examples alone cannot teach an excluded material to this branch. |
| One perfectly reconstructed supported pixel, 63 unchanged erroneous fallback pixels | Supported error improves 100%; global loss improves only 0.08065% | The 2% global health threshold can warn despite successful local learning. |

For the 2× check, let `s = clamp((128 - spp) / 96, 0, 1)` and let `D` be the
full-strength guided prediction. With the upscaling head neutral, the spatial path
returns `raw + s*(D - raw)`. The current 2× path returns, up to floating-point
rounding, `upsample(raw) + s*s*upsample(D - raw)`. The duplicated blend is visible
at both blend sites in [detail.py](../ml/src/raytracer_ml/models/detail.py).

## What to do next

1. Make the two deterministic head/blending corrections in explicitly versioned
   model variants. Preserve historical weights and verify Python/ONNX/native parity.
2. Add supported-region learning telemetry and run bounded fitting controls before
   interpreting a global stalled-learning flag as an optimizer failure.
3. Build a small retained, independently converged reference cohort. Compare guide
   footprints, loss/normalization and decomposition separately on this cohort.
4. Scale the winning recipe across the larger collection, with held-out families,
   assets, transport and resolutions; then qualify temporal/progressive behavior
   and total time at matched quality.

Paired renders, independent noise streams, layout-separated splits, physical
radiance/variance augmentation, autoregressive history, native parity and the new
eligible-first checkpoint rules are useful foundations already in place. Neither
a transformer nor a particular parameter count is established as necessary by
this review. The immediate evidence favors correcting known constraints and
testing representation/objective choices before another width-only expansion.

This review adds findings and experiment priorities. It does not change runtime
models, active frozen generation jobs, historical checkpoints or release weights.
