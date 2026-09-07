# Neural reconstruction: issue audit and improvement plan

Status: **implementation and development experiments in progress; no replacement
model qualified yet**. See the [current study record](../ml/reports/detail-study.md).
The [research alignment audit](neural-reconstruction-research-audit.md) identifies
training-recipe gaps and additional controls discovered during implementation.
The baseline audit was performed on 2026-09-06 against
`cb53883bc487a522d1dce25d1c96e748491e4d04` and the recorded `pilot-v1` artifacts.
This is the next research iteration after the [original roadmap](neural-rendering-plan.md).

The immediate objective is to preserve corners, silhouettes, thin geometry and
surface detail while removing low-sample noise. A successful release must also
reach that quality sooner than the existing a-trous filter. Temporal reconstruction,
2× upscaling and additional materials have separate qualification gates.

This inventory covers the documented limitations and additional concerns found in
the current implementation. It is not a claim that every possible artifact has
been discovered. **Measured** means supported by saved results or this audit;
**implemented constraint** means directly established by code; **risk** means an
effect that requires an experiment before assigning its cause or severity.

## 1. What the evidence establishes

The [model card](../ml/reports/model_card.md), [dataset card](../ml/reports/dataset_card.md),
[per-image results](../ml/reports/pilot-test-per-image.jsonl.gz) and
[timing report](../ml/reports/timing.md) are the baseline records.

| Existing evidence | Implication |
| --- | --- |
| Across 360 test images, custom PSNR is 35.96 dB versus a-trous 33.97, but SSIM is 0.893 versus 0.944. | Lower average image error has not delivered comparable structural fidelity. SSIM alone does not locate or explain an edge artifact. |
| At 4 spp, custom PSNR/SSIM is 35.29 / 0.891; a-trous is 35.34 / 0.958. | The user's early-preview/detail concern is consistent with measured results at a useful preview budget. |
| OIDN achieves 39.44 dB / 0.971 across the same pilot test. | There is substantial quality headroom against an established learned baseline. This does not establish equal-runtime performance. |
| Median best passing native time is 100.34 ms for custom versus 78.94 ms for a-trous at PSNR ≥32 and SSIM ≥0.90. A-trous wins on all 20 views. | The custom model has not demonstrated acceleration over the existing filter. |
| Training used 32 layouts of one diffuse room family, 256×144 images and 1–32 spp. | Textures, complex assets, different scene families, specular transport, animation, larger resolutions and 64+ spp remain unqualified. |
| 2× and temporal models have only tiny smoke studies; both are weaker than a-trous in those studies. | Functional integration is complete; their claimed visual benefits remain unproven. |

The existing [four-panel example](../ml/reports/figures/pilot-4spp.png) shows residual
mottling on the wall and differences around objects and contact shadows. Its
[error image](../ml/reports/figures/pilot-4spp-error.png) is a useful illustration,
not a substitute for region-specific measurements on many scenes.

### Why edges need more than one fix

The [network](../ml/src/raytracer_ml/models/unet.py) reconstructs only pixels whose
diffuse support is at least `0.999999`. A pixel containing both a diffuse object and
background, glass, metal or an emitter returns its raw RGB. At 1 spp, one sampled
hit can make this decision appear certain; later samples can change it. This
creates a mechanism for noisy boundaries and changing fallback patterns.

An audit of all 60 existing 4-spp test inputs found 4,373 mixed-support pixels out
of 2,211,840 pixels (0.198%); another 9.170% had zero support. Saved predictions
equaled raw RGB exactly on every fallback pixel. This is a real, localized
limitation, **not evidence that fallback explains all softened corners**.

Where both sides of an edge are diffuse, support can remain one. The guides then
average several surfaces' albedo, normal and distance into one vector. Convolutions
can mix their colors: guides condition the network but do not constrain filtering
across a discontinuity. The network already has high-resolution skip connections;
its average pooling and bilinear resizing are plausible contributors to lost detail,
not proven sole causes. The [loss](../ml/src/raytracer_ml/losses.py) has no explicit
edge, corner, texture, temporal or clean-input preservation objective.

## 2. Complete issue register

| ID | Concern and evidence status | Remediation | Verification / stage |
| --- | --- | --- | --- |
| Q1 | Softened detail and bright edges: documented/visible; structural-score deficit measured. Rounded corners, thin-feature loss and halos need separate quantification. | Edge-focused data and crops, gradient/detail objectives, compare a high-resolution residual branch with learned filtering. | Edge profiles, corner localization, thin-feature recall and halo error; A–C. |
| Q2 | Noisy or changing silhouettes: hard mixed-support fallback is an implemented constraint. Visible popping from changing masks is a risk. | Record boundary/coverage information; train supported boundary handling; use a validated conventional fallback or additional real samples where uncertain. | Separate all-diffuse edges, mixed edges and unsupported interiors; sample-prefix sequences; B–C, F. |
| Q3 | Residual speckle, mottling and low-frequency illumination error: documented/visible. | Compare guide quality, variance inputs, receptive field and noise-aware losses; balance flat regions with detail crops. | Flat-region residual spectrum/variance, shadow gradients and full-image error across noise seeds; A–C. |
| Q4 | Lost bright features, highlight energy/color shifts, shadow/contact-detail loss: bright-edge error documented; other effects are risks. Log/relative loss downweights large absolute radiance errors. Output clamps log radiance to [0,12]. | Add HDR/light-size/exposure-range cases, region energy constraints and bounded normalization available at inference; test the clamp instead of removing it blindly. | Linear energy bias, bright-feature area/centroid, dark-region error, finite outputs and saturation counts; B–C. |
| Q5 | Inadequate scene diversity and resolution generalization: confirmed training restriction. | Expand geometry, textures, cameras and lighting in staged curricula; reserve truly new families/assets. | Separate camera, light, layout, asset/family and resolution cohorts; B, E. |
| Q6 | Glass, mirrors, rough metal, emitters and background remain raw; diffuse pixels can still contain unseen reflected/refracted light. Confirmed support policy and detection gap. | Retain conservative gating, add scene transport flags, then investigate material-aware/path-aware features and separate radiance components. | Full-image plus supported/fallback regions, glass-over-diffuse and mirror/color-bleed stress cases; E. |
| Q7 | Defocus, photon mapping and approximate glass shadows trigger whole-frame fallback. Confirmed restriction. | Keep these explicit until separately trained and evaluated; address renderer/reference bias first. Defocus requires lens-sampled guides. | Per-transport qualification, correct metadata and unchanged raw fallback; E. |
| Q8 | Predictions may pop as samples increase, preserve artifacts in clean inputs, or fail outside 1–32 spp. Unmeasured risks; 64-spp timing already extrapolates. | Train on higher budgets and clean/near-clean inputs; evaluate cumulative prefixes; use time-based update scheduling and validated confidence/blending. | Error-versus-budget, consecutive-preview change and clean-input distortion; C, F. |
| Q9 | No confidence calibration, adaptive sampling or learned stopping. Implemented absence; variance is unavailable at 1 spp and is not prediction confidence at any budget. | Calibrate risk on held-out development data, retain a sampling floor, route difficult pixels to additional real samples; evaluate stopping separately. | Risk/error calibration, false acceptance, true rays/time and rare-event stress; F. |
| Q10 | 2× model bilinearly enlarges both low-resolution correction and raw RGB. No learned high-resolution reconstruction head. Confirmed constraint. | Add a learned high-resolution head; test full-resolution geometry guides and correct pixel footprints. | Thin structures, corners, tiny emitters and aliasing versus ordinary upscaling at equal output size/time; H. |
| Q11 | Temporal training uses raw/a-trous history, validation uses a-trous, deployment uses previous predictions. Confirmed distribution mismatch. | Train short autoregressive sequences using model history and select using longer rollouts; introduce masked temporal losses. | Drift, flicker and detail retention across complete unseen sequences; G. |
| Q12 | Temporal reprojection uses nearest-neighbor lookup, center-ray positions with averaged guides, fixed tolerances and a translation-only explicit cut threshold. Ghosting/disocclusion/cut errors are risks, not demonstrated failures. | Geometry-aware interpolation, footprint-consistent history, scale-aware rejection, camera/FOV cuts, history age/confidence and neighborhood checks. | Reveal/hide edges, rotations, zooms, cuts, lighting changes, history resets and stationary sequences; G. |
| Q13 | Geometry/light changes reset all history; object motion and reflected-image motion are not modeled. Confirmed limitation. | Keep reset behavior as the safe baseline; add object motion and transport-aware history only after dynamic data and motion support exist. | Fast motion, moving lights/objects, reflected motion and lag after invalidation; E–G. |
| Q14 | References are noisy 512-spp estimates; convergence checks cover four views of one layout. Confirmed evidence gap. | Independent reference batches and region-level convergence checks across every challenge stratum; higher budgets where needed. | Reference disagreement below each acceptance margin; censor unresolved targets; B. |
| Q15 | One training seed, no completed feature/architecture/loss ablation matrix. Checkpoint selection uses global reconstruction loss. Confirmed research gap. | Controlled ablations and three-seed finalist runs; select under detail/HDR constraints on validation. | Group-level uncertainty, reproducible selection and retained failed experiments; C, I. |
| Q16 | Global mean PSNR/SSIM misses specific failures; first 12 saved comparisons are not a worst-case gallery. New external datasets fail the current checkpoint/manifest equality check. Confirmed evaluation limits. | Add region metrics, distributions, worst cases and explicitly registered external evaluation manifests with overlap checks. | Untouched challenge test, no removed failures or silently bypassed provenance checks; A, I. |
| Q17 | Custom inference loses to a-trous at matched quality; feature probes repeat primary intersections, and snapshots/packing allocate and copy. Timing deficit measured; each overhead's importance unprofiled. | Profile stages first; reuse first hits, preallocate/reduce copies, tune model size and execution provider. | Total matched-quality latency, including all extra feature work; D, I. |
| Q18 | Core ML has only a small parity probe, compiler warnings and no verified execution-device profile. Memory, large-size inference and GUI contention are unmeasured. | Fixed-size deployment trials, operator/device profiling, bounded tiling, CPU thread tuning and precision validation. | Warm/cold p50/p95, peak memory, tiled seams, provider parity and interactive latency; D. |
| Q19 | No equal-time curves, fair warmed native OIDN timing or complete interactive/flicker benchmark; coarse native budget grid misses 2/8 spp. Confirmed evidence gaps. | Add in-process baseline timing, full budget grid and progressive display measurements with equivalent resources. | Quality/time curves, all threshold failures, per-scene results and repeat variability; D, I. |
| Q20 | Pilot was generated by a dirty development tree; hashes exist but the recorded parent commit alone is insufficient. Confirmed provenance limitation. | Preserve pilot history; generate v2 from a clean commit with versioned schema, scene assets, binary/environment hashes and release recipes. | Clean regeneration, resume/compatibility tests and artifact audit; B, I. |

## 3. Execution plan and dependencies

### A. Freeze the baseline and build an artifact diagnosis suite

Deliverable: an issue report with raw, a-trous, custom-v1, OIDN and independent
reference crops at identical scale/exposure, paired with the metrics below.

- Retain the current weights and all pilot records. Pilot test data is now an
  inspected historical benchmark; use it for regression/development, not fresh
  evidence for v2 model selection.
- Define masks from independent reference geometry and scene annotations: straight
  edges, corners, silhouettes, material boundaries, thin features, contact shadows,
  highlights, dark areas, flat areas and fallback bands. Do not derive evaluation
  masks from the candidate's predictions. Separate texture, geometry and lighting
  edges so one type cannot conceal failures in another.
- Record linear/log error and fixed-exposure PSNR/SSIM, plus gradient error in
  1/2/4-pixel edge bands, edge width/location, corner displacement, thin-feature
  precision/recall, halo overshoot/undershoot and region luminance/color energy.
  Annotated test scenes define true geometry; account for pixel-filter antialiasing.
- Publish median, p95 and worst cases by scene and budget. Bootstrap **scene groups**,
  not correlated pixels, crops or noise variants; report sample counts and intervals.
- Extend external evaluation through an explicit dataset registration/compatibility
  path. Keep the training-manifest hash in the checkpoint and verify family/asset
  overlap rather than deleting the existing manifest check.

Start in `ml/src/raytracer_ml/{metrics,evaluate,benchmark}.py`; add challenge
annotations in `data/scenes.py` and fixtures under `ml/tests/`.

### B. Generate better data and boundary information

Yes: continue generating our own low-sample inputs and independent high-sample
targets with this renderer. Changing data coverage is necessary alongside changing
the network.

1. Start with four supported diffuse strata: corners/slanted edges/thin geometry;
   procedural texture and material boundaries; shadows/small lights/HDR contrast;
   varied clutter/scale/camera framing. Include acute/obtuse corners, low-contrast
   boundaries and subpixel features; a network must not sharpen correct antialiasing
   into jagged edges. Never use tone-mapped images as training targets.
2. Target 768 train / 128 validation / 128 sealed test layouts across eight scene
   families and four strata, with four views, two noise realizations, and
   1/2/4/8/16/32/64 spp. This is 4,096 references and 57,344 input examples before
   extra stress cohorts. Keep every
   layout's views, crops, sample budgets and noise variants together. Reserve
   camera-only and light-only cohorts separately and label their shared-layout
   relationship; reserve additional unseen families/assets for external testing.
3. Begin with a 16-layout development pilot and profile cost before expanding.
   Retain resumable generation and explicit per-invocation time/storage caps.
   The full proposed input count is about 24.9× the original pilot; new channels,
   reference batches and larger sizes add further cost. Shard generation and set
   total resource limits from measured pilot bytes/time, not an assumed render rate.
4. Validate reference quality with independent batches across every stratum, using
   512→2048→8192 spp as an initial escalation ladder. Set region tolerances on the
   development pilot, below the proposed model improvement margin. Continue only
   within the configured cap; flag unconverged targets and exclude them from claims
   requiring tighter ground truth. Check depth truncation separately from sampling
   variance. Archive the exclusions and remaining noise floor.
5. Keep sample-aligned anti-aliased albedo/normal/coverage. Averaged normals are
   intentional, not automatically a bug to fix by renormalizing. Add separate
   nearest/center depth and boundary indicators, depth/normal spread and material
   class/coverage. Compare richer two-layer boundary features only if their measured
   quality benefit pays for the visibility rays and memory. Object IDs should form
   equality/boundary masks, not arbitrary ordered numeric features.
6. Version the feature schema and model metadata. Use only features available at
   runtime; high-sample geometry may define evaluation masks but must not leak into
   low-sample inputs without counting its acquisition cost. Give temporal positions
   an explicit footprint/coverage contract.
   Test depth/world-scale and radiance normalization across the declared camera,
   scene-unit and light ranges; the existing fixed depth/sample transforms are not
   evidence of scale invariance. Store fitted statistics from training data only.
7. Extend the scene-file generator/parser for procedural textures and meshes before
   claiming those domains are covered. The conventional OBJ importer does not make
   mesh objects available automatically in the current JSON dataset generator.

Guide alignment and auxiliary noise deserve their own ablation. OIDN's primary
documentation requires matching color/auxiliary pixel filters and cautions against
marking noisy guides as clean. Prefiltering has a cost that must be counted.
[OIDN RT documentation](https://www.openimagedenoise.org/documentation.html#rt)

Primary changes: `RayTracer/{renderer,feature_buffers,scene_file}.cpp`,
`ml/src/raytracer_ml/data/{scenes,generate,dataset,validate}.py`, preprocessing,
schema/export metadata and new data configs. Preserve raw-image/RNG invariance.

### C. Train and select a detail-preserving spatial model

Do not select a larger network or add a sharpening filter on appearance alone.
The implemented initial screen contains ten runs. The table records their actual
comparisons; the research audit adds one-factor controls and an upstream OIDN
training baseline before finalists are selected:

| Run | Change from comparison parent | Question |
| --- | --- | --- |
| C0 | Current architecture/loss trained on v2 data | How much does data alone fix? |
| C0a | C0 + paired augmentation and independent-seed fusion | Does physically consistent data reuse help? |
| C1 | C0a + balanced edge/detail crop sampling and edge-weighted gradient loss | Does explicitly supervising structure improve corners without amplifying noise? |
| C2 | C1 + high-sample identity and HDR region-energy terms | Are detail, clean-input behavior and bright regions preserved together? |
| C3 | C2 + boundary channels, changed head, support and blending | Does this combined schema-2 design help? Separate controls are required to isolate guides. |
| C4 | C3 + full-resolution residual refinement branch | Does learned high-resolution processing outperform existing skip connections at acceptable cost? |
| C5 | C3 with a compact geometry-conditioned kernel-prediction head | Does learning how to filter measured radiance give a better fidelity/time tradeoff? |
| C6 | Best C3–C5 candidate with RGB-only inputs | How much do guides actually contribute? |
| C7 | Same candidate without variance/availability inputs | Do low-sample statistics help or propagate noise? |
| C8 | C3 with width 64 | Does greater capacity help under this particular recipe? |

Keep one-factor parent comparisons explicit; publish parameter count, training
compute and inference cost. These runs screen designs, not every possible
hyperparameter. Promote the two strongest feasible candidates to three seeds each
on the full dataset; do not choose the best seed after inspecting test results.

The loss should retain scene-linear/log reconstruction, with bounded edge weights,
multi-scale derivative terms, high-sample identity pairs and highlight/shadow energy
terms. Derive coefficients and sampling proportions on development/validation only.
Include flat-region error so training does not merely exchange blur for speckle.
Avoid adversarial/detail-generation losses for this reference-faithful task.

Kernel prediction is a research alternative, not a guaranteed cure: learned local
weights can still blur or spread a firefly. Compare constrained weights and a small
residual branch, especially where valid radiance is absent in the neighborhood.
[Bako et al., SIGGRAPH 2017](https://la.disneyresearch.com/publication/deep-learning-denoising/)
provides the primary reference for predicting local denoising kernels.

Train coverage-aware reconstruction before relaxing the current support threshold.
Otherwise a mask change would simply enable untrained predictions on glass,
background and mixed pixels. Validate a boundary fallback compositor against raw
and a-trous; do not smear colors across surfaces to hide a seam.

Checkpoint selection must satisfy validation constraints on edges, HDR and clean
inputs before comparing the quality/time frontier. Maintain full raw accumulation;
predictions are never labels for new measurements or accumulated samples.

Primary changes: `models/unet.py`, optional new model module, `losses.py`,
`data/dataset.py`, `train.py`, training configs and native/export compatibility.

### D. Make deployment fast enough to justify reconstruction

Instrumentation can start in A; optimize the qualified spatial candidates from C.

- Measure scene/BVH setup, primary/secondary tracing, guides, snapshots, packing,
  transfers, synchronized inference, compositing, display upload and file output
  separately. Measure first useful displayed preview and every subsequent update
  while the tracer is running. Record CPU/GPU memory, cold load/compile, warm p50/p95
  and sustained-run contention with the same tracing-worker budget.
- Eliminate duplicated primary intersections only after profiling: return the first
  hit from the path tracer for guide collection without changing RNG consumption or
  raw radiance. Reuse center probes where equivalent. Preallocate tensors and reduce
  snapshot/float-conversion copies without breaking immutable worker ownership.
- Compare ORT CPU thread counts and fixed-size Core ML exports at 256×144 and
  512×288 before larger outputs. Profile operator placement and CPU fallback; choosing
  `ALL` compute units does not prove Neural Engine execution. Evaluate model caching
  and its cold-start semantics. Dynamic shapes can cost performance according to
  [ORT's Core ML documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html).
- Test FP16 only against region/HDR and native parity gates, including large and
  small radiance. Keep FP32 fallback. Tiling must use receptive-field halos and
  tested crop alignment to avoid seams; report its real memory/latency tradeoff.
- Add optional in-process OIDN timing with declared device, guide prefiltering and
  equivalent warm/cold scopes. Keep custom training as its own research track.
- Sweep 1/2/4/8/16/32/64 spp, extending raw to 128/256 within a declared cap when it
  misses quality targets. Include model updates, all guide work and tracing
  contention in total time. Publish failed thresholds rather than invented speedups.

Primary changes: `RayTracer/{integrator,renderer,neural_denoiser,main}.cpp`,
`ml/src/raytracer_ml/{export,benchmark}.py` and native tests.

### E. Expand supported materials and transport deliberately

After diffuse detail is qualified, create separate cohorts for textured/mesh scenes,
rough/glossy metal, mirrors/glass, then caustics and defocus. Vary light size, dynamic
range, indirect illumination and environment strength independently of geometry.
Include a diffuse receiver beside glass/metal to expose indirect-domain leakage.

Investigate roughness/material descriptors and diffuse/specular or direct/indirect
radiance components, with consistent recombination and all extra runtime costs
counted. Perfect-specular paths may benefit from guides at subsequent non-delta
hits; this needs a separate feature contract and test rather than reusing ordinary
surface motion blindly. OIDN describes such auxiliary-guide handling in its
[RT material guidance](https://www.openimagedenoise.org/documentation.html#rt).

Keep unqualified modes visibly on raw/conventional rendering. A model trained on a
renderer approximation cannot establish physical correctness beyond its targets:
finite-depth bias, photon-map radius bias/leakage and straight-line glass-shadow
approximations require renderer/sampler work or a separately validated reference
integrator. Image-texture import, participating media and environment importance
sampling are renderer capabilities, not defects that more model training resolves.
Use independent seeds and converged references for tiny lights and rare paths.

### F. Stabilize progressive previews and spend rays where they help

First evaluate static cumulative sequences at 1/2/4/8/16/32/64/128 spp: each snapshot
shares earlier samples. Measure reference-relative error changes, fallback-mask
changes and visible updates. More samples need not improve every stochastic frame,
but systematic clean-input degradation or abrupt model artifacts are unacceptable.

Add a wall-time inference throttle with latest-snapshot scheduling and input-sample
metadata. Keep existing stale-generation rejection and bounded pending work. Any
display blending must reset on scene/model changes and be evaluated for lag; it
must not conceal a worse reconstruction or alter raw samples.

Calibrate a lightweight risk predictor or heuristic against validation residuals,
using sample count, valid variance, coverage and model disagreement only as inputs
available at runtime. Measure high-confidence failures and out-of-domain behavior.
An unsampled rare light path may produce low variance: uncertainty cannot prove its
absence. Keep a minimum real-sampling budget and periodic exploratory samples.

Then prototype tile/pixel adaptive sampling and stopping as a separate experiment.
This requires renderer scheduling, per-pixel counts/variance, training with uneven
budgets and correct metadata; it is not a model-only toggle. Measure selection bias
with independent sampling streams and preserve a uniform reference mode. Compare
at equal total rays and wall time, including calibration/inference overhead. Prefer
more real samples where the model cannot reliably recover detail.

### G. Train temporal reconstruction on its actual deployment history

Generate dense trajectories, not just the pilot's four sparse views: stationary
frames, slow/fast pans, rotations, zoom/FOV changes, occlusion/disocclusion, cuts,
moving geometry and light changes as supported by E. Split complete sequences and
parent scenes before generating noise. Start with camera motion in diffuse scenes.

Train 4–8-frame unrolls using previous model predictions, with explicit first-frame,
invalid-history and dropped-history examples; progressively reduce auxiliary
teacher history. Validate on independent 32–64-frame autoregressive rollouts to
expose drift. Apply reference-motion-compensated temporal loss only where geometry
and history are valid; retain spatial detail/HDR objectives.

Add geometry-aware subpixel history lookup, coverage-aware position matching,
scale-aware depth tests, rotation/FOV cut detection and history confidence/age.
Validate neighborhood clamping for ghost suppression without erasing bright motion.
Keep global resets for unsupported moving-light/object cases until motion support
is qualified. First-hit motion is insufficient for all reflected/refracted images.

Report motion-compensated reference residual, temporal variance, edge shimmer,
ghost-trail duration and recovery after invalidation, including first frames and
disocclusions. Compare spatial-only, non-learned temporal filtering and learned
history at equal total time. Recurrent Monte Carlo denoising is established research;
its published results do not validate our model without these sequence experiments.
[Chaitanya et al., SIGGRAPH 2017](https://research.nvidia.com/publication/2017-07_interactive-reconstruction-monte-carlo-image-sequences-using-recurrent)

Primary changes: `data/dataset.py`, `temporal.py`, model/training/evaluation and
the mirrored native history implementation in `RayTracer/neural_denoiser.cpp`.

### H. Replace interpolation-only upscaling with learned reconstruction

Use the qualified spatial model as the starting point. Compare a learned subpixel
output head and a high-resolution refinement head, optionally conditioned on
full-resolution geometry/coverage. Align low/high-resolution jitter and pixel
footprints explicitly; prohibit high-sample target leakage into inference guides.

Compare raw bilinear/bicubic, a-trous/OIDN then upscale, and full-resolution
low-sample reconstruction at identical output sizes, wall-time budgets and hardware.
Count high-resolution visibility rays, transfers and memory. Test 0.5/1/2-pixel
features, diagonal edges, corners, texture frequencies, tiny emitters and odd sizes.
If a feature is absent from the measurements and guides, reliable recovery may
require extra real samples. Visual plausibility does not qualify invented detail.

Qualify spatial upscaling independently before combining it with temporal history;
the current implementation explicitly disallows the temporal + 2× combination.

### I. Release only after the new evidence passes

Freeze thresholds and selection on validation before opening the new test set.
These are **proposed acceptance targets**, not measured accomplishments:

| Gate | Proposed requirement |
| --- | --- |
| Structural improvement | At 4/8 spp in the supported domain, reduce mean annotated edge-gradient error ≥20% against v1; mean SSIM at least a-trous. Scene-group intervals must rule out regression larger than 0.005 SSIM or 5% edge error against a-trous. Report thin-feature/corner/halo metrics separately and reject systematic missing geometry. |
| HDR and broad quality | No >5% cohort regression in linear MSE or absolute region-energy error versus v1, allowing only predeclared reference-noise tolerances. No NaNs, infinities or unexplained clipping. Bright/dark cohorts must each pass. |
| Clean and progressive inputs | Near-clean/128-spp inputs must not show systematic degradation beyond independently measured reference disagreement; report every sampled-prefix regression. No persistent boundary switching or stale preview after invalidation. |
| Speed | Beat a-trous median time to the joint structural/HDR quality target with equal or better scene pass rate; p95 no worse by >5%. A 2× gain remains a stretch target. Do not infer speed from Python-only inference time. |
| Temporal | Reduce motion-compensated residual/flicker ≥20% versus the qualified spatial model without >5% spatial-edge regression; invalidate history on declared cuts/changes and reject visible persistent ghost trails. Test longer rollouts, not only adjacent pairs. |
| 2× | Beat the best measured interpolation/denoise-upscale quality/time frontier while passing the same thin-feature, edge and HDR constraints at output resolution. |
| Reliability | Three-seed finalists, scene-level intervals, full failures, native/provider parity, raw invariance, memory limits and model/schema fallback tests pass. No supported-domain expansion without its own qualified cohort. |

Finalize metric definitions, tolerances and sample sizes in A/B before training
selection. If the reference noise floor or scene count cannot resolve a margin,
increase measurement quality or report an inconclusive result; do not relax a gate
after seeing test performance. Targets may be refined on development evidence, with
the reason and pre-test revision recorded.

Publish the new model/dataset cards, issue-by-issue outcomes, training curves,
per-scene metrics, labeled detail crops and worst cases, temporal videos, time/quality
curves, memory and generation/training cost. Preserve v1 for comparison and rollback.
If quality improves but speed does not, label the model a quality experiment and
retain a-trous as the faster qualified option. Do not equate a completed pipeline
with a successful acceleration result.

## 4. Work order and completion boundary

| Milestone | Dependencies | Reviewable deliverable |
| --- | --- | --- |
| 1. Diagnose | A | Reproducible edge/detail and fallback report, registered metrics and new split policy |
| 2. Fix spatial fidelity | B → C; D instrumentation begins alongside these | Converged challenge targets, v2 features and controlled spatial ablations |
| 3. Qualify useful speed | C → D → spatial portion of I | Native candidate with measured quality/time advantage, or an explicit failed speed hypothesis |
| 4. Broaden robustness | E and F on qualified spatial work | Per-domain material results, stable progressive previews and separately measured adaptive sampling |
| 5. Qualify motion and upscaling | G and H after spatial gates; combined mode last | Sequence/upscale models with their own quality, stability, latency and memory evidence |
| 6. Publish | All applicable I gates | Weights, schemas, reproducible artifacts, labeled results and remaining exclusions |

Each milestone ends with a report and a stop/go decision before expanding compute.
No universal artifact-free guarantee is possible from sparse samples. The practical
resolution for an unqualified case is a declared fallback or more real measurements,
alongside a documented experiment to expand support. This planning change does not
replace the current weights, change rendering behavior or claim any fix has landed.

## Appendix: reproduce the fallback audit

With the original local pilot dataset and saved test predictions present, run from
the repository root. This reads existing arrays; it does not render or train.

```sh
ml/.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('artifacts/datasets/pilot-v1')
with (root / 'manifest.jsonl').open() as stream:
    rows = [r for r in map(json.loads, stream) if r['split'] == 'test' and r['samples'] == 4]
pixels = mixed = zero = 0
maximum = 0.0
for row in rows:
    with np.load(root / row['path'], allow_pickle=False) as data:
        x = data['features']
    path = Path('artifacts/evaluations/pilot-test/predictions') / (row['id'] + '.npz')
    with np.load(path, allow_pickle=False) as data:
        prediction = data['prediction']
    support = x[11]
    fallback = support < 0.999999
    pixels += support.size
    mixed += int(((support > 0) & fallback).sum())
    zero += int((support == 0).sum())
    if fallback.any():
        raw = x[:3].transpose(1, 2, 0)
        maximum = max(maximum, float(np.max(np.abs(prediction[fallback] - raw[fallback]))))
print(dict(images=len(rows), pixels=pixels, mixed_support=mixed,
           zero_support=zero, max_abs_fallback_minus_raw=maximum))
PY
```
