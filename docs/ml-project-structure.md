# ML project structure

Status: implementation blueprint. The paths and entry points below are proposed;
the dataset generator, training package, and learned model do not exist yet.
The [AI-assisted ray tracing plan](neural-rendering-plan.md) defines the research
scope, renderer behavior, and later upscaling/temporal stages.

## Research contract

**Question:** can our trained reconstruction model reach a specified image quality
sooner than the current renderer with its conventional denoiser?

**Inputs:** low-sample linear RGB, geometry guides, masks, sample count, and sample
statistics. **Target:** independent high-sample RGB from the same renderer, scene,
camera, resolution, and transport settings. **Output:** reconstructed linear RGB.

The renderer is our synthetic data generator. Training happens offline; inference
uses saved weights. The project's ML contribution includes data construction,
custom training, controlled experiments, failure analysis, and deployment.

## Repository layout

```text
RayTracer/                         Existing C++ renderer
  reconstruction.h                 Planned reconstruction interface
  feature_buffers.cpp              Planned aligned features/statistics export
  neural_denoiser.cpp               Planned optional inference adapter
ml/
  pyproject.toml                   Installable package and CLI entry points
  uv.lock                          Reproducible resolved Python dependencies
  README.md                        Setup and complete small-run walkthrough
  configs/
    data/pilot.yaml                Scene family, budgets, seeds, resource caps
    train/spatial_unet.yaml         Model, preprocessing, optimizer, losses
    evaluate/standard.yaml          Splits, baselines, metrics, timing protocol
  schemas/
    example.schema.json            Versioned example/manifest contract
  src/raytracer_ml/
    data/
      scenes.py                    Bounded scene/camera configuration generation
      generate.py                  Renderer jobs, resume, progress, failure log
      validate.py                  Alignment, finite values, completeness, hashes
      split.py                     Group splits and leakage checks
      dataset.py                   Streaming, crops, supported augmentation
    models/
      conv_baseline.py             Small debugging/reference model
      unet.py                      Main trainable reconstruction network
    preprocessing.py               Shared training/inference channel transforms
    losses.py                      HDR-aware objectives
    train.py                       Training, validation, checkpoints, resume
    evaluate.py                    Held-out predictions and quality metrics
    benchmark.py                   End-to-end time/quality comparisons
    export.py                      Deployment export and numeric parity
    cli.py                         Stable command interface
  tests/                           Small data, model, export, and pipeline tests
  reports/
    dataset_card.md                 Generation provenance and data limitations
    model_card.md                   Training, intended use, results, failures
    experiments.md                 Experiment ledger and decisions
    figures/                       Selected final comparisons/plots
docs/
  neural-rendering-plan.md          Research and renderer integration roadmap
  ml-project-structure.md           This implementation blueprint
```

Keep large outputs in a configurable artifact directory outside the source tree:

```text
<artifact_root>/
  datasets/<dataset_version>/
    manifest.jsonl
    splits.json
    examples/<example_id>/          Float buffers and per-example metadata
  runs/<run_id>/
    config.yaml                    Resolved configuration
    environment.json               Versions, device, renderer/code commits
    metrics.jsonl                  Training and validation measurements
    checkpoints/                   Latest resumable state and best model
  evaluations/<evaluation_id>/      Per-image metrics, timings, predictions
  models/<model_version>/           Exported weights, metadata, parity report
```

Small manifests, configurations, reports, and selected figures belong in Git.
Large arrays/checkpoints do not. Version artifacts through content hashes and
manifests; a new dataset or preprocessing schema must not overwrite an old one.

## Dataset contract and generation

Each example identifies the scene, camera, lights, geometry/material configuration,
resolution, integrator settings, renderer commit, scene-generation seed, input
sampling seed, target sampling seed, input/target sample counts, and split group.
Store channel order, coordinate conventions, radiance scale, file paths, checksums,
and schema version. A noise seed must not also regenerate the scene.

Inputs and targets share the scene and camera but use independent sampling streams.
The target should not contain the input's low-sample prefix. Nested input snapshots
at different budgets are valid, but must stay together in the same dataset split.

The generator should create configurations, assign groups, schedule bounded renderer
jobs, validate outputs, and atomically mark examples complete. On restart it checks
hashes and resumes missing/failed examples. Record failures rather than silently
training on incomplete pairs. A dry run estimates count, storage, and render cost;
full generation has explicit disk, concurrency, and elapsed-time caps.

Pilot: 128 distinct configurations at 256 × 144, three input noise realizations,
and 1/2/4/8/16/32-sample snapshots. This produces 2,304 input examples sharing 128
independent reference images; shared targets stay within their configuration's
split. Begin references at 512 samples and validate
against higher-sample independent repeats. High sample count is an approximation
to a clean target, not proof that rare lighting paths have converged.

Validate finite arrays, matching shapes, nonnegative radiance, normal conventions,
masks, sample/variance semantics, camera alignment, checksums, and missing files.
Produce contact sheets and inspect edges, shadows, bright regions, and targets.
An automatic validator cannot establish visual alignment or sufficient convergence
by itself.

## Split and experiment design

Create group-aware train/validation/test manifests before crops or augmentation.
An initial 70/15/15 allocation is a configuration choice, not a per-image shuffle.
Use validation groups for checkpoint/hyperparameter choices. Use the test groups
only for the final comparisons.

Maintain separate test cohorts for held-out camera paths, new lights, and unseen
layouts/assets. A known-scene camera test is distinct from unseen-scene performance.
Keep all noise variants, nested snapshots, crops, and nearby trajectory siblings
with their parent group. Hash checks and group assertions catch accidental overlap.
All preprocessing statistics come from training data or inference-visible inputs.

Train the first model on diffuse scenes. Keep specular/glass/caustic cases as a
reported stress set until the model and reference pipeline support them. Track
dataset licenses for imported assets; begin with our own procedural/example meshes.

## Training loop

Use a streaming dataset and bounded caches. Read full examples for evaluation;
random training crops retain enough context for lighting structures. Any spatial
augmentation must transform normals, depth, camera-related channels, and target
consistently. Arbitrary natural-image augmentation is not automatically valid here.

First overfit a tiny debug set with a small convolutional network. Then train the
compact U-Net with sample-budget conditioning. Initial losses combine log-radiance
L1 and linear reconstruction error. Tune coefficients on validation data and
measure bright-feature energy as well as ordinary image metrics.

Save a resolved config, data/split hashes, code commits, dependency versions,
device, random seeds, learning curves, and elapsed time for every run. Checkpoints
include model, optimizer, scheduler, epoch/step, preprocessing state, and RNG state
for a tested resume path. Keep a latest checkpoint for recovery and a validation-
selected checkpoint for evaluation. Detect non-finite losses and record the failure.
Set maximum steps/epochs and a measured resource budget; avoid unbounded training.

PyTorch with MPS is the initial Mac training path, with CPU smoke tests. Do not
promise bitwise reproducibility across devices; report the tested reproducibility
and export tolerances. Validate an optional C++ deployment runtime early.

Proposed starting configuration, to be checked in the tiny debug run:

| Setting | Initial choice |
| --- | --- |
| Training input | 128 × 128 aligned crops; full 256 × 144 validation images |
| Batch size | 8, reduced if measured memory requires it |
| Optimizer | AdamW, learning rate `3e-4`, weight decay `1e-4` |
| Schedule | Cosine decay over at most 50 epochs |
| Stability | Gradient norm clipping at 1; fail on non-finite loss |
| Checkpoint selection | Lowest validation reconstruction loss, with image/region metrics also reported |
| Early stopping | 8 epochs without validation improvement |
| Repeatability | One development seed; three training seeds for the final selected experiment |

These are initial experimental settings, not established optimal values. Record
every change and its validation evidence. Check peak memory and step time before
running all epochs. Begin with a proposed 10 GiB artifact cap and two-hour cap per
pilot generation/training run; revise the configuration from measured pilot cost
before launching larger work. The caps are planning defaults, not an instruction
to launch generation or training now.

## Baselines and ablations

| Experiment | What it establishes |
| --- | --- |
| Raw low-sample renders | Quality without reconstruction |
| Existing a-trous filter | Improvement over the renderer's current feature |
| Pretrained Open Image Denoise | Comparison with an established learned denoiser |
| Small convolutional baseline | Whether pipeline/model complexity is justified |
| U-Net using RGB only | Value of geometry/statistical conditioning |
| U-Net with guides; then variance/count | Contribution of additional inputs |
| Sample-budget sweep | Where reconstruction helps and where it loses detail |
| Full-resolution and later upscale variants | Separate benefit of fewer samples and fewer pixels |

Avoid treating a prettier selected image as the result. Evaluate complete held-out
cohorts and show failures, including any advantage of the pretrained baseline.

## Evaluation and delivery

Generate machine-readable per-image quality metrics, distributions, worst cases,
fixed-exposure comparisons, and error images. Include HDR error and display
PSNR/SSIM, difficult-region measurements, and full-image results when fallback
leaves unsupported regions raw. Set quality thresholds using validation data.

Benchmark time to matched quality, equal-budget quality, and equal-time quality.
Count tracing, guides/statistics, copies, transfers, synchronized inference,
compositing, and display. Separate cold start, warm median/p95, first useful preview,
and fixed-budget completion. For continuous refinement, count repeated inference
and tracer contention. Track memory, model size, generation cost, and training cost.

Export the selected model with its exact preprocessing/channel schema, supported
domain, input dimensions/budgets, and checksum. Compare Python and C++ predictions
on held-out examples before enabling the viewer. Missing or incompatible models
must produce a clear fallback. Raw radiance remains untouched by inference.

The dataset card documents provenance, generation, splits, licensing, and reference
noise. The model card documents training, intended domain, metrics, timing hardware,
fallbacks, and known failures. The final report includes enough configuration and
artifact identifiers to reproduce the study.

## Automation and acceptance checks

Planned CLI stages are `generate`, `validate-data`, `train`, `evaluate`, `benchmark`,
and `export`. Configuration files and explicit artifact paths control each stage;
these commands are not implemented yet. A walkthrough should connect them into
one small reproducible run before a full dataset or long training run is launched.

CI should run a tiny CPU dataset/model smoke test, split/schema checks, checkpoint
resume, preprocessing parity, and optional export tests. Large datasets, training,
hardware benchmarks, and MPS-specific checks run separately. Existing C++ tests
must continue to pass with ML support disabled. The renderer integration additionally
tests raw-output invariance, stale-result handling, model failure, and cancellation.

The first release is complete when data generation resumes reliably, the model
trains and evaluates reproducibly, held-out comparisons are reported, exported
predictions match, and the viewer demonstrates early reconstruction with measured
total latency. Upscaling and temporal reuse follow as separate studies.
