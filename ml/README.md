# Learned ray reconstruction

This package trains our own model to reconstruct linear HDR images from noisy
path-traced measurements. It includes data generation, integrity checks, grouped
splits, training and resume, evaluation against raw/a-trous/optional OIDN, ONNX
export, and native timing. The C++ viewer runs the exported model while preserving
its original radiance accumulation.

The first study is diffuse, pinhole, same-resolution reconstruction. Trainable 2×
and temporal variants are implemented and tested separately. Their small smoke
runs establish functionality, not a demonstrated speed or flicker advantage.
See [model card](reports/model_card.md), [dataset card](reports/dataset_card.md),
and [experiment ledger](reports/experiments.md) for measured results and limits.

The [next model improvement plan](../docs/neural-reconstruction-improvement-plan.md)
maps the edge/detail, material, temporal, upscaling and performance concerns to
specific experiments and acceptance gates. Those changes are planned, not trained.

| Raw path tracing · 4 samples/pixel | A-trous denoising · 4 samples/pixel |
| :---: | :---: |
| ![Raw path tracing at 4 samples per pixel](reports/figures/pilot-4spp-raw.png) | ![A-trous denoising of the same 4-sample input](reports/figures/pilot-4spp-atrous.png) |
| **Our trained U-Net · 4 samples/pixel** | **Independent reference · 512 samples/pixel** |
| ![Custom U-Net reconstruction of the same 4-sample input](reports/figures/pilot-4spp-neural.png) | ![Independent raw reference at 512 samples per pixel](reports/figures/pilot-4spp-reference.png) |

*Each panel is 256 × 144, with the same camera, depth 16 and exposure 0.
Input: `g0027-v000-n0-s4`, the first held-out configuration, noise realization 0.
Only the reference uses 512 samples; the three comparison methods use 4.*

[Results gallery, error image and artifact index](reports/README.md) ·
[Quality metrics](reports/model_card.md#held-out-image-quality) ·
[Measured rendering time](reports/timing.md).

## Setup

From the repository root, install [uv](https://docs.astral.sh/uv/getting-started/installation/),
then use Python 3.12 and the checked-in lockfile:

```sh
uv sync --project ml --python 3.12 --extra dev --locked
cmake -S . -B build-headless -DCMAKE_BUILD_TYPE=Release -DRAYTRACER_ENABLE_SDL=OFF
cmake --build build-headless --parallel
```

Commands below use `ml/.venv/bin/rtml`; on Windows use
`ml/.venv/Scripts/rtml.exe` and `build-headless/Release/raytracer.exe`.
`device: auto` selects PyTorch MPS on supported Macs, otherwise CPU. Linux uses
CPU-only PyTorch wheels. CUDA training is not currently an option. Large datasets,
checkpoints and predictions go under ignored `artifacts/`; keep space available.

## Complete small experiment

```sh
ml/.venv/bin/rtml generate --config ml/configs/data/smoke.yaml --renderer build-headless/raytracer --output artifacts/datasets/my-smoke --dry-run
ml/.venv/bin/rtml generate --config ml/configs/data/smoke.yaml --renderer build-headless/raytracer --output artifacts/datasets/my-smoke
ml/.venv/bin/rtml validate-data --data artifacts/datasets/my-smoke
ml/.venv/bin/rtml train --config ml/configs/train/smoke.yaml --data artifacts/datasets/my-smoke --output artifacts/runs/my-smoke
ml/.venv/bin/rtml evaluate --data artifacts/datasets/my-smoke --checkpoint artifacts/runs/my-smoke/best.pt --output artifacts/evaluations/my-smoke
ml/.venv/bin/rtml export --checkpoint artifacts/runs/my-smoke/best.pt --output artifacts/models/my-smoke.onnx
```

Rerunning `generate` checks hashes and resumes completed jobs. A changed config or
renderer binary requires a new dataset directory. The generator runs one bounded
renderer process at a time, with explicit thread, disk and time caps. References
and inputs have independent random streams; changing sampling seeds does not
change scene geometry. Each reference is shared only by its configuration's inputs.
The dry run reports counts and uncompressed input storage, not a promised runtime
or complete peak disk estimate. The cap also accounts for output and temporary files.

Use `pilot.yaml` with `spatial_unet.yaml` for the 2,304-example study: 128 views,
256×144 inputs, 1/2/4/8/16/32 spp, three noise realizations, and 512-spp targets.
Four references are checked against independent 2048-spp renders. High sample count
is still a noisy approximation to ground truth.

Data configs require width divisible by 16 so 16:9 input/target sizes align exactly.
`cohort_groups: 2` adds separately labeled held-out-camera and new-light tests using
training layouts; the ordinary test split contains unseen layouts. All variants of
a full scene/camera and all crops stay in the same split. `cohort-smoke.yaml`
demonstrates these cohorts. They must not be reported as unseen-layout tests.

## Reproduce the pilot study

The following starts a new run at the published dataset/model settings. The native
benchmark at the end also requires the ONNX build described below. Choose a new
output directory if the configuration or renderer binary changes.

```sh
ml/.venv/bin/rtml generate --config ml/configs/data/pilot.yaml --renderer build-headless/raytracer --output artifacts/datasets/my-pilot --dry-run
ml/.venv/bin/rtml generate --config ml/configs/data/pilot.yaml --renderer build-headless/raytracer --output artifacts/datasets/my-pilot
ml/.venv/bin/rtml validate-data --data artifacts/datasets/my-pilot
ml/.venv/bin/rtml train --config ml/configs/train/spatial_unet.yaml --data artifacts/datasets/my-pilot --output artifacts/runs/my-pilot
ml/.venv/bin/rtml evaluate --data artifacts/datasets/my-pilot --checkpoint artifacts/runs/my-pilot/best.pt --split val --output artifacts/evaluations/my-pilot-val
ml/.venv/bin/rtml export --checkpoint artifacts/runs/my-pilot/best.pt --output artifacts/models/my-pilot.onnx
ml/.venv/bin/rtml evaluate --data artifacts/datasets/my-pilot --checkpoint artifacts/runs/my-pilot/best.pt --split test --output artifacts/evaluations/my-pilot-test
```

For the published test's OIDN comparison, add `--oidn /absolute/path/oidnDenoise`
and `--oidn-device metal` to the final command. Omit those flags when OIDN is not
installed. The saved pilot timings use `configs/evaluate/pilot.yaml`; the more
general `standard.yaml` contains initial defaults. Fix the threshold using
validation before timing test views. New runs need their own measurements: the
committed metrics describe the original model and hardware, not every retraining.

## Training and experiment controls

The network predicts an HDR log-radiance residual, initialized to preserve raw RGB.
Loss combines log L1 and relative linear L1; AdamW, cosine decay, gradient clipping,
epoch/time limits, validation checkpoint selection and early stopping are configured.
The model accepts `kind: conv|unet`, `inputs: rgb|guides|all`, and width 4–128.
These switches support architecture and feature ablations with independent run dirs.
The original recipes use random aligned crops. Detail recipes also support paired
flips/rotations, radiance/variance-consistent lighting augmentation, independent-noise
fusion and measured near-clean inputs. World-space normals are not treated as image
colors. Validation uses complete images.

Every run writes resolved `config.json`, `environment.json`, `metrics.jsonl`,
`latest.pt`, `best.pt`, `best_score.pt` and `summary.json`. `best_eligible.pt` exists
only after a candidate passes configured constraints. `best.pt` prioritizes that
candidate; when none passes it retains an explicitly unqualified diagnostic model.
The atomic `latest.pt` contains optimizer, scheduler, random generators, data
contract and selection snapshots; the model aliases contain weights and selection
metadata. Resume is tested bit-for-bit on CPU at epoch
boundaries; equivalence across devices/platforms is not promised.

For new runs, [C14](configs/train/detail/c14-quality-selection.yaml) adds HDR and
measured 96-spp preservation constraints, while
[temporal quality](configs/train/detail-temporal-quality.yaml) adds comparable
a-trous temporal constraints. Missing required coverage fails selection. Read the
[qualification corrections and measured results](reports/qualification-fixes.md)
before treating any development pass as release qualification.

```sh
ml/.venv/bin/rtml train --config ml/configs/train/smoke.yaml --data artifacts/datasets/my-smoke --output artifacts/runs/paused --max-new-epochs 1
ml/.venv/bin/rtml train --config ml/configs/train/smoke.yaml --data artifacts/datasets/my-smoke --output artifacts/runs/paused --resume
```

A partially interrupted epoch replays from the previous completed epoch. Changing
a run config is a new experiment, not a resume. A cap too short to finish the first
epoch requires a new run directory with a smaller workload or larger budget.
Do not load untrusted checkpoints; loading uses PyTorch's weights-only format.

## Native inference and viewer

Download the official [ONNX Runtime C++ distribution](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0)
matching the OS and CPU architecture, and extract it. For Apple silicon:

```sh
cmake -S . -B build-neural -DCMAKE_BUILD_TYPE=Release -DRAYTRACER_ENABLE_ONNX=ON -DONNXRUNTIME_ROOT=/absolute/path/onnxruntime-osx-arm64-1.29.0
cmake --build build-neural --parallel
./build-neural/raytracer --scene-file assets/scenes/diffuse-room.json --width 256 --samples 8 --depth 16 --model assets/models/diffuse-pilot-v1.onnx --output renders/ai.png --raw-output renders/raw.pfm
```

Use `--headless` for a fixed-budget render. `N` toggles learned reconstruction,
`D` toggles a-trous, and `S` saves. Inference runs on a background worker, with
bounded pending snapshots and generation checks to discard stale UI results.
Additional real samples continue to improve raw accumulation; predictions are
never inserted into it. Snapshot updates occur at increasing sample budgets.
Model failure gives a clear raw fallback. Native inference is optional: ordinary
builds retain all conventional rendering capabilities.

CPU inference is the portable default. `--neural-provider coreml` requests the
ONNX Core ML execution provider on macOS, which can partition unsupported operations
to CPU. Provider selection alone does not prove GPU/Neural Engine utilization.
The dynamic pilot graph produced Core ML shape/compiler warnings in the local
probe, so Core ML remains experimental. CPU is the validated deployment baseline.
Never compare CPU and Core ML timings without recording the provider and cold start.
The Xcode project defaults to the portable build; CMake is the documented ML build.

Generate a matching independent raw reference to enable `V` (reference) and `E`
(4× absolute linear error). Identity checks include scene/camera, output size and
transport settings; sample seed and sample count intentionally differ:

```sh
./build-neural/raytracer --headless --scene-file assets/scenes/diffuse-room.json --width 256 --samples 512 --depth 16 --seed 9001 --output renders/reference.pfm
./build-neural/raytracer --scene-file assets/scenes/diffuse-room.json --width 256 --samples 8 --depth 16 --model assets/models/diffuse-pilot-v1.onnx --reference renders/reference.pfm
```

The reference's JSON sidecar is required. Do not use a denoised image as reference.
Unsupported primary pixels (background, glass, metal, emitters and mixed boundaries)
preserve raw measurements; in a 2× model they use ordinary bilinear upscaling.
Defocus, photon mapping and approximate glass shadows cause whole-frame fallback.
Diffuse pixels affected by unseen specular transport are still out of training domain.

## Upscaling and temporal history

Use `data/upscale-smoke.yaml` with `train/upscale-smoke.yaml`: the input is 64×36,
target 128×72; the same train/evaluate/export commands apply. Evaluation compares
against bilinear raw and a-trous upscaling at the same target dimensions. Native
`--width` always specifies traced input width, and the model determines output scale.

Use `data/cohort-smoke.yaml` with `train/temporal-smoke.yaml`, or the pilot data
with `train/temporal.yaml`. These original configurations add reprojected previous RGB and a
validity mask. Their training uses noisy or a-trous previous observations, never targets;
evaluation and native playback feed the actual previous prediction. This history
distribution gap is an explicit research limitation of the initial implementation.

The newer `data/detail-sequence.yaml` and `train/detail-temporal.yaml` configurations
use differentiable autoregressive prediction history with whole-sequence validation.
See the [active study](reports/detail-study.md) for its implementation and outstanding
quality gates, and the [research audit](../docs/neural-reconstruction-research-audit.md)
for architecture, data and training-recipe findings. The
[OIDN toolkit control](reports/oidn-training-control.md) uses a separately installed
upstream training pipeline with our renders.

Before scaling a new U-Net recipe, use the bounded fixed-training-patch diagnostic
in `tools/diagnose_learning.py`, then a short matched run with the actual augmentation
policy. `model.radiance_scale` defaults to 1; the C13 development configuration tests
16 while restoring the original output units. Optional `learning_diagnostics`
records learning relative to the noisy input and flags ineffective epochs in
`rtml status`. See the [diagnosis and controls](reports/learning-diagnosis.md).

DetailNet now also supports explicit `output_head: additive_log` and internal
`radiance_scale: 16` conditioning for refinement/upscaling. New 2× experiments can
set `blend_policy: single` to avoid the legacy double attenuation. Guided experiments
can select `guide_policy: sampled`; loss experiments can select `loss.kind: relative_l2`.
These alternatives retain legacy defaults and require fresh experimental runs.
Set `learning_diagnostics.warning_metric: model_region_log_mae` to diagnose learning
where the model can change the image; whole-image loss and bypass counts remain visible.
See the [exact contracts and bounded comparison procedure](../docs/neural-reconstruction-controls.md)
and [paired research configuration](configs/research/literature-controls.yaml).

`--sequence DIRECTORY` plays sorted JSON scene frames in the viewer and exports
numbered outputs. In headless mode it renders all frames with one loaded model:

```sh
ml/.venv/bin/rtml sequence --data artifacts/datasets/my-pilot --group layout-0027 --output renders/camera-path
./build-neural/raytracer --headless --sequence renders/camera-path --model artifacts/models/temporal.onnx --width 256 --samples 4 --depth 16 --output renders/sequence/frame.pfm
```

Each JSON is a complete scene, with the camera changed for movement. `Space` pauses
playback. Seed `N` becomes `N + frame_index`. Reprojection uses world positions,
normal/support checks and geometric disocclusion rejection; light/geometry changes,
large camera translations and resolution changes invalidate history. Repeated
passes at the same camera use a fixed previous-frame history, not independent
observations. Dynamic objects, learned flow and simultaneous temporal+2× inference
are not supported. Evaluation reports valid-history fraction and comparable linear/log
temporal errors for raw, a-trous and neural outputs under identical geometry masks,
plus OIDN when requested. Cuts/unmatched frames are unmeasured, and coverage is
explicit. These measurements are part of the [development temporal audit](reports/qualification-fixes.md),
not a complete motion or flicker qualification.

## Evaluation and benchmarks

`evaluate` writes per-image JSONL, aggregate and cohort metrics, prediction arrays,
fixed-exposure four-panel comparisons (raw / a-trous / custom / reference) and error
images. Metrics include HDR MSE, log error, display PSNR/SSIM, and supported/highlight
region error. ONNX export checks dynamic odd/even sizes; integration tests compare
actual renderer features and temporal sequences against C++ inference.

Add `--oidn /absolute/path/oidnDenoise --oidn-device metal` to evaluation to compare
with official [Open Image Denoise](https://www.openimagedenoise.org/downloads.html).
Its process time includes file IO and startup and is not a warmed inference time.
The comparison does not train or fine-tune OIDN.

```sh
ml/.venv/bin/rtml benchmark --data artifacts/datasets/my-pilot --renderer build-neural/raytracer --model artifacts/models/my-pilot.onnx --config ml/configs/evaluate/pilot.yaml --output artifacts/benchmarks/my-pilot
```

Choose PSNR/SSIM thresholds using validation before evaluating final test frames.
The example thresholds are initial defaults. Benchmark all tested budgets/methods,
including failures; a missed target is `null`, never an invented speedup. Native
repeats include BVH construction, tracing, features, inference and image write;
cold time adds model load. JSON scene parsing, OS startup and display are excluded
from warm timing; whole subprocess time is separate. This command currently targets
the spatial scale-1 model. Python timing includes synchronized input/output transfers.
Neither timing is an interactive display/flicker or memory benchmark.

## Common setup and artifact issues

| Message or symptom | Action |
| --- | --- |
| Neural inference unavailable | Build with `RAYTRACER_ENABLE_ONNX=ON` and point `ONNXRUNTIME_ROOT` at the matching extracted C++ distribution. The ordinary build can still render and generate data. |
| Missing or incompatible model | Use the bundled spatial weights or export this package's checkpoint. Channel order, schema, domain and output scale are checked. Read `neural_error` in the render JSON. |
| Dataset configuration/renderer changed | Generate into a new directory. The existing dataset's hash contract intentionally prevents mixing renderer binaries. |
| Resume configuration or dataset differs | Resume with the original config and data; create a new run directory for a changed experiment. |
| No epoch completed within the budget | Reduce the workload or increase the time cap, then use a fresh run directory. There is no completed checkpoint to resume. |
| Reference missing or mismatched | Supply a raw PFM and its JSON sidecar. Match scene/camera, output dimensions and transport settings; use independent noise seeds. |
| Output is raw in unsupported scenes | The initial model excludes defocus, approximate glass shadows and photon mapping. Primary unsupported materials preserve raw values. |

To inspect or share results without the full dataset, use the committed
[reports index](reports/README.md). It contains the comparison panels, exact metrics,
compressed manifests, training curves, model hashes and timing records.

## Tests

```sh
RAYTRACER_BINARY=build-headless/raytracer ml/.venv/bin/pytest -q ml/tests
RAYTRACER_BINARY=build-neural/raytracer RAYTRACER_NEURAL_BINARY="$PWD/build-neural/raytracer" ml/.venv/bin/pytest -q ml/tests
ml/.venv/bin/ruff check ml/src ml/tests
ctest --test-dir build-neural --output-on-failure
```

CI uses a tiny CPU-generated dataset and tests schemas, leakage, raw invariance,
checkpoint resume, gradients, export, fallback, 2× and native temporal parity.
Full training and hardware performance measurements are separate experiments.
