# Joint denoising and 2× reconstruction

The objective is to approach a native high-sample ray-traced reference at the
requested output resolution and beat conventional denoising at comparable total
cost. A better training loss alone does not demonstrate rendering acceleration.

On 2026-09-08 the user stopped `full-mac-guided-s42-v1`. Its latest complete
checkpoint is epoch 33 (177,408 updates); the selected best is epoch 31. Both full
resume states and the immutable epoch archives remain on the external SSD. The
old training monitor is paused and must not restart that run. This is a new
experiment with a different task, architecture and dataset contract.

## What changes

| Component | Joint experiment |
| --- | --- |
| Target | Independent native image with twice the input width and height |
| Model | Width-32 U-Net with skip connections, refinement and a learned pixel-shuffle head; 508,172 parameters |
| Inputs | 27 low-resolution radiance, material, geometry, boundary and variance channels |
| Output | Linear HDR RGB; separate learned corrections for the four output subpixels |
| Pixel policy | The entire frame is learned, including mixed boundaries, emitters and non-diffuse transport |
| Preservation | Independent near-clean low-resolution measurements compared with the native high-resolution target |
| Loss | Log/relative radiance L1, gradients at 1/2/4-pixel offsets and patch/highlight energy |
| Checkpoints | Latest, selected best, best eligible and immutable full epoch archives |
| Platform | Local Mac/MPS training; CPU or Core ML native inference must be measured separately |

The legacy models and their material/sample bypass behavior are unchanged. For the
joint model, a near-clean low-resolution render still needs spatial reconstruction;
forcing its enlarged pixels to raw identity would prevent learning that task.
Learning the whole frame increases the need to check reflections, glass, highlights
and missing subpixel detail. It is not evidence those problems are already solved.

## Data and augmentation

The pilot recipe is [`joint-variety-pilot.yaml`](../ml/configs/data/joint-variety-pilot.yaml).
It produces 96 independently seeded layouts, with all eight scene families and four
detail strata in each 32-layout split. Two native rendering variants per layout
produce 192 reference views and 1,920 examples: 640 train, 640 validation and 640
sealed test. Sample budgets are 1/4/16/32/64, with two independent input streams.

| Pilot input W×H | Native target W×H |
| --- | --- |
| 64×48 | 128×96 |
| 80×60 | 160×120 |
| 64×64 | 128×128 |
| 48×80 | 96×160 |
| 112×48 | 224×96 |
| 96×64 | 192×128 |

These small frames bound the initial experiment. The separate
[`joint-variety-full.yaml`](../ml/configs/data/joint-variety-full.yaml) expansion
recipe includes native inputs through 384×216, square, portrait and wide frames,
2,048-spp targets and 57,344 examples. It is not launched automatically. Larger
1080p/1440p output evaluations and additional unseen geometry remain qualification
work; small-patch success is not proof of performance at those sizes.

Resolution selection uses reshuffled balanced cycles, avoiding a fixed association
between one scene family and one size. Every variant inherits its parent's split.
Camera field-of-view zoom (0.7–1.6× in the pilot), camera roll (±25°), and RGB lighting
filters are applied **before tracing both members of the pair**. Thirty percent
transport probability adds a real glass or metal object, recorded in the manifest.
Depth of field and temporal reconstruction are separate experiments.

Training additionally samples aligned 32×32 input / 64×64 target crops, emphasizes
edges and image borders, flips and rotates patches by 90°, changes exposure and
white balance, and fuses independent sample streams. RGB filters are positive,
diagonal linear gains: target radiance changes by the gain and variance by its
square. World-coordinate normals retain their components when pixels are rearranged.
Camera roll provides arbitrary-angle variation without interpolating those guides.

Do not apply arbitrary Gaussian blur, sharpening, JPEG, gamma changes or stretched
resize to these physical buffers as if they were ordinary photographs. That changes
noise, guide or radiance semantics. Enlarging an old reference also cannot create
a valid high-resolution training target. We retain all old data as baseline evidence;
the joint collection gets its own path and immutable generation fingerprint.

The first 512-spp collection exposed a noisy training reference (28.67 dB /
0.9342 SSIM against its independent check). It is retained as a superseded
collection; the training preparation uses the new `joint-variety-pilot-v2` dataset.

The pilot uses 2,048-spp references and retains 12 independent 8,192-spp check images,
balanced across splits. These allow regional noise/convergence inspection but do
not establish that every reference is converged. Reference-check error in edges,
dark regions and highlights must be assessed before making a quality claim.

## Training and acceptance

[`joint-mac-pilot.yaml`](../ml/configs/train/joint-mac-pilot.yaml) declares a 30-epoch
cosine schedule, patience eight, batch eight, FP32 MPS and two-hour invocation cap.
The preparation command performs inference and integrity checks without creating an
optimizer or starting fitting. It checks every validation resolution plus the actual
training crop/batch shape. Never resume the old scale-1 model into this experiment.

Validation uses complete native frames, without random augmentation. Each epoch
records PSNR, SSIM, edge error, whole-frame HDR error, preservation, learning health
and breakdowns by resolution, sample budget, scene family, detail stratum and
transport category. Eligibility requires global, **per-resolution and per-budget**
PSNR ≥30 dB, SSIM ≥0.95, ≥0.1 dB gain over a-trous+bilinear, no SSIM/edge/HDR regression,
and no near-clean preservation regression relative to bilinearly enlarged measured
inputs. These are predeclared development gates, not universal perceptual guarantees.
One failed gate keeps a model ineligible while ordinary early stopping remains active.
Budget gates prevent improvement on cleaner 64-spp inputs from hiding a failure on
1-spp inputs. Family, detail and transport slices are diagnostic reports; a good
overall score is not a guarantee that every scene or intersection of slices passes.

Native benchmarking compares five paths at the **same final dimensions**:

1. Native-resolution raw path tracing.
2. Native-resolution a-trous denoising.
3. Low-resolution raw rendering plus bilinear enlargement.
4. Low-resolution a-trous plus bilinear enlargement.
5. Low-resolution rendering plus the joint model.

`--output-scale 2` implements baseline enlargement inside the C++ pipeline so its
cost is included. Timers include BVH, tracing, geometry features, denoising/upscaling
or inference and image writing. Cold model load and whole process time are also
reported. A speedup is measured only where both the neural method and a baseline
reach the declared PSNR/SSIM threshold; failure to reach it is not an infinite win.
Optional pretrained OIDN remains an additional quality comparison in `rtml evaluate`;
the native timing report does not claim to benchmark OIDN.
Native benchmarks validate dataset checksums and bind their report to the training
manifest and source checkpoint. An external cohort requires the same registered
checkpoint/cohort contract used by offline evaluation (set `registration` in the
benchmark YAML). Zero warm repeats and invalid quality thresholds are rejected.

Evaluation comparison PNGs contain method names, sample counts, PSNR/SSIM, native
output dimensions and the independent reference budget. Up to 12 configurations
are spread through their IDs, choosing an input near 4 spp by metadata alone.
This avoids showing only the first scene's many noise/sample variants. Optional
OIDN appears in the panels when evaluated. `comparisons/index.json` records the
selection policy and the error-image display scale.

## Reproduce

Use a committed source snapshot and a renderer built from that snapshot. From the
repository root, with the SSD attached:

```sh
joint_root="/Volumes/Zach's SSD/RayTracer"
ml/.venv/bin/rtml generate --config ml/configs/data/joint-variety-pilot.yaml --renderer build-detail/raytracer --output "$joint_root/datasets/joint-variety-pilot-v2" --dry-run
ml/.venv/bin/rtml generate --config ml/configs/data/joint-variety-pilot.yaml --renderer build-detail/raytracer --output "$joint_root/datasets/joint-variety-pilot-v2"
ml/.venv/bin/rtml prepare-training --config ml/configs/train/joint-mac-pilot.yaml --data "$joint_root/datasets/joint-variety-pilot-v2" --output "$joint_root/runs/joint-mac-pilot-s42-v1" --report "$joint_root/preparations/joint-mac-pilot-s42-v1/preflight.json"
```

Generation is resumable with the identical code, configuration and renderer. Its
time cap is 30 minutes per invocation and disk cap is 12 GiB. Dataset manifests and
checksums record native dimensions and camera variants. Checkpoint history belongs
to a fresh run directory. The training run is a separate explicit launch after the
preflight is reviewed; unit-test optimizer steps are not a new portfolio experiment.
The user has authorized starting the new pilot after the completed dataset and
device checks pass. Review epochs one and two while training continues; do not
pause a healthy run merely to inspect metrics or because eligibility is still false.

Start a fresh run, or resume its complete latest state after a clean invocation
boundary with no other trainer active:

```sh
ml/.venv/bin/rtml train --config ml/configs/train/joint-mac-pilot.yaml --data "$joint_root/datasets/joint-variety-pilot-v2" --output "$joint_root/runs/joint-mac-pilot-s42-v1"
ml/.venv/bin/rtml checkpoints --output "$joint_root/runs/joint-mac-pilot-s42-v1"
ml/.venv/bin/rtml train --config ml/configs/train/joint-mac-pilot.yaml --data "$joint_root/datasets/joint-variety-pilot-v2" --output "$joint_root/runs/joint-mac-pilot-s42-v1" --resume
```

These are alternative launch actions, not concurrent commands. For the actual
2026-09-08 run, the SSD preparation launcher uses frozen training source `a9d8255`;
its manifest, configuration and source checksums are checked before every launch.
The [initial run receipt](../ml/reports/joint-mac-pilot-start.md) records the data
audit and the first two checkpoint reviews.

After a trained candidate exists:

```sh
ml/.venv/bin/rtml evaluate --data "$joint_root/datasets/joint-variety-pilot-v2" --checkpoint "$joint_root/runs/joint-mac-pilot-s42-v1/best.pt" --output "$joint_root/evaluations/joint-mac-pilot-val" --split val --device mps
ml/.venv/bin/rtml export --checkpoint "$joint_root/runs/joint-mac-pilot-s42-v1/best.pt" --output "$joint_root/exports/joint-mac-pilot.onnx"
ml/.venv/bin/rtml benchmark --data "$joint_root/datasets/joint-variety-pilot-v2" --renderer build-detail/raytracer --model "$joint_root/exports/joint-mac-pilot.onnx" --output "$joint_root/benchmarks/joint-mac-pilot-val" --config ml/configs/evaluate/joint-pilot.yaml
```

Export parity includes odd, portrait, square and landscape dimensions. The U-Net
does not advertise tiled inference; full-frame memory at larger sizes needs explicit
measurement. No new model is promoted and the existing README gallery remains
labeled as results from the original diffuse pilot.

## Research basis and limits

[NVIDIA's ray reconstruction research](https://research.nvidia.com/labs/adlr/DLSS4/)
describes the joint noisy-low-resolution to clean-high-resolution task and the
importance of preserving geometry, textures and lighting. Our spatial U-Net is a
bounded Mac experiment, not a reproduction of its temporal transformer or hardware.
[OIDN's training implementation](https://github.com/RenderKit/oidn/blob/master/training/dataset.py)
uses paired crops and spatial augmentation. Patch training is therefore a reasonable
memory strategy; native-size and scene diversity still require independent testing.

Passing functional checks establishes that the intended task can now be trained and
measured. It does not yet establish that the new model beats a-trous, OIDN or native
path tracing, reconstructs unseen subpixel geometry faithfully, or is temporally stable.

## Project review and remaining evidence

| Concern | Implemented control | Evidence still required |
| --- | --- | --- |
| Wrong learning task | Native high-resolution targets and a learned subpixel output head | Held-out reconstruction quality |
| Small model or insufficient data | 508,172-parameter pilot, diverse native views, larger collection recipe | Train/validation curves and a measured capacity/data scaling study |
| Lost edges, thin structures, textures | Geometry features, edge/border crops, skip connections, multi-offset gradient loss | Detail metrics and labeled comparisons, including failures |
| Noisy or biased supervision | Independent seeds, 2,048-spp references, retained 8,192-spp checks | Regional agreement on sampled targets; no blanket convergence claim |
| Glass, metal, highlights | Real non-diffuse scenes, whole-frame learning, HDR/energy checks and transport reports | Adequate visible transport coverage and error measurements |
| Augmentation corrupts labels | Aligned scale-aware crops/rotations; radiance and variance gains; camera transforms before tracing | Exact transform tests pass; learning benefit needs ablation |
| Misleading aggregate quality | Global, resolution and budget gates, plus family/detail/transport reports | Inspect worst cases and tradeoffs; do not select on sealed test |
| Interrupted or stale training | Frozen code/config/data, atomic latest/best/full epoch states, exact resume tests | Verify live checkpoint receipts at early epochs and invocation boundaries |
| Misleading speedups | Five native paths at equal output size, validated provenance and finite matched-quality thresholds | Trained-model cold/warm end-to-end timings on this Mac |
| Unclear images | Embedded labels and metadata-selected comparisons | View generated candidate panels at native size before claiming improvement |
| Larger images and motion | Dynamic shape/export checks; spatial model explicitly scoped | 1080p/1440p quality, peak memory, latency and motion-flicker experiments |

The pilot is an integration and learning experiment. Its 64 training views are not
evidence that the final data volume is sufficient. Expand only after the pilot shows
useful learning and the error analysis identifies what additional data or model
capacity is needed. Runtime success, learned quality and project completion are
separate measurements; unresolved experimental questions are not labeled fixed.
