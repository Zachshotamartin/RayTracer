# Model card — diffuse-pilot-v1

A custom, supervised U-Net trained from scratch on this renderer's measurements.
It reconstructs a same-resolution HDR image from 17 low-sample radiance and guide
channels. This is a working research baseline, not a general-purpose replacement
for a production denoiser.

[Bundled ONNX weights](../../assets/models/diffuse-pilot-v1.onnx) ·
[export schema and hashes](../../assets/models/diffuse-pilot-v1.json) ·
[run configuration](pilot-training-config.json) · [results index](README.md).

## Training

476,419 parameters, width 32 U-Net; 128×128 crops; batch 8; AdamW at 3×10⁻⁴;
cosine schedule for 50 epochs; gradient clipping at 1. Loss is log-radiance L1
plus 0.1 × relative linear L1. No pretrained weights or perceptual loss were used.
Training took **1,119.3 seconds (18.7 minutes)** in recorded completed epochs on
Apple M3 Pro / MPS, PyTorch 2.14.0, Python 3.12.11. Dataset validation and initial
setup are outside that training timer. The best checkpoint was selected on
validation loss (0.005188, epoch 48 using one-based numbering). Only one pilot
training seed was run; there are no multi-seed confidence claims.

![Training and validation loss across 50 epochs, with raw validation input as a baseline](figures/pilot-training.svg)

[Recorded learning curve](pilot-training.jsonl). Training uses crops, while
validation uses full images; the marker identifies the selected checkpoint.

## Labeled visual comparison

| Raw path tracing · 4 samples/pixel | A-trous denoising · 4 samples/pixel |
| :---: | :---: |
| ![Raw path tracing at 4 samples per pixel](figures/pilot-4spp-raw.png) | ![A-trous denoising of the same 4-sample input](figures/pilot-4spp-atrous.png) |
| **Our trained U-Net · 4 samples/pixel** | **Independent reference · 512 samples/pixel** |
| ![Custom U-Net reconstruction of the same 4-sample input](figures/pilot-4spp-neural.png) | ![Independent raw reference at 512 samples per pixel](figures/pilot-4spp-reference.png) |

The first test configuration, `g0027-v000-n0-s4`, at 256 × 144 and fixed exposure 0.
The target is independent of the shared 4-spp input. The metrics below summarize
all test images, rather than just this displayed example. See the
[error image and artifact index](README.md).

## Held-out image quality

All 360 test examples from 5 held-out layouts, 4 views each, 3 noise realizations
and 6 budgets are included. These are correlated views/noise variants, not 360
independent scenes. Metrics are averaged per image; PSNR/SSIM use a fixed ACES+sRGB
display transform, while HDR MSE uses untouched linear values.

| Method | Display PSNR ↑ | SSIM ↑ | Linear MSE ↓ |
| --- | ---: | ---: | ---: |
| Raw path tracing | 25.84 dB | 0.5224 | 0.0020503 |
| Existing a-trous | 33.97 dB | 0.9440 | 0.0002140 |
| Custom U-Net | **35.96 dB** | 0.8929 | **0.0001185** |
| OIDN 2.5.1, Metal | **39.44 dB** | **0.9714** | **0.0000608** |

The custom model reduces error substantially versus raw rendering. Relative to
a-trous it improves average PSNR and linear MSE, but **loses SSIM**. OIDN is better
on all three reported aggregate metrics. We do not describe the custom model as
state of the art or superior to existing learned denoisers.

| Input spp | Raw PSNR | A-trous PSNR | Custom PSNR | OIDN PSNR | Custom SSIM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.81 | 26.65 | 31.13 | 35.92 | 0.7881 |
| 2 | 21.50 | 31.95 | 33.27 | 37.54 | 0.8448 |
| 4 | 24.35 | 35.34 | 35.29 | 39.00 | 0.8913 |
| 8 | 27.24 | 36.40 | 37.16 | 40.30 | 0.9252 |
| 16 | 30.15 | 36.70 | 38.78 | 41.47 | 0.9474 |
| 32 | 33.01 | 36.79 | 40.12 | 42.42 | 0.9609 |

![Mean held-out PSNR and SSIM versus sample count for all four methods](figures/pilot-quality.svg)

Each plotted point summarizes 60 test images; the metrics are not inference-time
measurements. [Plot source](plot_results.py).

Full results: [test summary](pilot-test-summary.json), [per-image metrics](pilot-test-per-image.jsonl.gz),
[validation summary](pilot-val-summary.json). Difficult pixels and failure cases
remain in these full-image metrics. Highlight and support-region errors are also
included per image. The 512-spp reference has residual noise; see the dataset card.

## Deployment and timing scope

Native ONNX CPU predictions are checked against Python on actual renderer features;
temporal and 2× parity tests cover those separate model classes. This model's ONNX
export has maximum absolute random-input parity error **1.19×10⁻⁶** across tested
odd/even dynamic shapes. The inference graph contains preprocessing and unsupported
pixel fallback. Weights occupy approximately 1.8 MiB.

Median synchronized Python/MPS inference across test images was **4.73 ms** at
256×144, including tensor input/output transfers. This excludes tracing, feature
collection, image display and model loading. It is **not** an end-to-end speedup.
See [native timing results](timing.md) for the fixed-budget completion benchmark,
which counts BVH construction, tracing, guides, inference and image writing.
Interactive display latency and tracer/inference contention have not been profiled.
Peak device memory has not been measured; no memory advantage is claimed.

## Intended use and limits

Use with static diffuse pinhole scenes in the demonstrated procedural family,
initially at 256×144 and 1–32 spp. Dynamic sizes run, but arbitrary resolution,
lighting and assets are out-of-distribution tests. At 64 spp the timing sweep is
also outside the training budget range and is labeled in its report.

Predictions do not modify raw accumulation. Primary glass/metal/emitter/background
and mixed support pixels fall back to raw RGB. This cannot detect every indirect
specular influence on a diffuse surface. Defocus, photon mapping and approximate
glass shadows cause whole-frame fallback. Missing or incompatible models produce
an explicit raw fallback. The viewer can compare raw, reconstruction, reference and
error images; saved output records real sample counts.

Known weaknesses include residual texture/noise, bright-edge error, softened detail,
limited scene diversity and lack of a temporal stability study. The separate 2× and
temporal smoke models do not demonstrate better quality than a-trous. Temporal
training uses noisy/a-trous history, while deployment reuses predictions; this
mismatch needs further training research. These smoke weights are local artifacts,
not the bundled recommended model.

The [issue audit and improvement plan](../../docs/neural-reconstruction-improvement-plan.md)
separates measured failures, implementation constraints and untested risks, and
defines the next training, evaluation and deployment experiments.

Model SHA-256: `ec0f7176b5a785959e7f65603385a4618068b088e3f2d6f6368a8b050a08fa19`.
Checkpoint SHA-256: `49ab1e4eb8ff8b25ad0327739dda7c6a2dd15a9fadbcf1766a7e2c1d4caa64d2`.
Development-tree provenance limits are documented in the [dataset card](dataset_card.md).
