# Detail reconstruction study

Status: implementation and development experiments in progress. The bundled
`diffuse-pilot-v1.onnx` and its measured limitations remain the released baseline.
Passing implementation tests does not establish better image quality or acceleration.

Read current artifact progress with `rtml status --artifacts /path/to/RayTracer`.
The artifact directory contains `datasets/` and `runs/`. The command reports saved
example/reference receipts, epochs versus their configured maximum, validation
structure scores, parameter counts, and stopping reasons. It does not mistake
receipt counts for a fresh checksum audit, infer process liveness, or combine
dataset completion and model quality into a misleading overall percentage.

## Data and storage

| Collection | Layouts / views | Low-sample examples | References | Status |
| --- | --- | --- | --- | --- |
| Original `pilot-v1` | 32 / 128 | 2,304 | 512 spp | Historical benchmark |
| `detail-pilot-v2` | 16 / 64 | 896 | 1,024 spp; 12 independent 4,096-spp checks | Generated and integrity-validated |
| `detail-full-v3` | 1,024 / 4,096 | 57,344 | Planned 2,048 spp; 96 independent 8,192-spp checks | Generator/configuration implemented; not yet a completed dataset |

The new pilot has 8/4/4 training/validation/test layouts. Its manifest SHA-256 is
`e782c0dccf9ab0c47169e8bf2ffc8f6e5b3c0f038e2f49c916e58ee42e236894`.
It occupies about 1.4 GiB. It is deliberately small enough to debug training and
screen architectures; four validation layouts cannot establish generalization.

The larger configuration uses 768/128/128 layouts across eight distinct procedural
families: studio, courtyard, corridor, stairs, shelves, arches, terrain, and pavilion.
Each family has edge, texture, lighting, and clutter strata. Cameras, material colors,
texture frequencies, object sizes, occlusion, light count/size/color, and background
radiance vary reproducibly. Four views, two independent input seeds, and seven
sample budgets (1/2/4/8/16/32/64 spp) produce 56 examples per layout. All views,
budgets, noise variants, augmentations, and crops of a layout remain in its split.
These test layouts are unseen; their **families are represented in training**.
Unseen-family, higher-resolution, and specular-transport studies remain separate.

The SSD stores datasets, runs, reports, frozen source, binaries, and logs. A 160-GiB
generation cap and free-space reserve bound the full collection. Compressed size
depends on geometry and noise; the dry-run uncompressed estimate is not a required
allocation. Lossless compact storage shares deterministic center guides and world
positions between a view's examples. Every example retains its own sampled guide
means, coverage, variance, and radiance. Validation checks hashes and reconstructs
the full 27-channel tensor exactly. The SSD must stay mounted during a run.

Generation is resumable at completed examples. A clean source commit, generator
file hashes, renderer binary hash, configuration, independent input/reference seeds,
and output checksums identify the artifacts. A renderer's source commit is recorded
only when supplied by a matching frozen-binary receipt. References are Monte Carlo
estimates, not exact truth; convergence findings must accompany quality claims.

## Augmentation and capacity comparisons

Training-only paired crops, horizontal/vertical flips, and right-angle rotations
transform every input guide and its target together. World-space normals and
positions move to the corresponding pixel; their vector components do not rotate
as if they were screen-space vectors. Exposure varies within ±1.5 stops and RGB
illumination within ±0.2 stops. Linear radiance and target receive the same gain;
variance of the mean receives gain squared. Albedo, counts, and validity stay intact.
Validation and test images receive none of these random transformations.

Independent noisy inputs of the same view and sample budget can be combined using
parallel Welford moments. This produces a valid higher-budget measurement and its
variance, rather than duplicating noise or inventing a Gaussian approximation.
Fusion requires distinct input seeds, identical scene/target identity, and matching
center geometry. Crops and transformations increase training reuse; they do not
increase the number of independent scene layouts reported above.

The configurations in `ml/configs/train/detail/` preserve explicit comparisons:

| Run | Comparison |
| --- | --- |
| C0 | Original width-32 U-Net and loss on new data |
| C0a | C0 plus augmentation and independent-noise fusion |
| C1 | C0a plus edge-focused crops and gradient loss |
| C2 | C1 plus clean-input pairs and HDR energy loss |
| C3 | C2 plus 27-channel boundary features |
| C4 | C3 plus full-resolution refinement |
| C5 | Guided kernel model with C2's training policy and boundary features |
| C6 / C7 | C5 with RGB-only / without variance inputs |
| C8 | C3 with width 64, approximately four times the parameters |

Capacity is an experiment: the original network has about 476k parameters, the
wider candidate about 1.9M, and the guided alternative is much smaller but has an
explicit geometric filtering prior. Parameter count alone is not evidence of
adequacy. Compare train/validation curves, detail and HDR error, memory, and total
runtime. The pilot screens designs; finalists still require larger-data training,
three seeds, held-out tests, and matched-quality timing before model promotion.

Checkpoint selection records SSIM and edge error against a-trous alongside loss.
Constraint penalties influence selection and an explicit eligibility field records
whether a checkpoint meets them. An ineligible best checkpoint is an experiment
artifact, not a qualified model. Every run records optimizer, scheduler, random
states, dataset identity, and epoch-boundary resume state.

## Implemented safeguards and remaining qualification

Schema 2 adds center albedo/normal/depth, depth variance, normal spread, and center
material support. Primary sample hits are reused from tracing; a single shared
center probe supplies deterministic center guides. Raw-render invariance is tested.
The experimental models preserve unsupported primary pixels and return raw RGB
at 128+ spp for same-resolution output. The 2× detail model has a learned
high-resolution head; its accuracy remains unqualified. FP32 and mixed FP16
exports have numerical parity checks; precision quality requires image evaluation.

Native profiling separates packing, inference, and output copying; input storage
is reused, CPU thread count is configurable, and viewer requests are throttled by
time. These changes need end-to-end measurements. Schema-2 temporal training now
supports 2–8-frame differentiable autoregressive windows, paired sequence-level
augmentation, a masked temporal-change loss, and complete validation rollouts.
History comes from predictions; targets enter only supervised losses. Reprojection
uses four geometry-validated taps, center normals/albedo, footprint-scaled position
tolerance, and translation/orientation/roll/FOV cut checks. Explicit sequence frame
numbers distinguish a stationary camera video from cumulative still-image passes.
Native/Python moving and stationary sequence parity has functional tests; long
rollout quality remains unqualified. Adaptive sampling, bounded tiling, additional
transport support, long-sequence studies, and full release qualification remain
work in the [issue plan](../../docs/neural-reconstruction-improvement-plan.md).
