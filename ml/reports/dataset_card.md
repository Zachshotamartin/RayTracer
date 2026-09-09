# Dataset card — pilot-v1

Generated locally on an Apple M3 Pro for the first custom reconstruction experiment.
No external training images, pretrained weights, downloaded meshes or artist assets
were used. Scenes are deterministic procedural rooms from this repository.

## Contents and splits

| Property | Value |
| --- | --- |
| Configurations | 32 layouts × 4 camera views = 128 |
| Input images | 2,304 at 256×144 |
| Budgets | 1, 2, 4, 8, 16, 32 samples/pixel |
| Noise realizations | 3 per view/budget |
| Targets | 128 independent 512-spp images; maximum path depth 16 |
| Split | 22 / 5 / 5 whole layouts for train / validation / test |
| Examples per split | 1,584 / 360 / 360 |
| On-disk dataset | Approximately 3.1 GiB, compressed float arrays |
| Generation caps | 10 GiB and 7,200 seconds per invocation; one renderer, 8 workers |

The full pilot test is an **unseen-layout cohort within the same procedural room
family**, not general unseen-scene or arbitrary-asset performance. Extra held-out
camera and new-light cohorts are implemented by `cohort_groups`, and exercised in
the separate cohort smoke dataset. They were not part of this pilot's training or
full evaluation.

Each room has diffuse walls, two spheres, a rotated box, one rectangular emitter,
low constant environment illumination and a pinhole camera. Albedo, geometry,
light color/position and cameras vary across layouts. Object count and topology
remain simple. There are no textures, animation, glass, metal, caustics or defocus.

## Pairing, features and reference quality

Inputs and targets use distinct deterministic sampling seeds with the same complete
scene/camera JSON. Noise variants and sample budgets stay with their parent group.
17 float32 NCHW channels are raw linear RGB (3), albedo (3), world-facing normal (3),
mean primary-ray distance, hit coverage, diffuse support, variance of the mean (3),
sample count and variance availability. First-hit features average the same jittered
rays as radiance. Misses contribute zero. Normals are averaged, not renormalized;
depth is coverage-weighted. Variance is Welford M2 / [N(N−1)]; at 1 spp it is zero
with availability false. Separate center-ray positions support temporal reprojection.

Four 512-spp targets were compared against independent 2048-spp renders. Mean
linear MSE was **1.72×10⁻⁵**; this is disagreement between two noisy estimates, not
an exact ground-truth error estimate. The subset is the first four camera views of
one layout. It does not establish convergence of every layout or rare light path.
[Reference checks](pilot-reference-checks.json) preserve each measurement.

Validation checks schemas, paths, checksums, dimensions, finite/range constraints,
independent seeds, target budgets, exact scene identities, complete expected count,
and split leakage. Native tests verify that collecting features leaves raw output
unchanged. The selected comparison below is the first test configuration at 4 spp,
noise realization 0; it is not a best-case image selection.

| Raw path tracing · 4 samples/pixel | A-trous denoising · 4 samples/pixel |
| :---: | :---: |
| ![Raw path tracing at 4 samples per pixel](figures/pilot-4spp-raw.png) | ![A-trous denoising of the same 4-sample input](figures/pilot-4spp-atrous.png) |
| **Our trained U-Net · 4 samples/pixel** | **Independent reference · 512 samples/pixel** |
| ![Custom U-Net reconstruction of the same 4-sample input](figures/pilot-4spp-neural.png) | ![Independent raw reference at 512 samples per pixel](figures/pilot-4spp-reference.png) |

Each panel is 256 × 144. The three methods share a 4-spp input; the target is an
independent 512-spp reference. Fixed exposure 0, ACES-fit + sRGB display; metrics
use original float images. The [results index](README.md) explains the error view
and links the complete records.

## Provenance and distribution

- Full manifest SHA-256: `420fbc507b85dc016a80f002f5686249f2427c6430f666db77748e10020e2dce`.
- Generation binary SHA-256: `9cd8468c0fab5a90c915ac612c14022b1f99c66d9a02076be329ed50b840efa0`.
- [Compressed original manifest](pilot-manifest.jsonl.gz), [dataset contract](pilot-dataset.json),
  [split groups](pilot-splits.json), and [training environment](pilot-environment.json) are in Git.
- The renderer and Python package were being implemented in an uncommitted working
  tree based on `b6883a9`. That recorded commit alone does **not** reproduce this
  development binary. The final source, exact binary hash and original artifact
  hashes are retained; new generation requires a fresh dataset directory.
- The development renderer's `stats.scene_seed` incorrectly echoed the noise seed
  for scene-file renders. Complete scene JSON and `scene_sha256` are authoritative;
  geometry did not change with that seed. The shipped renderer reports null there
  for custom JSON. The original manifest is deliberately preserved unchanged.

Large arrays remain under local ignored `artifacts/datasets/pilot-v1`, and can be
regenerated from the checked-in configuration. The compressed manifest includes
original temporary paths for provenance; those paths are not required to use it.
There is no project-wide license declaration in this repository. Project asset
licensing should not be inferred from the separately licensed dependencies.
# Current reuse collection

The current full-data experiment uses `joint-reuse-v1`, a 59,992-example
composition of saved HDR renders. See the [reuse data card and training contract](../../docs/joint-reuse-training.md)
for source counts, inherited splits, synthetic area-reduction semantics,
aligned augmentation, native validation, storage and limitations. Its builder
traces zero new rays. The historical dataset evidence below remains unchanged.
