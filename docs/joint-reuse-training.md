# Train joint reconstruction from existing renders

The full regeneration job was stopped at the user's request on 2026-09-09 UTC.
The replacement uses the saved HDR arrays, geometry buffers, reference images
and independent noise streams. No new ray-traced images are needed to build it.
The stopped experiment, original datasets and all checkpoints remain intact.

## Sources and storage

`datasets/joint-reuse-v1` combines these SSD collections:

| Existing collection | Use |
| --- | --- |
| `detail-full-v3` | All 57,344 saved noisy examples and 4,096 high-quality references; derive 2× pairs from existing pixels |
| `joint-variety-pilot-v2` | All 1,920 native 2× pairs, retaining native validation and test cases |
| `joint-variety-full-v1` | Complete saved views from the interrupted generation job; never resume its renderer |

The built snapshot contains **59,992 examples**: 44,376 training, 7,808 validation
and 7,808 sealed test. It retains 728 completed native pairs from the stopped job.
No new images were rendered, and no existing reference was enlarged to fabricate
target detail. The 50-epoch recipe has 5,547 training batches per epoch, 3,904
first-noise validation inputs and 576 independent measured preservation views.

`rtml reuse-data` hard-links the original arrays on the same SSD. This adds file
entries and manifest snapshots rather than another image collection. It has no
renderer argument, invokes no renderer and has no automatic copy fallback onto
another volume. Original source metadata and checksums are retained. Treat both
source and linked arrays as immutable; atomic replacement of a file does not
update its other hard links. Validation rejects changed payloads or metadata.

```sh
rtml reuse-data \
  --source "$ssd/datasets/detail-full-v3" \
  --source "$ssd/datasets/joint-variety-pilot-v2" \
  --source "$ssd/datasets/joint-variety-full-v1" \
  --output "$ssd/datasets/joint-reuse-v1"
```

Here `$ssd` is the existing RayTracer directory on the SSD. The command snapshots
complete saved views. An interrupted view lacking some noise/budget pairs is
recorded explicitly and left in the original dataset. Selection never uses image
quality, and no difficult complete view is removed. All variants inherit the
original geometry group's train/validation/test assignment; sealed test stays
out of fitting and checkpoint selection.

## Pair construction and augmentation

For the original 256×144 collection, a 2×2 area reduction of the **noisy input**
produces a 128×72 input. The 256×144, 2,048-spp reference stays at its original
detail level. A deterministic crop or 90° rotation per camera supplies square,
landscape, portrait and wide frames. The same pixel transform applies to the
reference, guides and annotation masks. Existing native 2× pairs are unchanged.
Nothing from a clean reference is injected into input radiance or input guides.

The reduction handles features separately from ordinary RGB resizing:

- Radiance and sampled geometry moments use non-overlapping box averages.
- Four source pixels contribute four times the original path count. Old 1–64-spp
  inputs therefore become **4–256 contributing paths per synthetic input pixel**,
  not native 1–64-spp low-resolution renders.
- For more than one sample per source pixel, variance of the reduced mean is
  the sum of the four source mean variances divided by 16. With one sample per
  stratum, a conservative between-stratum estimate includes spatial variation.
- Depth variance includes within- and between-pixel moments; normal spread is
  recomputed. Center guides select a saved probe instead of blending surfaces.
  Its offset from the new pixel center is a documented synthetic-domain limitation.
- The Python à-trous baseline filters the reduced input, then enlarges it with
  bilinear interpolation. It does not use the already denoised high-resolution
  image. Its filter was checked against a saved native C++ baseline.

The [training recipe](../ml/configs/train/joint-mac-reuse.yaml) draws aligned
48×48 base crops, emphasizing edges and borders, then chooses one shape per
batch: 48×48, 32×48, 48×32, 32×32 or 40×40. Targets always have twice those
dimensions. This varies field of view and patch context without blurring or
stretching reference pixels. Flips, 90° rotations, exposure, white balance,
positive RGB filters and independent-measurement fusion vary each epoch.
Radiance gains also scale variance by their square. World-vector components
stay in world coordinates when their pixels move.

These augmentations reuse observations; they do not create new geometry or
reveal hidden scene content. Synthetic reduction does not establish performance
on native low-sample inputs. Native and synthetic metrics are tracked separately,
including their individual sample-budget slices. All existing global, resolution,
PSNR, SSIM, edge, HDR and preservation quality gates remain strict.

## Execution and evidence

The existing-data controller plan uses `data_mode: existing` and pins the reuse
manifest, dataset descriptor, training configuration and source revision. It
rejects renderer or generation configuration fields. It validates existing
artifacts and MPS shapes, then trains with batch eight toward 50 epochs, cosine
decay and patience ten. Best/latest/full epoch checkpoints retain exact optimizer,
scheduler and RNG resume state. Clean time limits continue automatically; epoch
one and two reviews do not pause healthy training.

The new source/manifest/augmentation contract requires a fresh training run.
It must not resume the old denoising-only optimizer or the completed pilot's
optimizer as though their data and schedules were unchanged.

An input created by reducing saved pixels cannot establish native rendering
speedup. The benchmark command rejects this composite dataset. Native rendering
qualification must use original native pairs under a separately authorized
benchmark; no such rendering is launched as part of reuse preparation or training.
Image comparisons label synthetic inputs, and their copied source render times
are not reported as low-resolution render timings.
