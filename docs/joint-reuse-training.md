# Train joint reconstruction from existing renders

**Completed study:** the initial 50 epochs finished, and the extension toward 100
stopped at epoch 69 under patience ten. Epoch 59 is the selected best; full quality
requirements remain unmet. See the [comparison gallery and charts](../ml/reports/joint-reconstruction-results.md).

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

## Extension to 100 total epochs

An additional 50 epochs is authorized **after the current run completes epoch 50**.
The active source and 50-epoch configuration remain pinned. If patience ten stops
the parent early, report its results before considering an extension; the extension
command refuses incomplete, paused or early-stopped parents.

The original cosine schedule reaches zero at epoch 50. Changing `epochs` in its
configuration is not an exact resume and fails the normal contract check. Use the
explicit `extend-training` preparation with
[`joint-mac-reuse-100.yaml`](../ml/configs/train/joint-mac-reuse-100.yaml). It changes
only the total epoch limit and initial learning rate: a new 50-epoch cosine starts
at 0.00003, one tenth of the parent's initial rate. This rate is a conservative
continuation setting, not a measured optimum or a guarantee of improvement.

After the parent controller has finished, prepare a seed in a separate directory:

```sh
rtml extend-training \
  --parent-run "$ssd/runs/joint-mac-reuse-s42-v1" \
  --config ml/configs/train/joint-mac-reuse-100.yaml \
  --output "$ssd/preparations/joint-mac-reuse-s42-extension-seed"
```

This command performs **zero optimizer updates** and reads no rendered arrays.
It locks the parent against training, verifies its full completion and immutable
epoch receipt, compares the entire latest state to that archive, and writes a new
`seed.pt` plus `extension.json`. It preserves model weights, Adam moments and step
counts, random state, validation selection, elapsed training time and the patience
counter. The changed schedule/configuration receives a new contract and explicit
parent provenance. Parent checkpoint bytes are never overwritten.

The subsequent, separate training run uses the seed with `train --resume-from`,
or a new pinned existing-data controller plan containing `extension_seed` and
`extension_seed_sha256`. That controller performs its normal preflight, loads the
seed once, and uses the new run's latest checkpoint for subsequent clean time-cap
resumes. Global epoch numbering continues at 51 and ends at 100; validation rules,
data splits and sealed test are unchanged. No renderer is involved.

The extension keeps the parent's selected best until a new checkpoint improves
selection. An older `best_resume.pt` retains its **original** configuration and
contract; it must be resumed with that configuration. Use the extension's
`latest.pt` to continue its extended schedule. Best/latest and new epoch archives
remain portable, and the recorded lineage identifies the parent and schedule
change. Tests cover preserved optimizer/RNG state, exact interrupted continuation,
the learning-rate trajectory, retained older best states and rejected unsafe
extensions, using tiny CPU fixtures rather than new ray-traced images.

SSD source snapshots use named local branches `codex/snapshot-<revision>`, pinned
to their recorded commits. Do not advance those branches during an experiment.
Their commits are also retained in the published development branch. A CI test
or documentation correction on the development branch does not require changing
the source of an already running experiment.

The new source/manifest/augmentation contract requires a fresh training run.
It must not resume the old denoising-only optimizer or the completed pilot's
optimizer as though their data and schedules were unchanged.

An input created by reducing saved pixels cannot establish native rendering
speedup. The benchmark command rejects this composite dataset. Native rendering
qualification must use original native pairs under a separately authorized
benchmark; no such rendering is launched as part of reuse preparation or training.
Image comparisons label synthetic inputs, and their copied source render times
are not reported as low-resolution render timings.
