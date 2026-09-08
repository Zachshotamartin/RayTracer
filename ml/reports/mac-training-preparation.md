# Mac training readiness, 2026-09-07

Historical preparation record. Update 2026-09-08: that run completed 33 epochs and
was stopped at the user's request; epoch 31 is its selected best. The next task is
[joint denoising and upscaling](../../docs/joint-reconstruction.md).

The full dataset passed its file/schema/split audit. Apple MPS inference passed at
the proposed batch shape, with zero optimizer updates. The first full-data run is
prepared and awaiting approval; no full-data checkpoint exists yet.

The [run plan](../../docs/mac-training-run.md) proposes the established 9,003-parameter
guided baseline for an initial two-epoch stage of a fixed 50-epoch schedule. It uses
43,008 training examples, 2,560 validation images and 512 measured preservation
pairs each epoch, with all test layouts excluded from model selection.

## Verified readiness

- [Dataset integrity receipt](detail-development/full-data-validation.json):
  57,344 examples, 1,024 layouts, 4,096 references; checksums, tensor schemas and split
  isolation passed. The 96 independent checks are legacy scalar-only receipts;
  regional reference convergence remains unqualified.
- Device: MPS, PyTorch 2.14.0, macOS 26.6.2, 18 GiB unified memory.
- Training-shaped forward pass: `8×27×96×96` input to `8×3×96×96` finite output.
- Full-frame forward pass: `1×27×144×256` input to `1×3×144×256` finite output.
- No optimizer was constructed by preflight. These forward passes do not measure
  training throughput, backward-pass memory or learned quality.
- Checkpoint tests recover exact CPU weights, optimizer moments, scheduler and RNG
  from latest, a copied best full-state file and a copied earlier epoch. They preserve
  the original run, repair missing history/best aliases, enforce run locks and reject
  incompatible configs, weight-only restart files and altered archive receipts.
- Full ML/native integration suite: 121 passed, two optional OpenImageIO skips.
  The first unconfigured test invocation found an older `build/raytracer`; the verified
  run explicitly used the current `build-detail/raytracer` for both renderer paths.
- Ruff lint/format and Git whitespace checks passed.

The preflight receipt, dataset audit and frozen launch commands live on the SSD under
`RayTracer/preparations/full-mac-guided-s42-v1`. Training will use that frozen source
with the existing local Python environment and the external dataset. Recovery keeps
each completed epoch and the selected best full state, without automatic pruning.

The initial approval covers two epochs only. Continuation, model comparison and final
qualification are subsequent steps. The existing released ONNX model is unchanged.
