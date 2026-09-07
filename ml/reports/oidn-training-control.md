# OIDN training control

Status: development control; no qualified weights from this experiment yet.
This trains a randomly initialized upstream network on our renderer's data.
It is distinct from the pretrained OIDN comparisons in the historical model card.
See the [research audit](../../docs/neural-reconstruction-research-audit.md).

## Protocol

The source is unmodified [OIDN v2.5.1](https://github.com/RenderKit/oidn/tree/v2.5.1),
commit `6602ee2ca38a1a2a02135beed8f6e68eed630180`. The launcher records its training
file hashes and rejects a different or dirty checkout. Its isolated Mac environment
uses Python 3.12.11, Torch 2.14.0, OpenImageIO 3.1.17.0 and TensorBoard 2.21.0.
MPS is present in the source but is not covered by the README's Linux-only training
support statement. Locally successful execution is the relevant evidence here.

The pilot export contains all 448 training and 224 validation examples from
`detail-pilot-v2`, manifest
`e782c0dccf9ab0c47169e8bf2ffc8f6e5b3c0f038e2f49c916e58ee42e236894`.
The exporter verifies source checksums and split isolation, writes float32 linear
HDR/albedo/normal EXRs, and records each source-to-export mapping and output hash.
It preserves sampled normals, including their signs and unnormalized means. The
training and deployment guides are noisy; `clean_aux` is false. References remain
independent raw renders. Test examples are excluded from this export.

Upstream preprocessing supplies its PU HDR transform and memory-mapped TZA files.
No image resizing or target-guide substitution is performed. The 256×144 source
images require a **small-image adaptation**: 128×128 patches and the supported
`l1_grad` loss, not the default five-scale MS-SSIM mixture. The separate
`detail-pilot-512.yaml` configuration renders the same layouts at 512×288 for a
later default-loss study using 256×256 crops. The two resolutions are related data,
not independent scene cohorts.

The initial control uses upstream balanced U-Net, batch 8, FP32, Adam, 200 epochs
(11,200 training updates if uninterrupted), validation/checkpoints every five epochs,
and one-cycle cosine scheduling. This is not a matched-compute comparison with
the 50-epoch custom screen. Compare validation curves and final image metrics;
training duration under concurrent rendering is not an inference benchmark.

Two seeded LR range probes each processed 56 batches, increasing LR from `1e-7`
to `1e-2`. Both showed a substantial loss decrease around `4e-4`–`1e-3`; neither
established a divergence boundary. The first training candidate uses initial
`4e-5` and peak `1e-3`, chosen from these training-only diagnostics. This is a
starting recipe, not a demonstrated optimum. Retain both raw/smoothed curves and
compare validation-selected alternatives before full-data training. The wrapper
sets the Torch RNG before upstream `find_lr.py`, whose accepted `--seed` argument
is otherwise not used to initialize that script's model.

The completed probe records are [seed 42](detail-development/oidn-lr-seed42.csv) and
[seed 43](detail-development/oidn-lr-seed43.csv). The 200-epoch training run is in
progress; these learning-rate probes are not completed denoising models.

## Reproduction

Create an isolated environment with `ml[oidn-training]`, and clone the pinned
source. Set `RTML_ARTIFACTS` to a dataset/run directory, `RTML_OIDN_SOURCE` to the
checkout and `RTML_OIDN_PYTHON` to that environment's Python executable. Run from
the RayTracer repository root; use absolute paths for the variables.

Use the installed `rtml` executable in that environment to export:

```sh
"${RTML_OIDN_PYTHON%/*}/rtml" export-oidn-data \
  --data "$RTML_ARTIFACTS/datasets/detail-pilot-v2" \
  --output "$RTML_ARTIFACTS/datasets/oidn-pilot-v2"

"$RTML_OIDN_PYTHON" "$RTML_OIDN_SOURCE/training/preprocess.py" hdr alb nrm \
  --filter RT --device cpu --train_data train --valid_data valid \
  --data_dir "$RTML_ARTIFACTS/datasets/oidn-pilot-v2" \
  --preproc_dir "$RTML_ARTIFACTS/datasets/oidn-pilot-preproc-v2"

ml/.venv/bin/python ml/tools/oidn_run.py \
  --source "$RTML_OIDN_SOURCE" --python "$RTML_OIDN_PYTHON" \
  --stage train --seed 42 --seconds 7200 \
  --run-dir "$RTML_ARTIFACTS/reports/oidn-pilot-training-s42" -- \
  --config "$PWD/ml/configs/train/oidn/pilot-balanced.json" --device mps \
  --train_data train --valid_data valid \
  --preproc_dir "$RTML_ARTIFACTS/datasets/oidn-pilot-preproc-v2" \
  --results_dir "$RTML_ARTIFACTS/runs/oidn-pilot-v2" --result balanced-s42
```

The launcher stores exact arguments and output in its run directory and terminates
its own process group if the wall-time cap is reached. A run receipt describes the
last recorded state; it cannot prove a process is still alive after an abrupt host
shutdown. Use a new receipt directory when resuming the same upstream result.
Upstream resumes weights/optimizer and reseeds by epoch; this is not bitwise
continuation of an uninterrupted run. Preserve that distinction in comparisons.

For LR probes use launcher stage `find_lr`, omit the training JSON config, and pass
`hdr alb nrm --filter RT --quality balanced --loss l1_grad --tile_size 128
--batch_size 8 --num_loaders 1 --device mps --precision fp32 --lr 1e-7 --max_lr 1e-2`,
along with the same training/preprocessed/result directories. Use separate result
names and receipt directories for seeds 42 and 43.

## Evaluation still required

Select a checkpoint using validation only. Run upstream inference with exposure
derived from the noisy input, then compute our full-frame/region metrics from its
linear outputs under the same display transform as other methods. Upstream's own
tonemapped metrics must not be placed directly in our tables as equivalent scores.
Record the different support policies: upstream predicts the entire image, while
custom candidates preserve unsupported pixels. Qualify the default-loss,
higher-resolution control separately. No held-out test selection, full-data model
promotion, or runtime advantage is established by these initial training jobs.
