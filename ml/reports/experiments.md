# Experiment ledger

Models were trained on this renderer's paired samples. Continuation runs below
resume their named parent rather than training from scratch.
Configurations, artifact contracts, summary metrics and limitations are retained.
The test results were not used to retrain or select a replacement pilot model.

## Completed joint reconstruction and preceding runs

| Run | Purpose | Result / decision |
| --- | --- | --- |
| `full-mac-guided-s42-v1` | Large same-resolution guided baseline | Stopped at user request after 33 complete epochs; best 31. Superseded by joint 2× training. |
| `joint-mac-pilot-s42-v1` | Native multi-resolution joint 2× pilot | Completed 30 epochs; best 30. Native validation 27.16 dB / 0.8424 SSIM; ineligible. |
| `joint-mac-full-s42-v1` | Proposed full native collection | Generation stopped at user request; no training run. Its 728 completed pairs were retained for reuse. |
| `joint-mac-reuse-s42-v1` | Joint model trained on 59,992 existing examples | Completed 50 epochs; best 50. Continued from the verified epoch-46 checkpoint after an interruption. No new renders. |
| `joint-mac-reuse-s42-100-v1` | Lower-LR continuation from epoch 50 toward 100 total | Early stopping at global epoch 69; selected best 59. Best native validation 29.82 dB / 0.9173 SSIM vs a-trous 25.98 / 0.8768. Full qualification unmet; acceleration unproven. |

The **[current gallery, charts and exact metrics](joint-reconstruction-results.md)**
cover the completed joint study. Its sealed test remains unused; no replacement
viewer model has been promoted. Training and scheduled monitoring are stopped.

## Original pilot and smoke studies (historical)

| Run | Purpose | Result / decision |
| --- | --- | --- |
| `smoke-v1` | Test 36-example generation, MPS training, export and native parity | Pipeline works. Tiny width-16 model barely improves raw PSNR; a-trous/OIDN are much stronger. Not a recommended model. |
| `pilot-v1` | Width-32 U-Net, 2,304 examples, 50 epochs, seed 42 | 476,419 parameters; 18.7 minutes training. Validation-selected epoch 48. Test PSNR 35.96 versus raw 25.84, a-trous 33.97, OIDN 39.44. Custom SSIM below a-trous. Bundled as the research baseline. |
| `upscale-smoke-v1` | Paired 64×36→128×72 training/evaluation and dynamic ONNX/native output | 2× path works and preserves fallback semantics. Small-run results are insufficient to establish a resolution or timing advantage. |
| `temporal-smoke-v1` | Trainable 21-channel history model on 60 examples, with camera/light cohorts | Reprojection, autoregressive evaluation and native sequence parity pass. Tiny model is weaker than a-trous; no flicker or ghosting improvement is claimed. |
| `pilot-v1` native timing | 20 test views × 5 budgets × 3 methods × 4 frame repeats | Custom model meets joint 32 dB/0.90 SSIM target on all views. A-trous is faster at that target on every view. See timing report. |

The [results index](README.md) links every summary, compressed per-image record,
comparison image and plot.
The convolutional baseline and RGB/guides/all ablations are implemented through
model configuration; the convolutional pipeline is exercised in CPU tests. A full
ablation matrix and three-seed repeat are future experiments, not completed results.

Validation and testing include:

- Deterministic paired renders, independent target seeds, complete grouped splits,
  checksum/path/schema checks and generator resume contracts.
- Raw-output invariance with feature collection and learned reconstruction.
- CPU uninterrupted versus epoch-boundary-resumed training, compared bit-for-bit.
- Gradient flow and positive finite output for convolutional, U-Net and 2× models.
- Python/ONNX parity on dynamic odd/even sizes and actual C++ feature buffers.
- Temporal scene-change, camera-cut and disocclusion rejection, followed by actual
  two-frame C++ versus Python prediction checks.
- Model metadata rejection and missing-model raw fallback.
- Native SDL keyboard controls, asynchronous final export, sequence completion,
  fallback/shutdown, plus existing conventional renderer tests and ASan/UBSan.
- Manual macOS viewer comparison of raw, AI, matching reference and 4× error.

The reproducibility contract is artifact and configuration based. The development
runs predate their implementation commit; original hashes/records are preserved
without pretending the recorded parent commit contains that implementation.
Tests establish repeatability on tested platforms; MPS/CPU bitwise identity is not
promised. ONNX uses the pinned legacy exporter with `dynamo=False`; its deprecation
warnings are visible in test output and migration can be validated separately.
