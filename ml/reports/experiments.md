# Experiment ledger

All models below were trained from scratch on this renderer's paired samples.
Configurations, artifact contracts, summary metrics and limitations are retained.
The test results were not used to retrain or select a replacement pilot model.

| Run | Purpose | Result / decision |
| --- | --- | --- |
| `smoke-v1` | Test 36-example generation, MPS training, export and native parity | Pipeline works. Tiny width-16 model barely improves raw PSNR; a-trous/OIDN are much stronger. Not a recommended model. |
| `pilot-v1` | Width-32 U-Net, 2,304 examples, 50 epochs, seed 42 | 476,419 parameters; 18.7 minutes training. Validation-selected epoch 48. Test PSNR 35.96 versus raw 25.84, a-trous 33.97, OIDN 39.44. Custom SSIM below a-trous. Bundled as the research baseline. |
| `upscale-smoke-v1` | Paired 64×36→128×72 training/evaluation and dynamic ONNX/native output | 2× path works and preserves fallback semantics. Small-run results are insufficient to establish a resolution or timing advantage. |
| `temporal-smoke-v1` | Trainable 21-channel history model on 60 examples, with camera/light cohorts | Reprojection, autoregressive evaluation and native sequence parity pass. Tiny model is weaker than a-trous; no flicker or ghosting improvement is claimed. |
| `pilot-v1` native timing | 20 test views × 5 budgets × 3 methods × 4 frame repeats | Custom model meets joint 32 dB/0.90 SSIM target on all views. A-trous is faster at that target on every view. See timing report. |

Machine-readable summaries and compressed per-image records accompany this ledger.
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
