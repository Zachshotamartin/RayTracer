# Native completion timing — pilot-v1

[Results index](README.md) · [Image-quality results](model_card.md)
· [How to rerun the benchmark](../README.md#evaluation-and-benchmarks)

Apple M3 Pro, macOS 26.6.2, Release C++ build, four tracing workers and ONNX Runtime
1.29.0 CPU inference (two intra-op threads). 256×144 output, depth 16, fixed input
noise realization 0, all 20 test views. Each method/budget starts a new process,
loads once, then performs one cold and three warm full frames. Median and p95 are
saved per view. The table below is the median of those per-view warm medians.

| Samples/pixel | Raw | A-trous | Custom model |
| ---: | ---: | ---: | ---: |
| 1 | 7.72 ms | 64.00 ms | 76.69 ms |
| 4 | 22.85 ms | 78.94 ms | 96.55 ms |
| 16 | 81.23 ms | 138.15 ms | 175.76 ms |
| 32 | 159.98 ms | 219.84 ms | 281.96 ms |
| 64 | 325.47 ms | 387.09 ms | 491.96 ms |

Reconstruction adds work at equal sample counts. Its potential benefit is reaching
a useful quality at a smaller sample budget, not making the same rays free.
The 64-spp model evaluation is beyond its 1–32-spp training range.

The declared quality target was **PSNR ≥32 dB and SSIM ≥0.90**, selected after
validation and before native test timing. Both metrics must pass for a view.
The best passing time is selected only from the tested budget grid above:

| Method | Views meeting target | Median best passing time |
| --- | ---: | ---: |
| Raw | 3 / 20 | 333.45 ms on those 3 views only |
| A-trous | 20 / 20 | 78.94 ms |
| Custom model | 20 / 20 | 100.34 ms |

**A-trous was faster than the custom model at this joint quality target on every
tested view.** The custom model therefore does not demonstrate an acceleration
over the existing denoiser in this study. On the three views where raw also passed,
custom reconstruction reached the threshold 3.26–4.32× sooner. That is a restricted
paired result; the 17 failures are retained, and no ratio is invented for them.
Raw might pass with more than 64 samples, which this timing grid did not test.

Warm timing includes BVH construction, path tracing, additional first-hit guide and
statistic collection, tensor packing, CPU inference, snapshot copies and PFM write.
Cold-frame timing additionally includes model loading. Native `process_total_seconds`
includes subprocess startup and all repeats. Warm timing excludes one-time scene
JSON parsing, operating-system startup, display and metadata JSON writing. GUI
rendering/inference contention, first useful displayed preview, device peak memory,
and equal-time quality curves are not measured here. OIDN's separate quality results
use a CLI baseline; its startup+IO time is not compared with warmed native inference.

[Configuration and every matched result](pilot-timing-summary.json) ·
[all timings, cold starts, p95 and image metrics](pilot-timing-per-image.jsonl.gz) ·
[browsable table](pilot-timing-report.html).

The benchmark records the exact measured development binary SHA-256
`b51aaa42db42833074655400b262581bd21e5a34b7367e82443d5a72fd38805e`.
Subsequent changes to help text, output-directory creation and tests are not a claim
of a newly benchmarked binary. Timing is one local run, with no cross-device or
statistical confidence claim. Use the checked-in CLI to measure your hardware.
