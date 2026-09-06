# Recorded performance

Measured on an Apple M3 Pro (12 logical CPUs), macOS 26.6.2, on September 6,
2026. CMake Release build with Apple clang 21, `-O3 -DNDEBUG`, C++20, and no
fast-math option. The scene contains 484 spheres at 320 × 180, 16 samples/pixel,
maximum depth 12, and seed 42.

| Traversal | Workers | Median render time | Observed range | Speedup vs. linear / 1 |
| --- | ---: | ---: | ---: | ---: |
| Linear | 1 | 1.9611 s | 1.9250–1.9932 s | 1.00× |
| BVH | 1 | 0.6978 s | 0.6947–0.7508 s | 2.81× |
| BVH | 8 | 0.1987 s | 0.1786–0.4922 s | 9.87× |

The script performs one unrecorded single-worker BVH warmup, then runs each
configuration three times sequentially. The timer includes worker startup,
tracing, pass synchronization, and framebuffer publication. Scene construction,
BVH construction, PNG encoding, and the headless polling interval are outside
that timer; BVH build times are recorded separately in the raw data.

All nine runs traced 2,430,316 path rays and produced the **same PNG SHA-256**:

`b30ab7a99b8e3faf2d94bf5d07362a9182764355ef9fb3cdf1f103dd6f22bb70`

These are measurements on a normal desktop, not a controlled benchmark machine.
The first multi-worker run was slower than the following runs; the table includes
that full range and uses the median. Small images expose scheduling overhead,
and worker scaling depends on scene, image size, hardware, and system load.
The comparison holds the corrected integrator constant; it does not compare
against the historical renderer's different lighting model.

[Raw runs and metadata](benchmarks.json) include every time and configuration.
Reproduce with:

```sh
python3 scripts/benchmark.py --binary build/raytracer --width 320 --samples 16 --depth 12 --threads 8 --repeats 3 --output renders/benchmark.json
```
