# Local validation

Checked on macOS 26.6.2 / Apple M3 Pro with Apple clang 21 and Xcode 26.6.

| Check | Result |
| --- | --- |
| Release CMake desktop build and CTest | 4 suites passed: core, rendering features, CLI/images, SDL viewer |
| Release headless build and CTest | 3 suites passed; executable has no SDL dependency |
| AddressSanitizer + UndefinedBehaviorSanitizer | All 3 headless suites passed |
| ThreadSanitizer | All 11 core and 5 rendering-feature groups passed, including photon preparation, worker teardown, and cancellation |
| Headers compiled independently | 27 headers passed |
| Native Xcode Debug and Release builds | Passed |
| Native Xcode geometry tests | 3 passed |
| Native Xcode UI automation | Intermittent window-discovery failures; not treated as a pass |
| Native viewer checks through desktop interaction | Progressive display, pause, scene switching, restart, save, and quit verified; new denoiser, caustic scene, and glass-shadow modes verified |
| Gallery | Original three images plus the imported mesh and caustic scene inspected; transparent-shadow comparison checked in the viewer |
| Historical v2.0 benchmark | All 9 benchmark renders had identical PNG bytes under the previous PNG encoding |

The core groups include geometry and BVH equivalence, finite light visibility,
material sampling, independent numerical integration of area-light energy,
deterministic execution, concurrency lifecycle, validation, and four reference
images. The three original references remain byte-identical. Rendering-feature
tests cover OBJ/MTL import, triangle/BVH agreement, denoiser error reduction and
edge preservation, finite glass transmission, photon-energy normalization,
independent path-traced energy comparison, and photon-prepass cancellation.
CLI tests independently decode PNG, RGBE/RLE HDR, and PFM files; validate CRCs,
row order, radiance above one, metadata, and exposure-independent linear exports;
exercise errors; and compare worker/traversal configurations.

## Interactive macOS test limitation

The default **RayTracer** Xcode scheme runs deterministic native geometry tests.
Choose **RayTracer UI** to run the interactive launch and keyboard tests on an
unlocked desktop available to the test runner. They use SDL's software presenter
to avoid an occluded Metal window waiting for a drawable.

On the machine above, Xcode sometimes reports the running app as a disabled
accessibility application containing only its menu bar, so it cannot find the
SDL window. A launch test passed in one run and failed in later runs. Direct
desktop interaction discovered the window and verified the controls. The UI
tests retain their assertions and remain separate from the default scheme;
they are not silently skipped or counted as successful.

GitHub Actions runs headless checks on Linux, macOS, and Windows, plus SDL and
address/undefined-behavior checks on Linux. Those jobs do not require an
interactive macOS desktop.
