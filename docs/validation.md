# Local validation

Checked on macOS 26.6.2 / Apple M3 Pro with Apple clang 21 and Xcode 26.6.

| Check | Result |
| --- | --- |
| Release CMake desktop build and CTest | 3 suites passed: core, CLI/PNG, SDL viewer |
| Release headless build and CTest | 2 suites passed; executable has no SDL dependency |
| AddressSanitizer + UndefinedBehaviorSanitizer | Both headless suites passed |
| ThreadSanitizer | All 11 core groups passed, including worker teardown and cancellation |
| Headers compiled independently | 23 headers passed |
| Native Xcode Debug and Release builds | Passed |
| Native Xcode geometry tests | 3 passed |
| Native Xcode UI automation | Intermittent window-discovery failures; not treated as a pass |
| Native viewer checks through desktop interaction | Progressive display, pause, scene switching, restart, save, and quit verified |
| Gallery and benchmark | Three images inspected; all 9 benchmark renders had identical PNG bytes |

The core groups include geometry and BVH equivalence, finite light visibility,
material sampling, independent numerical integration of area-light energy,
deterministic execution, concurrency lifecycle, validation, and three reference
images. CLI tests independently decode PNG files and validate their CRCs and
metadata, exercise errors, and compare worker/traversal configurations.

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
