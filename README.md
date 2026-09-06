# RayTracer

A C++20 CPU path tracer with a progressive SDL viewer and a portable headless
renderer. It traces diffuse, reflective, and refractive materials; samples area
lights with multiple importance sampling; and renders reproducibly across CPU
worker counts.

![Jade and brass studio: textured sphere, glass, and a rotated metal box](docs/renders/studio.png)

*Studio · 960 × 540 · 256 samples/pixel · depth 16 · seed 42.
Rendered by this project. Full settings and timing: [studio.json](docs/renders/studio.json).*

## Build and run

Requires a C++20 compiler with latch support and CMake 3.21 or newer.
The macOS repository includes SDL2. On Linux or Windows, install SDL2's development
package or select the headless build below.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
./build/raytracer
```

On Windows with the Visual Studio generator, the executable is
`build/Release/raytracer.exe`. Point `CMAKE_PREFIX_PATH` at an SDL2 installation if
CMake cannot find it. On Debian/Ubuntu, the development package is `libsdl2-dev`.

**Without SDL or Xcode:**

```sh
cmake -S . -B build-headless -DCMAKE_BUILD_TYPE=Release -DRAYTRACER_ENABLE_SDL=OFF
cmake --build build-headless --config Release --parallel
./build-headless/raytracer --scene studio --width 960 --samples 128 --output renders/studio.png
```

The headless build has no SDL, image-library, or asset-download requirement.
CMake presets `release`, `headless`, and `sanitize` are also available when Ninja
is installed.

**Xcode:** Open `RayTracer.xcodeproj`, select the shared **RayTracer** scheme,
and Run. The project uses local ad-hoc signing and no personal development team.
The scheme runs in the project directory, so default exports go into `renders/`.
CMake and Xcode compile the same rendering sources.

If a local build warns about nonexistent SDL search paths, check your shell's
`LIBRARY_PATH`. The repository does not need hard-coded Homebrew version paths.
For example, `env -u LIBRARY_PATH cmake --build build` avoids stale shell entries
on macOS/Linux.

## Desktop controls

The preview refines over successive sample passes. The HUD shows the scene,
render status, completed samples, elapsed wall time, worker count, and exposure.

| Key | Action |
| --- | --- |
| `1`, `2`, `3` | Switch to demo, sphere field, or studio |
| `Space` | Pause/resume |
| `R` | Restart the current scene with the same seed |
| `S` | Save the latest complete pass as PNG + JSON |
| `Esc` | Cancel rendering and keep the preview |
| `Q` / window close | Quit |
| `+`, `-` | Adjust display exposure by 0.25 stops |

Completed renders export automatically. Use `--output` to choose a path;
otherwise the destination is `renders/SCENE.png`. Saving or rerendering to the same
path replaces that image and its metadata. Exposure changes affect preview/export
without restarting the paths; press `S` to save the adjusted image.

When Finder launches the app with `/` as its working directory, exports instead
use `~/Library/Application Support/RayTracer/renders/`. An explicit `--output`
always takes precedence.

## Reproducible renders

```sh
./build/raytracer --headless --scene field --width 960 --samples 128 --depth 16 --seed 42 --threads 8 --output renders/field.png
```

Headless mode prints machine-readable render metadata on stdout and progress on
stderr. Add `--quiet` to suppress progress. Interrupting with Ctrl-C saves the
last completed sample pass when one is available and exits with code 130.

Use `--help` for all options. Defaults are studio, width 640, 64 samples/pixel,
depth 16, seed 42, automatic worker count, BVH enabled, and exposure 0.
`--no-bvh` enables the linear traversal baseline. `--quit-after-render` closes
the desktop viewer after exporting.

The same seed and settings produce identical image bytes across worker counts
and BVH/linear traversal within a build. JSON sidecars record all rendering
options needed to reproduce the image. Compiler/platform floating-point
differences are handled with tolerances in the reference-image tests.

## Scenes

| Preset | What it demonstrates |
| --- | --- |
| `demo` | Original five-sphere setup, hollow glass, mirror metal, and directional lighting |
| `field` | Hundreds of seeded spheres, three material families, depth of field, and BVH scaling |
| `studio` | Procedural textures, GGX metal, rotated geometry, glass, soft shadows, and two area lights |

![Restored sphere-field scene](docs/renders/field.png)

*Sphere field · 960 × 540 · 128 samples/pixel · depth 16 · seed 42.
[Render metadata](docs/renders/field.json).*

The field's distribution and camera settings were recovered from commit
`488bb80`. Its explicit RNG and updated material models replace the older global
RNG and fuzzy-metal approximation, so it is a reproducible restoration of that
scene rather than a pixel-identical reconstruction of the historical screenshot.

![Original demo with corrected lighting](docs/renders/demo.png)

*Demo · 960 × 540 · 64 samples/pixel · depth 16 · seed 42.
[Render metadata](docs/renders/demo.json).*

Rebuild the entire gallery:

```sh
python3 scripts/render_gallery.py --binary build/raytracer
```

## Engineering changes

- Correct entry/exit intersections and face normals for translated and rotated boxes.
- Finite point/area-light shadow segments and explicit parallel slab handling.
- Additive emission, direct illumination, and indirect scattering; BSDF/light MIS.
- Lambertian textures, GGX conductors, ideal mirrors, Fresnel glass, and emissive quads.
- Median-split BVH and persistent workers processing 16 × 16 tiles.
- Per-pixel/sample random streams, independent of scheduling and worker count.
- Complete-pass publication, cancellation, pause/resume, and owned SDL resources.
- Shared display/export conversion, portable PNG writing, and JSON sidecars.
- CMake/CTest, native Xcode tests, and CI for Linux, macOS, and Windows.

See [architecture and numerical details](docs/architecture.md) for the estimator,
threading model, component map, tests, and limitations.

## Performance

[Recorded benchmark results](docs/benchmarks.md) compare linear traversal on one
worker, BVH on one worker, and BVH on multiple workers. Every timed configuration
must produce the same PNG SHA-256 before the script reports a speedup.

```sh
python3 scripts/benchmark.py --binary build/raytracer --width 320 --samples 16 --depth 12 --threads 8 --repeats 3
```

Results include individual runs, medians, CPU/platform information, build time,
ray counts, and settings. Timings measure this corrected renderer with different
acceleration settings; they are not a comparison against the old renderer's
different lighting model.

## Tests

```sh
ctest --test-dir build -C Release --output-on-failure
```

Python 3 enables the CLI/PNG integration suite. The C++ suite covers intersections,
BVH equivalence, light visibility, material sampling, energy against numerical
quadrature, deterministic output, pause/cancellation, and three image regressions.
The SDL suite tests texture updates, keyboard controls, and shutdown with its
dummy driver. Xcode's default **RayTracer** scheme runs native geometry tests.
The separate **RayTracer UI** scheme contains interactive launch/control tests
and needs an available desktop. See [validation results](docs/validation.md) for
the current Xcode UI automation limitation and manual viewer checks.

Address and undefined-behavior sanitizers:

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DRAYTRACER_ENABLE_SDL=OFF -DRAYTRACER_SANITIZERS=ON
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure
```

[Reference-image policy](tests/reference/README.md) explains tolerances and
intentional regeneration. C++ formatting is defined in `.clang-format`.

For a separate race-checking build, use `-DRAYTRACER_THREAD_SANITIZER=ON` instead
of `-DRAYTRACER_SANITIZERS=ON` with Clang/GCC. Do not combine the two sanitizers.

## Scope and attribution

This started by following
[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).
The progressive application, reproducible execution, corrected box/light
extensions, tests, benchmarks, and studio scene build on that foundation.
[The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html) and
[The Rest of Your Life](https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html)
are references for acceleration and sampling.

The current renderer uses analytic geometry and a finite path depth. It has no
mesh importer, denoiser, HDR export, or dedicated caustic sampler; shadow rays treat
glass as opaque. PNGs use uncompressed DEFLATE for dependency-free portability.
SDL2 remains under its bundled upstream license.
