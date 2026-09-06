# Image references

These 64 × 36 RGB PPM images use seed 42, 32 samples per pixel, depth 12, and the
named scene presets. They are checked after the ACES fit and sRGB conversion.
The comparison permits mean channel error below 0.75/255 and fewer than 0.5% of
channels with error above 8/255, allowing small floating-point differences across
compilers while catching substantive visual regressions.

Regenerate only after intentionally changing image formation or a preset:

```sh
./build/raytracer_tests --update-references
```

Inspect the new renders before accepting them. This is separate from the analytic
geometry, radiometry, sampling, and traversal checks: a generated reference image
alone does not establish physical correctness.
