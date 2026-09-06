# Rendering extensions

## Mesh import

`mesh.cpp` uses pinned tinyobjloader source to parse OBJ/MTL and produce triangles
that participate in the same BVH as analytic primitives. Intersections are
two-sided, use double precision, and interpolate UVs and supplied vertex normals.
Geometric normals determine face orientation and ray offsets; shading normals
are oriented consistently with the geometric surface. Missing vertex normals
use flat shading. Invalid indices/non-finite positions are rejected, degenerate
triangles are skipped with a warning, and an empty usable mesh is an error.

The command-line mesh studio centers the mesh in X/Z, places its minimum Y at
zero, and scales its longest axis to two units. Basic MTL diffuse, metallic, and
dielectric materials are mapped to renderer materials. MTL image textures and
emitter registration are not implemented; warnings make those omissions visible.
The importer does not promise watertight intersections or robust tessellation
of every self-intersecting/non-planar polygon. Pre-triangulate production assets.

## Image output

PNG keeps RGB8, the existing ACES/sRGB transform, and an sRGB chunk. Its IDAT
stream is now compressed with the vendored stb DEFLATE encoder. CRC, zlib
decompression, pixel determinism, and compression of a flat fixture are tested
independently in Python. Compression changes file bytes; it does not change the
three original RGB reference images. The recorded v2.0 benchmark's SHA-256 is
historical and belongs to the former uncompressed PNG encoding.

Radiance HDR stores scene-linear RGBE using scanline RLE where supported by the
format. RGBE provides high dynamic range with shared-exponent quantization.
PFM stores scene-linear RGB as little-endian IEEE float32, bottom row first;
values are rounded once from the renderer's doubles. No exposure, tone mapping,
or gamma conversion is applied to either linear format. Tests independently
decode known values, including radiance above one, row order, and black, and
verify that changing exposure leaves HDR/PFM image bytes unchanged.

## Denoising

The filter is an original implementation of geometry-guided spatial a-trous
filtering, inspired by [Dammertz et al.](https://cg.ivd.kit.edu/publications/pubhanika/2010_atrous.pdf).
Three passes use a separable B3-spline kernel with spacings 1, 2, and 4. Depth,
normal, albedo, and luminance differences reduce weights across boundaries.
Only diffuse first-hit pixels are filtered. Accumulation and reference samples
remain unchanged, so the viewer can switch between raw and filtered images.

The first pass collects deterministic pinhole guides separately from jittered
path samples. Their cost is included in render time. Filtering is a separate
postprocess, timed in export metadata, and the viewer limits preview updates to
avoid filtering every arriving pass. It is spatial and biased: there is no
temporal reprojection or learned prior, and strong depth of field can invalidate
the guide correspondence. Synthetic noise tests verify mean-square error
reduction and preservation of an albedo edge and excluded specular pixels.

## Caustic transport and path ownership

The photon map covers paths beginning at a registered light, passing through
one or more ideal specular vertices, and arriving at the first diffuse surface:
`L S+ D`. Other paths remain with the original estimator. Flux emission weights
include light-selection probability and the position/direction sampling PDFs.
Point emission uses `4*pi*intensity`; a cosine-sampled quad uses
`pi*area*emission`; directional emission covers the scene's bounding disk.
Weights are divided by the total attempted photon count, not the stored count.
Dielectric photon transport uses importance/flux weights without the radiance
eta-squared factor used by camera paths.

Stored positions, incoming directions, normals, fluxes, and specular depths are
indexed into spatial cells. A fixed-radius Epanechnikov disk kernel
`2*(1-distanceSquared/radiusSquared)/(pi*radiusSquared)` estimates incident flux
density. The receiving BSDF converts it to outgoing radiance. Normal and tangent
plane checks reduce leakage; they do not eliminate every boundary artifact.
Photon depth must fit the remaining camera-path budget.

When the eye path reaches a registered emitter through the same diffuse/specular
sequence, its emission contribution is suppressed because the photon gather
already estimates that path class. Directly visible emitters, camera-specular
paths without a diffuse receiver, non-specular indirect transport, and the
environment remain with the ordinary estimator. Analytic thin-glass energy,
photon-count normalization, and an independent area-emitter path-traced
reference check this partition.

The map is prepared by one owned worker while other workers wait at a latch.
Cancellation interrupts that work and releases all workers. No incomplete
photon result is published as a completed camera pass. The immutable completed
map is shared by the tile workers. Preparation time and photon-ray counts are
reported separately; preparation is also included in total render time.
Repeated camera passes share a fixed map. This is finite-radius photon mapping,
not stochastic progressive photon mapping. Its bias/noise can dominate even
after many camera samples, and nearby surfaces can leak energy at large radii.
The implementation follows the particle-transport principles described in
[Physically Based Rendering](https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping).

## Transparent shadow approximation

The optional straight shadow connection multiplies Fresnel transmission across
dielectric interfaces, respects total internal reflection, stops at opaque
surfaces, and retains the finite endpoint for point/area lights. It deliberately
does not refract the connection. Connections that cross glass receive light-only
weight rather than competing with a BSDF sample of a different, bent path.
The corresponding purely transmitted BSDF-to-emitter contribution is suppressed.

This approximation cannot reproduce focusing or exact thick-glass transport.
It is incompatible with caustic mapping so that both do not estimate the same
transmitted illumination. Choose photon mapping for refracted caustics, or the
approximation for a cheap transmissive shadow preview.
