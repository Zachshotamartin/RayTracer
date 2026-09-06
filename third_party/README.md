# Vendored libraries

These headers compile into the portable renderer; they do not require package
installation or additional runtime libraries. Their complete license notices
are preserved inside the files.

The importer selects tinyobjloader's supported portable number parser with
`TINYOBJLOADER_DISABLE_FAST_FLOAT`. Its embedded fast_float implementation fails
MSVC's C++20 constexpr validation at this pinned revision. All platforms use the
same parser, and the vendored header remains unchanged.

| Library | Pinned upstream commit | Use | License |
| --- | --- | --- | --- |
| [stb_image_write](https://github.com/nothings/stb/blob/2c980bb59875b0d32144a71867fbdebb2f77cd20/stb_image_write.h) | `2c980bb59875b0d32144a71867fbdebb2f77cd20` | PNG DEFLATE compression and Radiance RGBE export | MIT option |
| [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader/blob/45636bdcef1a4fec140346b90c0b50bf0bc3e23b/tiny_obj_loader.h) | `45636bdcef1a4fec140346b90c0b50bf0bc3e23b` | OBJ/MTL parsing and triangulation | MIT |

SDL2 is distributed separately in the existing `SDL2.framework` with its upstream
license notices. The renderer's wavelet filter, photon mapping implementation,
triangle intersections, and PFM export are project code.

| Additional library | Pinned release | Use | License |
| --- | --- | --- | --- |
| [nlohmann/json](https://github.com/nlohmann/json/releases/tag/v3.12.0) | `v3.12.0`, unmodified single header `json.hpp` | Scene files and model metadata | MIT |

Optional learned inference uses the separately installed
[ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT; tested with 1.29.0).
Its binaries are not vendored. The Python environment's packages retain their
respective upstream licenses, recorded by the lockfile and installed distributions.
The optional [Open Image Denoise](https://github.com/RenderKit/oidn) benchmark uses
its official executable (Apache-2.0; tested with 2.5.1), supplied separately.
The procedural ML scenes and bundled custom weights are project-authored assets.
No project-wide license has been declared; dependency licenses do not license the
project itself. No external image dataset or pretrained weights trained our models.
