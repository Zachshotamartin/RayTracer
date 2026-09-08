# Joint Mac pilot: start receipt

Recorded 2026-09-08. This is the initial review, not a live progress page or a final
model qualification report. Full records live under `/Volumes/Zach's SSD/RayTracer`.

| Item | Recorded value |
| --- | --- |
| Run | `runs/joint-mac-pilot-s42-v1` |
| Training code | `a9d8255d8d9f34152a422c43c5a9052b2326d488` |
| Generator code | `2ecfd9c650fc67cca651baf5e22966bbe89622cc` |
| Renderer | `renderers/raytracer-f2ecc9f`; later commits changed Python/config/docs |
| Data | `datasets/joint-variety-pilot-v2` |
| Manifest SHA-256 | `c8ee7f0b90370a24c063cbb810888201ab181c5168311419457d3c47852e3ad5` |
| Training contract | `16aecc94c1b0f4f8708dc6ce0c39a008a4973f08a4fd0feb429791b106ba9b1b` |
| Device/model | MPS, FP32, 508,172 parameters, joint denoising and 2× reconstruction |
| Schedule | 30 epochs, 80 updates per epoch, patience eight; no pause for early reviews |
| Data split | 640 train / 640 validation / 640 sealed test inputs; 32 independent layouts and 64 native views per split |
| Per-epoch validation | 256 full frames at 1/4/16/64 spp, plus 64 measured near-clean views |

The old same-resolution run remains stopped at 33 complete epochs; its latest and
best full resume checkpoint hashes were verified unchanged. The new run starts
from new weights, with a different learning task and data contract.

All 1,920 generated examples passed integrity validation. MPS preflight checked the
actual training batch and every validation shape without optimizer updates. A
separate untrained feasibility check produced finite 720p, 1080p, 1440p and portrait
outputs; this does not measure trained image quality or native renderer acceleration.

Each training and validation split contains all eight scene families and all four
detail strata. The training variants include 40 diffuse, 13 metal and 11 glass
scenes; validation includes 53 diffuse, four metal and seven glass variants. These
are scene categories, not a guarantee of visible material coverage at every pixel.

Eight retained train/validation reference pairs were compared at 2,048 versus
8,192 spp: mean PSNR 50.20 dB, mean SSIM 0.9918, minimum PSNR 33.71 dB and minimum
SSIM 0.9604. They passed the recorded sample audit. This does not establish that
every target or high-energy region is converged. Sealed-test quality was not used
to select the reference budget or tune the model.

| Completed epoch | Updates | Training loss | Validation loss | Validation PSNR | Validation SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 80 | 0.032931 | 0.028176 | 25.41 dB | 0.8144 |
| 2 | 160 | 0.032491 | 0.026640 | 25.68 dB | 0.8157 |

Raw-plus-bilinear validation loss was 0.028956. Learning was finite and validation
loss improved, but both initial checkpoints failed qualification: PSNR/SSIM and
edge comparisons, plus resolution and budget slices, remained insufficient. These
results do not show that the model beats the denoiser. Healthy training continued.

Both initial epoch archives matched their SHA-256 receipts. Their full model,
optimizer, scheduler, Python/NumPy/Torch/loader/MPS RNG and selected-best resume
states were checked. `latest.pt`, `best_resume.pt` and immutable `checkpoints/`
remain available; `best.pt` is the selected inference alias and may be ineligible.

The five-minute monitor targets this new run, reviews meaningful changes without
pausing healthy training, and reports completion or failures. Initial local and
native ML validation passed 128 tests with two optional OpenImageIO tests skipped;
GitHub build/tests passed for training code `a9d8255`.

See [the project review](../../docs/joint-reconstruction.md#project-review-and-remaining-evidence)
for remaining capacity/data, detail, large-image, motion and matched-quality timing
evidence. None is inferred from a falling training loss.
