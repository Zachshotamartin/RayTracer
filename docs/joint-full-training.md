# Full joint reconstruction experiment on the Mac

**Stopped and superseded:** the user stopped this regeneration experiment on
2026-09-09 UTC. Do not restart its plan. The active approach is
[training from existing renders](joint-reuse-training.md), including crops and
augmentations. The workload and commands below are historical provenance.

The user authorized the larger run on 2026-09-08 after the 30-epoch pilot finished.
This experiment tests the same joint denoising/2× reconstruction architecture with
substantially more data and context. It is a fresh run with a new data contract;
the completed pilot and older same-resolution checkpoints remain intact.

## What the pilot established

On 256 validation inputs, the selected epoch-30 checkpoint reached 27.16 dB PSNR
versus 25.90 dB for à-trous plus bilinear enlargement. Its whole-frame HDR error was
12.6% lower. SSIM was worse (0.8424 versus 0.8721), and edge error was 2.5% higher.
The model failed qualification; these results do not establish rendering speedup.

| Input budget | PSNR gain over à-trous + bilinear | SSIM change | HDR error ratio | Edge error ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 spp | −0.68 dB | −0.1051 | 1.073 | 1.104 |
| 4 spp | +0.72 dB | −0.0462 | 1.025 | 1.059 |
| 16 spp | +2.19 dB | +0.0052 | 0.652 | 0.980 |
| 64 spp | +2.80 dB | +0.0272 | 0.532 | 0.935 |

The full run targets the weak low-sample regime with additional 2- and 8-spp
measurements and far more examples. Keeping the 508,172-parameter architecture
avoids assuming that a larger network alone will solve the problem. Larger crops
provide more spatial context; this is a scale-up experiment, not a controlled
single-variable ablation. Data sufficiency and final model capacity remain measured
questions rather than guaranteed outcomes.

## Declared workload

| Setting | Full run |
| --- | --- |
| Dataset | `joint-variety-full-v1` |
| Run | `joint-mac-full-s42-v1` |
| Independent layouts | 1,024: 768 train / 128 validation / 128 test |
| Native views | 4,096: two cameras × two resolution/camera variants per layout |
| Examples | 57,344: 43,008 train / 7,168 validation / 7,168 sealed test |
| Sample budgets | 1, 2, 4, 8, 16, 32, 64; two independent noise streams |
| Targets | Native 2× width and height, 2,048 spp, independent seeds |
| Reference checks | 96 retained 8,192-spp checks; report train/validation agreement separately |
| Native input sizes | 128×72, 192×108, 256×144, 384×216, 192×192, 128×192, 336×144, 256×192 |
| Augmentation | Native camera zoom/roll/lighting variation plus aligned crop/flip/rotation/exposure/color/noise fusion |
| Training | MPS FP32, 64×64 input / 128×128 target crops, batch eight |
| Schedule | 50 epochs, 5,376 updates per epoch, patience ten, cosine decay |
| Validation | 3,584 complete frames per epoch plus 512 measured preservation views |
| Execution limits | Two hours per generation/training invocation; automatically continue clean time caps |
| Storage | 360-GiB dataset cap; 20-GiB experiment/training free-space reserve |

The input-only uncompressed estimate is 247.3 GiB. Compact storage and compression
reduce actual disk use; reference/guide/metadata files add storage. The 360-GiB cap
allows the collection without assuming a favorable compression ratio. The SSD had
about 765 GiB free at preparation time.

A pixel/sample-scaled projection from pilot CPU renders estimated approximately
59 hours for the full set of references alone at the original six-worker setting.
Inputs, check images and file processing add time. Different scenes, thermal state
and worker scaling make this a rough projection, not an ETA. Early full-size
measurements should replace it. Data generation and optimizer training are
distinct phases; the full model cannot train on an incomplete manifest.

## Execution and recovery

[`joint-variety-full.yaml`](../ml/configs/data/joint-variety-full.yaml) and
[`joint-mac-full.yaml`](../ml/configs/train/joint-mac-full.yaml) define the workload.
The SSD preparation at `preparations/joint-mac-full-s42-v1/plan.json` pins the source
commit, Python package hashes, both configurations, renderer checksum and output
directories. Invoke the controller from that frozen source with the workspace venv:

```sh
ml/.venv/bin/rtml run-experiment --plan "/Volumes/Zach's SSD/RayTracer/preparations/joint-mac-full-s42-v1/plan.json"
```

The launcher supplies the frozen `PYTHONPATH` and keeps the Mac awake during the
job. `preparations/joint-mac-full-s42-v1/state.json` and `events.jsonl` distinguish
rendering, validation, training, completion, pauses and failures. The run log is
`logs/joint-mac-full-s42-v1.log`. The five-minute monitor watches the same controller
and must not launch another renderer, audit or trainer while it is active.

Generation continues after a real time cap only when new examples were committed.
An unexpected timeout, checksum failure or lack of progress stops the controller.
Completed data passes integrity/device preparation before fitting. Retained
train/validation reference agreement is reported after generation invocations and
before training; difficult targets are flagged, not silently removed. Reference
agreement and model qualification remain separate measurements.

Training saves `latest.pt`, `best_resume.pt`, the selected inference alias `best.pt`,
and immutable `checkpoints/epoch-*.pt` full states. Clean training time caps resume
the exact completed-epoch state. Partial epochs are replayed. Epoch-one/two reviews
do not pause healthy optimization. A user pause, early stopping, completed schedule,
runtime error, corrupted checkpoint or unexplained disappearance does not trigger
an automatic training restart. No checkpoint is promoted automatically.

The controller lock prevents duplicate jobs. `STOP_EXPERIMENT` in the preparation
directory prevents the next stage/invocation; during training, `rtml pause-training`
requests a stop after the current epoch. These controls have different boundaries.
An immediate user stop requires terminating the verified active process, preserving
already committed examples and epoch checkpoints.

## Qualification

Global, per-resolution and per-budget PSNR/SSIM/edge/HDR constraints are unchanged
from the pilot, with preservation coverage raised to 512 views. Low-sample and
texture failures must remain visible in reports. Sealed test is excluded from
tuning. Full-size image quality, motion stability, additional model/data ablations
and native matched-quality timing remain necessary before claiming a useful renderer
acceleration. A long or successfully completed training run is not sufficient.
