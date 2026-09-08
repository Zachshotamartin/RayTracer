# Mac full-data training preparation

Status (2026-09-08): **stopped at the user's request after 33 completed epochs**.
The selected best is epoch 31. Latest/best full-state checkpoints were verified
unchanged after stopping the trainer; automatic continuation is paused. Do not
resume this run. The next experiment is [joint reconstruction](joint-reconstruction.md).
The tables below retain the original run preparation and checkpoint contract.

All 57,344 examples passed the file/schema/split audit on 2026-09-07. The dataset has
4,096 reference images and 96 legacy scalar-only independent reference checks; the
independent check images were not retained by that frozen generator. Their regional
convergence remains unqualified. The validation receipt is retained with the SSD
preparation packet under `RayTracer/preparations/full-mac-guided-s42-v1`.

## Proposed first run

| Setting | Proposal |
| --- | --- |
| Machine | This Mac, Apple MPS GPU, FP32, two CPU threads |
| Dataset | External SSD `RayTracer/datasets/detail-full-v3` |
| Dataset identity | `921a69991b3d243097601c72d409ea5c7b973651a9eda873de390d04da4aece3` |
| Split | 43,008 train / 7,168 validation / 7,168 sealed test examples |
| First model | Guided width 16, three stages, center guides, 9,003 parameters |
| Recipe | [`full-mac-guided.yaml`](../ml/configs/train/detail/full-mac-guided.yaml) |
| Optimizer | AdamW, initial learning rate 0.0003, weight decay 0.0001, gradient clip 1 |
| Schedule | 50 total epochs; cosine schedule remains identical across pauses |
| Review policy during the run | Review epochs 1/2 without pausing; later superseded by the explicit user stop |
| Work per epoch | 5,376 optimizer steps at batch 8, random paired 96×96 crops |
| Validation | 2,560 complete images at 1/4/8/16/64 spp, first input noise realization |
| Preservation check | All 512 validation views, independently combined 32+64 spp inputs |
| Per invocation cap | 7,200 seconds in the training loop; initial data audit is additional |
| Early stopping | Ten completed epochs without a selected-checkpoint improvement |
| Storage reserve | At least 10 GiB free; no automatic checkpoint pruning |
| Proposed output | `/Volumes/Zach's SSD/RayTracer/runs/full-mac-guided-s42-v1` |

The guided model is the strongest established structural baseline from our pilot and
paired controls. This first run measures the benefit of the larger collection while
retaining its known recipe. Its limited filtering support and raw material fallback
remain limitations. The conditioned additive U-Net/refinement models are separate
candidate comparisons; increasing capacity has not yet beaten this baseline. This
run does not claim all rendering concerns are solved.

Training applies aligned flips, right-angle rotations, exposure/color changes with
correct variance scaling, edge-biased crops and independent-noise fusion. Validation
uses whole images without those random transformations. The sealed test images are
checked for integrity but never used for gradient updates or checkpoint selection.

The first two epochs provide a measured wall-time and learning-health baseline.
The MPS preflight performs inference only. It does not estimate backward-pass peak
memory, convergence time or rendering acceleration. Data loading currently uses the
existing deterministic single-process implementation. No remote GPU is involved.

## Best and latest are both restartable

| File | Purpose |
| --- | --- |
| `latest.pt` | Full state at the most recently completed and validated epoch |
| `best_resume.pt` | Full training state at the selected best epoch |
| `checkpoints/epoch-000001.pt` | Immutable full state after epoch 1; each subsequent epoch is retained |
| `checkpoints/epoch-000001.json` | Epoch, step, contract, byte size and SHA-256 receipt |
| `best.pt` | Selected model weights and evaluation metadata, for evaluation/export |
| `best_eligible.pt` | Best weights that passed all configured development gates, when any pass |
| `best_score.pt` | Lowest selection-score weights, even if gates failed |
| `checkpoint-status.json` | Latest/best epoch and whether the selected model passed its gates |
| `metrics.jsonl` | Epoch losses, learning rate, image metrics, constraints and learning diagnostics |
| `lineage.json` | Parent file, hash and restart position when continuing into a new run directory |

Full state includes weights, AdamW moments, scheduler position, epoch/step counts,
early-stopping state, Python/NumPy/Torch/MPS and data-loader RNG state, the dataset/config
contract, and model-selection snapshots. Latest/epoch files also embed a nonrecursive
copy of the best full state, allowing recovery without the original best alias.
Copying one full-state file is sufficient for a restart with the matching dataset,
configuration and source. The dataset itself is not embedded in the checkpoint.

Best selection prioritizes candidates passing all development gates: SSIM within
0.005 of a-trous, edge error at most 1.05× a-trous, whole and model-region HDR MSE
at most 1.01× a-trous, and 96-spp preservation log error at most 1.02× its raw input.
Among eligible checkpoints, the lowest selection score wins. If none qualify,
the retained best is explicitly diagnostic. These tolerances allow small regressions;
they are development rules, not a certificate of improved quality or a release gate.

New writes are flushed and atomically replaced. Immutable history, best-state
snapshots and resume repair protect prior completed work. A run lock rejects a second
writer to the same output. Keep the SSD attached while running. An abrupt stop or
time cap during an epoch preserves earlier committed checkpoints and replays that
partial epoch on resume. There are no mid-epoch recovery points. A safe pause waits
for training, validation and checkpoint publication to finish for the current epoch.

## Commands, only after approval to fit

Use the prepared frozen source/environment and resolved config recorded in the SSD
preparation receipt. The source freeze is separate from the dataset's older frozen
renderer. A continuation restores optimizer and scheduler state; it is not a new
random initialization or a warm start from weights alone.

```sh
RTML_DATA="/Volumes/Zach's SSD/RayTracer/datasets/detail-full-v3"
RTML_RUN="/Volumes/Zach's SSD/RayTracer/runs/full-mac-guided-s42-v1"
RTML_CONFIG="ml/configs/train/detail/full-mac-guided.yaml"

# Initial approved stage: stop after two epochs of the 50-epoch schedule.
caffeinate -i ml/.venv/bin/rtml train --config "$RTML_CONFIG" \
  --data "$RTML_DATA" --output "$RTML_RUN" --max-new-epochs 2

# Inspect available recovery points and file hashes; this does not train.
ml/.venv/bin/rtml checkpoints --output "$RTML_RUN"

# Continue the next two epochs after reviewing the initial stage.
caffeinate -i ml/.venv/bin/rtml train --config "$RTML_CONFIG" \
  --data "$RTML_DATA" --output "$RTML_RUN" --resume --max-new-epochs 2

# Request a pause after the current epoch is safely saved.
ml/.venv/bin/rtml pause-training --output "$RTML_RUN"

# Branch from the selected best into a NEW directory, preserving the original.
caffeinate -i ml/.venv/bin/rtml train --config "$RTML_CONFIG" \
  --data "$RTML_DATA" --output "${RTML_RUN}-from-best" \
  --resume-from "$RTML_RUN/best_resume.pt" --max-new-epochs 2

# Or choose a specific completed epoch. Substitute an existing filename.
caffeinate -i ml/.venv/bin/rtml train --config "$RTML_CONFIG" \
  --data "$RTML_DATA" --output "${RTML_RUN}-from-epoch-2" \
  --resume-from "$RTML_RUN/checkpoints/epoch-000002.pt" --max-new-epochs 2
```

Omitting `--max-new-epochs` continues toward the configured total or early stop,
subject to the per-invocation time cap. The cap is checked between training batches,
not during a validation pass, so wall time can exceed it by initial integrity
validation and the current validation/checkpoint pass. Checkpoints are written each
epoch regardless of whether best-model quality improves.

The recipe cannot be silently changed on resume, including its 50-epoch schedule.
Extending the total training horizon or changing optimization is a separate experiment
design decision. Resuming a run already at its epoch or early-stopping limit does not
perform more updates. CPU tests verify exact state recovery; bitwise equivalence
across devices, library versions or nondeterministic MPS kernels is not promised.

Before approving continuation, inspect losses and actual model-region error, gradient
health, SSIM/edge/HDR gates, 96-spp preservation, and representative validation images.
Compare the best and latest checkpoints explicitly. Final full validation, unseen
test evaluation, reference convergence work and uncontended native timing still
precede any replacement of the released model. None is automatically promoted.
