# Checkpoint and evaluation corrections

The selection bug, missing baseline flicker comparison, and discarded independent
reference-check images are corrected. New development recipes also request HDR
and measured-preservation or temporal constraints. These changes improve the
experiment's acceptance process; they do not qualify a replacement renderer.

## Eligible checkpoints take priority

Previously, a failing model with a low reconstruction loss could outrank a passing
model after the finite structural penalties were added. For example, a passing
score of 0.030 could lose to a failing score of 0.015.

The `eligible-first-v1` policy now retains:

- `best_score.pt`: lowest scalar score, including failed candidates, for diagnosis.
- `best_eligible.pt`: lowest score among candidates passing every configured check;
  absent when none passes or no eligibility constraints were configured.
- `best.pt`: the best eligible candidate when available; otherwise the explicitly
  unqualified lowest-score candidate. File existence is not approval to release it.
- `latest.pt`: current training/optimizer/scheduler/RNG state plus immutable model
  snapshots for both selections. This is the authoritative resume checkpoint.

The epoch checkpoint commits before aliases are published. Resume repairs missing,
corrupt or stale aliases from those snapshots. Failing candidates cannot reset
early-stopping patience after an eligible candidate exists. Legacy resume can
recover the retained historical best only; discarded historical models cannot be
recreated. Original frozen experiments and selected files have not been rewritten.

## Explicit development quality constraints

Every epoch records each requested measurement, baseline, limit and pass/fail
result. Missing required measurements or insufficient coverage fail eligibility.
Unknown threshold names and invalid values are rejected before training.

| Check | Measurement and comparator | Fresh development recipe |
| --- | --- | --- |
| Structure | SSIM and display-space edge gradient MAE against a-trous | Both recipes retain SSIM tolerance 0.005 and edge ratio 1.05 |
| HDR | Whole-image linear MSE and conditional model-region linear MSE against a-trous, using the same input support mask | Both require ratios ≤1.01 |
| Near-clean preservation | Log-radiance MAE against the independent reference, compared with the measured near-clean input's error | C14 requires ratio ≤1.02 over at least 16 validation views |
| Temporal stability | Confidence-weighted, reference-motion-corrected log-radiance changes against a-trous | Temporal quality recipe requires ratio ≤1.02 over at least 128 valid transitions |

Use [C14](../configs/train/detail/c14-quality-selection.yaml) for a **new** spatial
run or [the temporal quality recipe](../configs/train/detail-temporal-quality.yaml)
for a new autoregressive run. Their tolerances are explicit development policy,
not noise-calibrated release margins. Original C13/temporal configurations retain
their old constraint sets for experiment identity. These new recipes have not
completed full training or passed the broader [release gates](../../docs/neural-reconstruction-improvement-plan.md#i-release-only-after-the-new-evidence-passes).

Near-clean validation deterministically combines independent 32- and 64-sample
streams of the same validation view, yielding 96 measured samples below the
128-spp raw bypass. It uses no augmentation, crop or target substitution, consumes
no training RNG, and compares complete images. References enter only scoring.
The temporal collection lacks these higher-sample input pairs; its recipe does
not claim near-clean preservation qualification. Temporal constraints require
autoregressive full-sequence validation, rather than unrelated spatial batches.

## Comparable temporal evidence from the completed model

The saved temporal development model was rescored without training or inference.
The original 512 predictions cover four validation layouts, 32 frames, two noise
streams and 4/16 spp. The reference-corrected change is
`(current prediction - warped previous prediction) - (current reference - warped previous reference)`;
the log version applies `log1p` before each subtraction, after RGB warping.
All methods use identical input-derived geometry, reprojection weights and confidence.

There are **488 measurable transitions** and 24 unmeasured initial/cut frames,
with 3,449,668 valid pixel observations. Cuts and unmatched frames contribute no
measurement, rather than artificial zero errors. Scene means are averaged equally
in the table; complete image-weighted statistics and scene-bootstrap intervals
are retained in the evidence.

| Method | Linear temporal MAE, scene mean | Log temporal MAE, scene mean |
| --- | ---: | ---: |
| Raw | 0.017447 | 0.014201 |
| A-trous | 0.004990 | 0.003993 |
| Trained temporal reconstruction | 0.003712 | 0.002965 |

The neural model's log temporal error is **25.8% lower than a-trous** on this
development cohort. This comparison does not establish moving-object or reflected
motion performance, persistent ghost-trail absence, a gain over a separately
qualified spatial neural model, or a runtime advantage. The independent reference
noise floor remains unresolved. Four layouts and one trained seed are limited
evidence. No sealed test predictions were inspected by this audit.

Evidence: [summary, hashes and per-budget statistics](detail-development/temporal-baselines.json)
and [all per-image records](detail-development/temporal-baselines-per-image.jsonl.gz).
The saved-prediction [audit tool](../tools/audit_temporal_metrics.py) verifies dataset,
checkpoint and evaluation identity, refuses sealed-test audits and existing output
paths, and records prediction/source hashes. Fresh `rtml evaluate` reports the
same metrics for raw/a-trous/neural and optional OIDN, with explicit coverage.
The old neural-only `temporal_residual_mae` remains available for historical
compatibility; new comparable metrics are versioned and confidence weighted.

## Independent reference images are retained

New generation saves each selected higher-spp check as a float32 NPZ before
committing its version-2 receipt. Receipts contain both reference identities,
actual independent seeds, sample counts, archive hashes, whole-image disagreement
and edge/highlight/dark/flat/geometry-region disagreement with pixel counts and
95th-percentile squared error. Masks come from the primary reference, so their
own uncertainty must also be considered.

Generation resume and dataset validation verify the retained check's pixels,
checksum, pairing, independent seed, larger sample count and recomputed regional
metrics. Missing expected archives fail validation. Disk accounting includes their
writes. Older scalar-only receipts are counted separately and remain usable as
legacy datasets, without claiming retained pixels or regional convergence.

The two active frozen generators continue with their original code and budgets.
Their old check images cannot be recovered from scalar receipts. Regional audits
of those collections require separately retained rerenders; increasing sample
counts, establishing acceptance margins and censoring unresolved references remain
research work. No old receipt has been relabeled as a new retained-image check.

## Verification and remaining work

The complete suite passed 91 tests, with two optional OpenImageIO tests skipped.
It includes native ONNX parity, exact augmented resume, eligibility versus scalar
ranking, alias corruption/recovery, deterministic measured preservation, reference
archive integrity/resume, temporal mask and stream isolation, and comparable
training/evaluation metrics. Additional targeted checks cover offline-audit parity
and refusal to inspect sealed test results. Ruff also passes.

The dominant raw-fallback HDR error needs a separately validated material/fallback
or sampling change. Broader family/asset/transport and resolution cohorts,
reference convergence, three-seed finalists, progressive previews, peak memory
and uncontended matched-quality timing remain open. These are not resolved by
adding gates. The bundled release model is unchanged.
