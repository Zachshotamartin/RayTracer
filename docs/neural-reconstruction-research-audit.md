# Research alignment audit

Reviewed 2026-09-06 against project commit `1a5249a` and Open Image Denoise
2.5.1, commit `6602ee2ca38a1a2a02135beed8f6e68eed630180`.

The project follows a supported research direction: supervised reconstruction of
low-sample path-traced measurements using auxiliary buffers. It does **not** yet
have a validated training recipe or a qualified replacement for the released
model. Its networks are experimental adaptations, not paper reproductions.
The [issue plan](neural-reconstruction-improvement-plan.md) describes the release
gates; the [study record](../ml/reports/detail-study.md) distinguishes implemented
machinery from measured results.

## What the sources support

| Primary source | Relevant practice | Relationship to this project |
| --- | --- | --- |
| [Bako et al., SIGGRAPH 2017](https://la.disneyresearch.com/publication/deep-learning-denoising/), [paper](https://studios.disneyresearch.com/wp-content/uploads/2019/03/Kernel-Predicting-Convolutional-Networks-for-Denoising-Monte-Carlo-Renderings-Paper33.pdf) | Learn local filtering kernels from noisy radiance and auxiliary features; separate diffuse/specular treatment and albedo demodulation help their setting. | Our guided model follows the kernel idea. Its three small learned filtering stages and hand-set geometry penalties differ substantially from their deeper network and 21×21 kernels. It is not KPCN reproduced. We currently lack separate diffuse/specular training buffers. |
| [Chaitanya et al., SIGGRAPH 2017](https://research.nvidia.com/publication/2017-07_interactive-reconstruction-monte-carlo-image-sequences-using-recurrent) | Recurrent reconstruction uses prior frames and auxiliary information to improve low-sample sequences. | Our prediction-history rollouts address the training/deployment mismatch. RGB reprojection with a learned gate differs from their recurrent autoencoder. Longer unseen sequences, ghosting, cuts and temporal quality still need evaluation. |
| [OIDN 2.5.1 training instructions](https://github.com/RenderKit/oidn/blob/v2.5.1/README.md#training) | Train on paired noisy/reference renders, including multiple budgets/seeds. Match auxiliary noise to deployment and tune the learning-rate range. | We generate these pairs ourselves. Historical OIDN results use pretrained weights. A separate same-data, 200-epoch toolkit control has now completed, with an explicitly labeled small-image loss adaptation. |
| [OIDN dataset code](https://github.com/RenderKit/oidn/blob/v2.5.1/training/dataset.py) | Paired random crops, flips, transposes and channel permutations reuse images; preprocessed tensors are memory-mapped. | Our crops/flips/rotations are paired. Our illumination and independent-seed fusion augmentations are separate choices requiring ablation. Augmentation does not create new independent scene layouts. Our compressed per-example loading also needs a throughput comparison. |
| [OIDN configuration](https://github.com/RenderKit/oidn/blob/v2.5.1/training/config.py), [loss](https://github.com/RenderKit/oidn/blob/v2.5.1/training/loss.py), [model](https://github.com/RenderKit/oidn/blob/v2.5.1/training/model.py) | RT defaults use a perceptual HDR transfer, an L1/MS-SSIM mixture, and U-Nets selected by quality mode. | Our log/relative/gradient/energy losses, two-pooling-level U-Net and learning rate are custom choices. Parameter count alone cannot establish comparable capacity or training quality. |

## Concrete findings from our code and experiments

1. **Larger has not meant better under the tested recipe.** The width-64 candidate
   has 1,898,627 parameters but failed the development structural gate. That result
   does not prove a larger network is unsuitable; optimization and representation
   may limit it. All initial runs use one seed and short training budgets.
2. **The guided prior does much of the work.** On all 224 pilot validation inputs,
   its untrained control scored 34.27 dB / 0.9609 SSIM; training improved this to
   35.55 / 0.9638. A-trous scored 33.09 / 0.9579, and pretrained OIDN 36.99 / 0.9682.
   Learned weights add about 1.27 dB over the fixed prior. Four validation layouts
   are insufficient evidence for broad generalization. Linear HDR MSE has not
   improved over a-trous, and these contended evaluation timings are not speedups.
3. **The bounded multiplicative U-Net head restricts missing-signal recovery.**
   Schema 2 computes `(raw + .01) * exp(2 * tanh(correction)) - .01`.
   A zero input channel can therefore approach only 0.0639, regardless of its
   neighbors. This is a mathematical limitation, not yet a measured explanation
   of all failed predictions. C9 now exposes an additive log-radiance head under
   the same guides, support policy, initialization and training budget; its
   functional tests establish signal recovery, not trained image quality.
4. **C3 is not a feature-only ablation.** Selecting schema 2 also changes the output
   head, support mask and high-sample blending. Preserve its screening result, but
   use C10's encoder-boundary-channel control against C9 before attributing an
   effect to boundary guides. C2 likewise combines identity sampling and an energy loss.
   C6 changes encoder features and manual filtering penalties together; it also
   retains the support policy. It is not a complete removal of geometry from output.
5. **A hard identity rule cannot demonstrate learned clean-input preservation.**
   At 128+ spp the schema-2 same-resolution model returns raw RGB. Synthetic clean
   examples marked 128 spp supply no reconstruction gradient to that model.
   C11/C12 now replace those synthetic examples with measured 96-sample inputs
   formed from independent 32/64-sample streams. Variance and sampled guides are
   combined consistently. Tests establish a nonzero training gradient; trained
   preservation quality remains to be measured. The hard fallback stays separate.
6. **Counts overstate diversity if read without their grouping.** The large
   collection targets 57,344 inputs, but they come from 1,024 layouts and 4,096
   viewpoints. All eight procedural families appear in training. New assets,
   unseen families, specular transport and resolution cohorts remain necessary.
7. **Reference quality is still incompletely qualified.** Higher-sample independent
   checks exist; new generators retain their images and regional disagreement,
   while frozen collections have scalar-only receipts. Region convergence,
   difficult-light escalation and unresolved-target exclusions are
   still required before small quality gains can be trusted.
8. **A stalled identity solution can be diagnosed before a large run.** Fixed-patch
   controls identified radiance conditioning as a contributor. C13 scales internal
   radiance and variance consistently and restores physical output units. New
   training-health measurements compare prediction and raw loss on identical
   batches. The [diagnosis report](../ml/reports/learning-diagnosis.md) records the
   controls and their limits; the failed models are not silently replaced.
9. **The previous supported-region HDR metric used an outdated mask.** New reports
   use the actual custom-model policy and report additive error contributions.
   About 99.38% of C5's whole-image HDR error lies in its raw-fallback region on
   the four-layout validation cohort. Its model-region error is slightly below
   a-trous, while whole-image error remains higher. Region-specific reference
   convergence and fallback qualification are still required.
10. **A scalar penalty was not a hard eligibility rule.** A low-loss failing model
    could displace a passing checkpoint. Eligible-first selection now prevents
    that, and fresh recipes add explicit HDR, measured preservation or temporal
    constraints with coverage requirements. The [correction report](../ml/reports/qualification-fixes.md)
    records crash-safe selection, retained reference checks and a comparable
    raw/a-trous/neural temporal audit. The existing temporal model reduces the
    scene-mean motion-corrected log error by 25.8% versus a-trous on four validation
    layouts; this does not qualify moving objects, broad generalization or speed.

## Next experiments and acceptance criteria

The current frozen screening runs remain unchanged. Their records must not be
rewritten to imply cleaner causal comparisons than were actually run.
The [completed upstream control](../ml/reports/oidn-training-control.md) reached
36.13 dB / 0.9542 SSIM on the full pilot validation split. It trails the guided C5
in SSIM and HDR error despite higher PSNR, so this audit has not established a
uniformly superior recipe. The next controls and larger cohorts remain necessary.

- **Upstream control:** export the original measured linear color, sampled albedo
  and sampled normals into OIDN's EXR naming convention, retaining scene splits,
  seeds and hashes. Default exports contain training and validation only. Use the
  unmodified pinned toolkit for preprocessing, training, inference and export.
- **Resolution control:** the existing 256×144 pilot cannot use OIDN's default
  five-scale loss: its [implementation](https://github.com/RenderKit/oidn/blob/v2.5.1/training/ssim.py)
  requires the smaller patch dimension to exceed 160. The small-image control uses
  its supported `l1_grad` loss with 128-pixel crops. Generate the same layouts at
  512×288 for a separate 256-pixel-crop, default-loss comparison. Do not enlarge
  the small images and report that as new rendered detail.
- **Optimization:** save learning-rate range curves; choose candidate ranges from
  training/development data. Compare longer runs by optimizer updates, training
  pixels and compute as well as epochs. Freeze validation-based selection before
  final tests. An LR sweep is a diagnostic, not proof of an optimal recipe.
- **Architecture and loss controls:** isolate the bounded head, guide policy,
  geometric prior, HDR transfer, gradient/multiscale objectives and near-clean
  training. Compare capacity only after the basic training can fit representative
  small data. Preserve failures and compare both train and validation curves.
- **Final qualification:** use three seeds for finalists, scene-group uncertainty,
  reference-region audits, untouched challenge cohorts and separate temporal/2×
  tests. Require detail and HDR constraints, then compare total time to matched
  quality against a-trous and warmed OIDN, including feature work and memory.

The pinned toolkit contains an MPS device path, while its README still lists Linux
as the supported training OS. Mac runs are therefore a locally tested use of that
code path, not an assertion of upstream platform support. Any compatibility edits
must be separately recorded; no upstream source edits are part of this control.
