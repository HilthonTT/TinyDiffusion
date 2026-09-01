# Architecture

How TinyDiffusion is put together, and why each piece is where it is. For
running it, see [USAGE.md](../USAGE.md); for installing it, see
[INSTALL.md](INSTALL.md).

- [The shape of a run](#the-shape-of-a-run)
- [Module map](#module-map)
- [The forward process](#the-forward-process)
- [The backbone](#the-backbone)
- [Parameterisation](#parameterisation)
- [Conditioning](#conditioning)
- [Sampling](#sampling)
- [Weight averaging](#weight-averaging)
- [Scaling out](#scaling-out)
- [Measurement](#measurement)
- [Reading order](#reading-order)

## The shape of a run

```
config (TOML)  ──▶  training/config.py  ──▶  training/train.py
                                                │
   data/datasets.py ── batch ──────────────────▶│
                                                │  noise to a random t
   diffusion/schedules.py ── coefficients ─────▶│  score the prediction
                                                │
   models/unet.py ◀── (x_t, t, class) ──────────┤
                                                │
                          training/ema.py ◀─────┤  shadow weights
                                                │
                    per epoch: checkpoint, sample grid, metrics.jsonl
                                                │
                                                ▼
        sampling.py / evaluation.py / metrics/ / server/ / tui/
```

Training and inference share one contract: a checkpoint carries the config that
produced it, so `sample`, `eval`, `fid`, `interpolate` and `serve` rebuild the
same model and the same process without being told how.

## Module map

| Module | Holds |
| --- | --- |
| `cli/` | `parser.py` declares every subcommand's flags, `options.py` turns their text into values, `commands.py` runs them |
| `training/config.py` | The config schema, its defaults, and validation |
| `training/train.py` | The assembly: resolve a run, build it, drive its epochs |
| `training/{setup,plan,batches,loop,artifacts,reporting}.py` | The decisions before the first batch, the line announcing them, the batches fixed once, a batch and an epoch, the grids and checkpoints written, and how the numbers get out |
| `training/{ema,lr,validation,checkpoints,distributed,observer}.py` | Weight averaging, the LR ramp, held-out scoring, save/resume, DDP, and the seam the TUI reports through |
| `diffusion/schedules.py` | Beta schedules and every coefficient derived from them |
| `diffusion/ddpm.py` | The baseline process: predict epsilon, fixed variance, MSE |
| `diffusion/gaussian_diffusion.py` | The generalised process, and the variational bound |
| `diffusion/parameterization.py` | The four choices it takes: mean type, variance type, objective, loss weighting |
| `diffusion/{ddim,dpm_solver,heun,plms,samplers}.py` | The reverse chains, and the registry that names them |
| `diffusion/{prediction,latents,timesteps,losses}.py` | Shared sampler arithmetic, x_T, timestep draws, likelihood terms |
| `diffusion/guidance.py` | Classifier-free guidance, as a wrapper round the network |
| `models/{unet,blocks,embeddings}.py` | The backbone, its ResBlocks and attention, its embeddings |
| `metrics/` | FID, sFID, KID, precision/recall, the Inception Score, Inception features, and the reference cache |
| `sweep.py` | The hyperparameter grid, and the directory per point that keeps its runs apart |
| `data/datasets.py` | The dataset registry — MNIST, Fashion-MNIST, CIFAR-10 |
| `data/folder.py` | `dataset = "folder"`: a directory of your own images, and how it is split |
| `tui/` | The training dashboard |
| `server/` | The JSON sampling API |
| `utils/` | Device selection, precision, seeding, metric logging |

## The forward process

`diffusion/schedules.py` builds the beta schedule (cosine by default) and every
coefficient derived from it; `diffusion/ddpm.py` noises an image to a random
timestep and scores the network's noise estimate.

Which timesteps a step scores at is itself a choice. `diffusion/timesteps.py`
draws uniformly by default, and offers Nichol & Dhariwal's loss-second-moment
resampler for the objectives whose per-timestep terms differ by orders of
magnitude.

## The backbone

`models/unet.py` is the DDPM U-Net: pre-activation ResBlocks with FiLM time
conditioning at every resolution, self-attention at chosen scales, and a
spatial bottleneck.

## Parameterisation

`diffusion/parameterization.py` names the three choices DDPM fixes — what the
network predicts, where the reverse variance comes from, and what is optimised.
`diffusion/gaussian_diffusion.py` takes them as arguments, and adds the
variational bound they are measured against.

| Config | Effect |
| --- | --- |
| `variance` | A learned reverse variance, trained on the hybrid objective — Nichol & Dhariwal's improved DDPM |
| `predict = "v"` with `zero_snr` | The velocity parameterisation, on a schedule that reaches zero signal at `t = T` |
| `loss_weighting = "min_snr"` | Stops the low-noise timesteps dominating the gradient |

Each is documented with its trade-off in
[USAGE.md](usage/configuration.md#choosing-the-parameterisation).

## Conditioning

`models/embeddings.py` adds a class embedding, summed into the timestep
embedding so the label rides the FiLM path the ResBlocks already have.

`diffusion/guidance.py` reserves one embedding row as a null token, trains it by
dropping a fraction of the labels, and extrapolates away from it at sample time
— classifier-free guidance, as a wrapper round the network rather than a change
to any sampler. `guidance_rescale` corrects the scale that extrapolation
inflates (Lin et al. 2023), which is what keeps a high guidance scale from
washing the images out.

## Sampling

`diffusion/ddim.py` runs the reverse chain over a subsequence of timesteps, so
50 steps stand in for 1000; `eta` interpolates between deterministic DDIM and
ancestral DDPM. Three deterministic solvers beat its first-order step, each
paying differently: `diffusion/dpm_solver.py` is DPM-Solver++(2M), second order
from the previous step's evaluation, so 15 to 20 steps for the same cost per
step; `diffusion/heun.py` is second order from a second evaluation per step, so
it is correct from its first step where a multistep method is not;
`diffusion/plms.py` is fourth order from three steps of history, at one
evaluation per step and an order ramp that wants 20 steps to pay for itself.
`--sampler` and the `sampler` config field pick between them, and
`fid --kid` at a fixed number of network evaluations is what settles which
wins on a given model.

`--spacing` decides *which* timesteps that budget visits: `quadratic` packs them
towards `t = 0` where a short chain needs them, and `karras` spaces them evenly
in noise level rather than in index. Neither costs an extra network evaluation,
and `fid --kid` will tell you which wins on your model.

`--precision fp16` runs the network in half precision and NHWC, about 1.5x the
throughput on a card with tensor cores. float32 stays the default, because
precision moves a score and a score is only ever a comparison.

`diffusion/prediction.py` holds the arithmetic both samplers need — the implied
clean image, and the noise consistent with it after clipping — so they cannot
drift apart on the classic subtle bug of clamping `x_0` while keeping the
epsilon that implied the unclamped one.

## Weight averaging

`training/ema.py`. DDPM's published sample quality depends on it, so training
and sampling both draw from the EMA weights.

## Scaling out

`training/distributed.py` makes the run data-parallel when a launcher says so,
and does nothing at all when one does not:

```bash
torchrun --nproc_per_node=4 -m tinydiffusion train --config configs/mnist.toml
```

Each GPU gets a complete copy of the network and a disjoint shard of each
epoch, and the gradients are averaged during the backward pass. Rank 0 alone
writes, and the checkpoints it writes are ordinary ones — a four-GPU run
resumes on one.

## Measurement

`metrics/fid.py` summarises a feature set as two moments, which is all FID needs
and all that fits in constant memory. `metrics/features.py` keeps the vectors
instead, for the two metrics that read pairwise structure:

- `metrics/kid.py` — an unbiased kernel distance that stays comparable across
  sample counts, and reports a spread.
- `metrics/precision_recall.py` — measures each set's manifold against the
  other's, which splits a bad score into its two causes.

Two more read the *other* heads of the same Inception pass rather than a
different network, so they cost the pass nothing extra:
`metrics/inception.py`'s `analyse` returns the pooled features alongside an
unpooled intermediate map — the space sFID is measured in, which sees the
spatial arrangement pooling averages away — and the class probabilities
`metrics/inception_score.py` reads. The Inception Score is the one metric here
that never looks at a real image, which is both its appeal and its limit.

`metrics/cache.py` stores the reference half of a score under `data/fid_cache`.
That half does not depend on the checkpoint, which is what makes sweeping
`--guidance` or `--steps` affordable. sFID's reference half is a different
feature space, so it is a separate entry under the same key.

`evaluation.py` reports the other kind of number: the held-out loss, and with
`--bpd` the full variational bound `gaussian_diffusion.py` defines. The bound is
the one figure comparable across parameterisations and against published
likelihoods, and it costs a network evaluation per timestep per image — so it is
opt-in and scored over a slice.

## Reading order

If you are reading the source for the first time:

1. `training/config.py` — every knob, with the reasoning in its docstrings.
2. `diffusion/ddpm.py` — the process, at its simplest.
3. `models/unet.py` — the network being trained.
4. `training/train.py` — how those three become a run, then `training/loop.py`
   for the batch and the epoch it drives.
5. `diffusion/ddim.py`, then `diffusion/guidance.py` — how a run becomes images.
