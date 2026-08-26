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
| `cli.py` | Every subcommand's arguments and dispatch |
| `training/config.py` | The config schema, its defaults, and validation |
| `training/train.py` | The loop: epochs, AMP, checkpoints, per-epoch grids |
| `training/{ema,lr,validation,checkpoints,distributed,observer}.py` | Weight averaging, the LR ramp, held-out scoring, save/resume, DDP, and the seam the TUI reports through |
| `diffusion/schedules.py` | Beta schedules and every coefficient derived from them |
| `diffusion/ddpm.py` | The baseline process: predict epsilon, fixed variance, MSE |
| `diffusion/gaussian_diffusion.py` | The generalised process, and the variational bound |
| `diffusion/{ddim,dpm_solver,samplers}.py` | The reverse chains, and the registry that names them |
| `diffusion/{prediction,latents,timesteps,losses}.py` | Shared sampler arithmetic, x_T, timestep draws, likelihood terms |
| `diffusion/guidance.py` | Classifier-free guidance, as a wrapper round the network |
| `models/{unet,blocks,embeddings}.py` | The backbone, its ResBlocks and attention, its embeddings |
| `metrics/` | FID, KID, precision/recall, Inception features, and the reference cache |
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

`diffusion/gaussian_diffusion.py` makes the three choices DDPM fixes — what the
network predicts, where the reverse variance comes from, and what is optimised
— explicit, and adds the variational bound they are measured against.

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
ancestral DDPM. `diffusion/dpm_solver.py` is DPM-Solver++(2M), which reaches the
same place in 15 to 20 steps for the same cost per step. `--sampler` and the
`sampler` config field pick between them.

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

`metrics/cache.py` stores the reference half of a score under `data/fid_cache`.
That half does not depend on the checkpoint, which is what makes sweeping
`--guidance` or `--steps` affordable.

## Reading order

If you are reading the source for the first time:

1. `training/config.py` — every knob, with the reasoning in its docstrings.
2. `diffusion/ddpm.py` — the process, at its simplest.
3. `models/unet.py` — the network being trained.
4. `training/train.py` — how those three become a run.
5. `diffusion/ddim.py`, then `diffusion/guidance.py` — how a run becomes images.
