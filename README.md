# TinyDiffusion

[![CI](https://github.com/HilthonTT/TinyDiffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/HilthonTT/TinyDiffusion/actions/workflows/ci.yml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of a diffusion model for image generation: a DDPM
U-Net trained on MNIST, sampled with DDIM, and conditioned on the digit class
with classifier-free guidance — so you can ask it for a 7.

![Generated digits after 10 epochs](docs/sample_0010.png)

_`docs/sample_0010.png`, written automatically after the sixth epoch. The
top half is generated, the bottom half is real MNIST — every grid pairs them so
stroke weight and contrast are directly comparable. Six epochs is about 11
minutes on an RTX 5060; most digits are already well formed, a few strokes are
still breaking up._

> **Status:** early development. Training and sampling work end to end.
> MNIST is the default and the best-exercised path; Fashion-MNIST and
> CIFAR-10 are wired up through the same registry (`dataset = "cifar10"`,
> and see `configs/cifar10.toml`) but have had far less mileage.

## Quickstart

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/). Full instructions,
including how to check the GPU was picked up, are in
[docs/INSTALL.md](docs/INSTALL.md).

```bash
uv sync --all-extras --dev              # creates .venv/ (~3.5 GB with CUDA)
./scripts/run.sh train  --config configs/smoke.toml
./scripts/run.sh sample --checkpoint runs/smoke/checkpoints/last.pt --out runs/smoke/gen.png
```

On Windows use `.\scripts\run.ps1` in place of `./scripts/run.sh`. The wrappers find an
interpreter that has the package installed and forward the rest to the CLI —
`python src/tinydiffusion/cli.py` does not work in a `src/` layout.

`configs/smoke.toml` is a deliberately tiny run for checking the pipeline
end to end (~28 s per epoch on an RTX 5060, ~3.5 min on a CPU). The real run is
`configs/mnist.toml`:

```bash
./scripts/run.sh train --config configs/mnist.toml
```

Any config field can be overridden from the command line with `--set`, which is
what turns a sweep into a shell loop:

```bash
./scripts/run.sh train --config configs/mnist.toml --set lr=1e-4 --set batch_size=64
```

MNIST (63 MB) downloads itself on first use. Each epoch writes a sample grid —
generated digits above real ones — and a resumable checkpoint. Afterwards,
`sample` generates images from any checkpoint and `eval` scores it on the
held-out test split:

```bash
./scripts/run.sh sample --checkpoint checkpoints/last.pt --num-images 16
./scripts/run.sh eval   --checkpoint checkpoints/last.pt
```

`fid` goes further and measures sample _quality_, by comparing generated images
with real ones in Inception-v3 feature space:

```bash
./scripts/run.sh fid --checkpoint checkpoints/last.pt --num-images 10000
```

The real images' half of that score does not depend on the checkpoint, so it is
computed once and cached under `data/fid_cache` — which is what makes sweeping
`--guidance` or `--steps` affordable.

FID needs those 10,000 images to mean anything: it fits a 2048-dimensional
Gaussian, and a smaller sample biases the score upwards by an amount that
depends on the sample count. Two more metrics are available for when that is
the wrong shape of answer:

```bash
./scripts/run.sh fid --checkpoint checkpoints/last.pt --num-images 2000 --kid --precision-recall
```

`--kid` is unbiased, so a score over 2,000 images means the same thing as one
over 50,000, and it reports a spread — which says whether two checkpoints have
actually been told apart. `--precision-recall` splits a bad score into its two
causes: how much of what the model draws is realistic, and how much of the real
data it reaches. Guidance trades one for the other, and no single number can
show that.

`tui` trains inside a terminal dashboard instead: live loss, progress and ETA,
the loss split by timestep quartile, and each epoch's sample grid drawn in the
terminal as you go. `s` starts, `x` stops at a batch boundary and checkpoints,
`q` quits.

```bash
./scripts/run.sh tui --config configs/mnist.toml --start   # needs the 'tui' extra
```

`plot` draws a run's `metrics.jsonl` — losses, the timestep quartiles, the
learning rate — as a figure, and several runs on shared axes:

```bash
./scripts/run.sh plot runs/mnist --out contents/metrics.png
```

`interpolate` walks between two latents and samples every point on the way,
which says something a grid of samples cannot: whether the space *between* two
images is populated, or whether the model snaps from one mode to another with
nothing in between.

```bash
./scripts/run.sh interpolate --checkpoint checkpoints/last.pt --labels 7 --steps 10
```

`serve` puts a checkpoint behind a JSON API, for anything that is not a shell:

```bash
./scripts/run.sh serve --checkpoint checkpoints/last.pt   # needs the 'server' extra
curl -X POST localhost:8000/api/sample -H 'content-type: application/json' \
  -d '{"num_images": 8, "labels": [7]}'
```

`configs/mnist.toml` trains conditionally on the ten digits, so `sample` can be
asked for a particular one — and for how hard to insist on it:

```bash
./scripts/run.sh sample --checkpoint checkpoints/last.pt --labels 7 --num-images 8
./scripts/run.sh sample --checkpoint checkpoints/last.pt --labels 0,1,2 --guidance 4
./scripts/run.sh sample --checkpoint checkpoints/last.pt --guidance 6 --guidance-rescale 0.7
```

With no `--labels` the grid holds one image per digit. `--guidance` is the
classifier-free guidance scale: 1.0 is the plain conditional prediction, and
higher values trade variety for cleaner, more emphatically class-typical
digits at the cost of a second forward pass per step.

**Using a GPU:** `uv sync` installs a CUDA build of PyTorch on Windows and
Linux, so an NVIDIA GPU is picked up automatically, and training falls back to
the CPU when none is visible — its first line says which it chose. Add
`--no-sources` for a CPU-only environment. For scale, one MNIST epoch is about
1.8 minutes on an RTX 5060 and 29 on a CPU. See
**[docs/INSTALL.md](docs/INSTALL.md)** for verification and troubleshooting,
and **[USAGE.md](USAGE.md)** for configuration and every CLI flag.

## Documentation

|                                    |                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------- |
| [docs/INSTALL.md](docs/INSTALL.md) | Install, GPU setup, verification, troubleshooting                                        |
| [USAGE.md](USAGE.md)               | Downloads and disk use, training, sampling, evaluation, every CLI flag, config reference |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow                                                                     |
| [RESULTS.md](RESULTS.md)           | Measured scores, with the commands and hardware that produced them                       |
| [CHANGELOG.md](CHANGELOG.md)       | Release history                                                                          |

## How it works

- **Forward process** — `diffusion/schedules.py` builds the beta schedule
  (cosine by default) and every coefficient derived from it; `diffusion/ddpm.py`
  noises an image to a random timestep and scores the network's noise estimate.
- **Backbone** — `models/unet.py` is the DDPM U-Net: pre-activation ResBlocks
  with FiLM time conditioning at every resolution, self-attention at chosen
  scales, and a spatial bottleneck.
- **Parameterisation** — `diffusion/gaussian_diffusion.py` makes the three
  choices DDPM fixes — what the network predicts, where the reverse variance
  comes from, and what is optimised — explicit, and adds the variational bound
  they are measured against. Set `variance` and `objective` in the config to
  train Nichol & Dhariwal's improved DDPM with a learned variance, `predict =
"v"` with `zero_snr` for the velocity parameterisation on a schedule that
  reaches zero signal at `t = T`, and `loss_weighting = "min_snr"` to stop the
  low-noise timesteps dominating the gradient.
- **Conditioning** — `models/embeddings.py` adds a class embedding, summed into
  the timestep embedding so the label rides the FiLM path the ResBlocks already
  have. `diffusion/guidance.py` reserves one embedding row as a null token,
  trains it by dropping a fraction of the labels, and extrapolates away from it
  at sample time — classifier-free guidance, as a wrapper round the network
  rather than a change to any sampler. `guidance_rescale` corrects the scale
  that extrapolation inflates (Lin et al. 2023), which is what keeps a high
  guidance scale from washing the images out.
- **Sampling** — `diffusion/ddim.py` runs the reverse chain over a subsequence
  of timesteps, so 50 steps stand in for 1000. `eta` interpolates between
  deterministic DDIM and ancestral DDPM. `diffusion/dpm_solver.py` is
  DPM-Solver++(2M), which reaches the same place in 15 to 20 steps for the same
  cost per step; `--sampler` and the `sampler` config field pick between them.
  `--spacing` decides *which* timesteps that budget visits: `quadratic` packs
  them towards `t = 0` where a short chain needs them, and `karras` spaces them
  evenly in noise level rather than in index. Neither costs an extra network
  evaluation, and `fid --kid` will tell you which wins on your model.
  `--precision fp16` runs the network in half precision and NHWC, which is
  about 1.5x the throughput on a card with tensor cores; float32 stays the
  default, because precision moves a score and a score is only ever a
  comparison.
- **Weight averaging** — `training/ema.py`. DDPM's published sample quality
  depends on it, so training and sampling both draw from the EMA weights.
- **Scaling out** — `training/distributed.py` makes the run data-parallel when
  a launcher says so, and does nothing at all when one does not:
  `torchrun --nproc_per_node=4 -m tinydiffusion train ...` gives each GPU a
  complete copy of the network and a disjoint shard of each epoch, averaging
  the gradients during the backward pass. Rank 0 alone writes, and the
  checkpoints it writes are ordinary ones — a four-GPU run resumes on one.
- **Measurement** — `metrics/fid.py` summarises a feature set as two moments,
  which is all FID needs and all that fits in constant memory.
  `metrics/features.py` keeps the vectors instead, for the two metrics that
  read pairwise structure: `metrics/kid.py`, an unbiased kernel distance that
  stays comparable across sample counts, and `metrics/precision_recall.py`,
  which measures each set's manifold against the other's.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow. Quick version:

```bash
uv run pre-commit install
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## License

[MIT](LICENSE)
