# Usage

Everything needed to install, run, and troubleshoot TinyDiffusion, one page per
topic. For what the project *is*, see [README.md](README.md); for how it is
built, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); for contributing, see
[CONTRIBUTING.md](CONTRIBUTING.md).

| Page | What it covers |
| --- | --- |
| [Installing and updating](docs/usage/install.md) | `uv sync`, confirming the GPU is really being used, what lands on disk, uv's cache, and the uv command reference |
| [Running the CLI](docs/usage/cli.md) | The wrapper scripts, and why `python src/tinydiffusion/cli/commands.py` does not work |
| [Training and checkpoints](docs/usage/training.md) | Starting a run, the terminal dashboard, `--set` overrides, hyperparameter sweeps, stopping early, and resuming |
| [Metrics and logging](docs/usage/metrics.md) | `metrics.jsonl`, the console, TensorBoard and Weights & Biases backends, and plotting a run |
| [Sampling](docs/usage/sampling.md) | Samplers, step counts and spacing, half precision, latent walks, and asking for a particular digit |
| [Evaluating a checkpoint](docs/usage/evaluation.md) | Held-out loss, bits per dimension, and FID, sFID, KID, precision and recall, and the Inception Score |
| [Serving a checkpoint over HTTP](docs/usage/serving.md) | The JSON API over one loaded checkpoint |
| [Configuration](docs/usage/configuration.md) | Every config field, going faster, multi-GPU, datasets, parameterisation, and class conditioning |
| [Troubleshooting](docs/usage/troubleshooting.md) | The failures that come up most, and what each one means |

## The short version

```bash
uv sync --all-extras --dev                                    # once
./scripts/run.sh train  --config configs/smoke.toml           # ~28 s/epoch on an RTX 5060
./scripts/run.sh sample --checkpoint runs/smoke/checkpoints/last.pt --out runs/smoke/gen.png
```

On Windows use `.\scripts\run.ps1` in place of `./scripts/run.sh`. Every command
takes `--help`, and any config field can be overridden without editing a file:

```bash
./scripts/run.sh train --config configs/mnist.toml --set lr=1e-4 --set batch_size=64
```
