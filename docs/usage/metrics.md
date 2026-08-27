# Metrics and logging

What a run records, where it goes, and how to draw it.

Part of [Usage](../../USAGE.md).

## Metrics and logging

Every epoch, training prints a table of what it measured and appends the same
numbers to `log_dir/metrics.jsonl`:

```text
------------------------------------
| step 0                           |
------------------------------------
| time/epoch_seconds     |  27.9143 |
| time/images_per_second | 2148.31  |
| train/amp_scale        | 65536.00 |
| train/ema_decay        |   0.9950 |
| train/grad_norm        |   0.4127 |
| train/loss             |   0.0421 |
| train/loss_q0          |   0.1183 |
| train/loss_q1          |   0.0312 |
| train/loss_q2          |   0.0208 |
| train/loss_q3          |   0.0179 |
| train/lr               |  2.0e-04 |
| train/skipped_step     |   0.0000 |
| val/best_loss          |   0.0388 |
| val/loss               |   0.0388 |
------------------------------------
```

Per-batch values (`train/loss`, `train/grad_norm`, the quartiles) are averaged
over the epoch; states (`train/lr`, `train/amp_scale`, the timings) are recorded
as they stood at the end of it.

`val/loss` is the held-out score described under
[Validation and `best.pt`](configuration.md#validation-and-bestpt) — the one number here that is
comparable between epochs, and between runs sharing a parameterisation.
`val/best_loss` is the lowest it has reached, which is the epoch `best.pt`
holds.

The quartile losses are the reason to look at this rather than just the progress
bar. `loss_q0` covers the lowest quarter of the diffusion schedule and `loss_q3`
the highest, so they say *where* the model is struggling: high-`t` error means
the denoiser cannot recover structure from near-pure noise, low-`t` error means
it cannot clean up the last little bit. The mean of the two moves for neither
reason. They are sampled from one batch in eight — slicing by timestep costs a
device sync, and every batch draws its timesteps independently, so the epoch
mean is the same either way. `train/skipped_step` is the fraction of batches AMP
threw away for inf/NaN gradients — a number that stays above zero is a reason to
lower `lr`.

`metrics.jsonl` is one JSON object per epoch, so a finished run is comparable
against the next one without re-reading a log. Every record carries three
reserved keys alongside the metrics: `step`, the wall-clock `time`, and
`session`, which counts how many times the file has been appended to. A metric
of the same name does not overwrite them.

`session` is what makes a resumed run readable. Resuming from epoch 5 appends a
second copy of every epoch from 5 on, so the raw file holds more lines than the
run has epochs, and reading it straight through shows the loss doubling back on
itself. `read_metrics` resolves that, keeping the newest session per step:

```bash
python -c "from pathlib import Path;from tinydiffusion.utils import read_metrics;[print(r['step'], r['train/loss']) for r in read_metrics(Path('runs/mnist/metrics.jsonl'))]"
```

A metric that went to NaN or an infinity — a diverged run, most often — is
stored as `null`. The bare `NaN` token Python's `json` would otherwise write is
not JSON, and `jq` and `pandas.read_json` reject a file containing one outright.

### Plotting a run

`plot` turns those records into a figure, which is the shape the questions
actually have — whether the loss is still falling, whether the held-out score
has turned back up, and which quarter of the schedule the error sits in. It
needs the `plots` extra (`uv sync --all-extras`, or
`pip install 'tinydiffusion[plots]'`):

```bash
./scripts/run.sh plot runs/mnist --out contents/metrics.png
```

Panels are chosen from what the run logged, so an unconditional run with no
validation split gets no empty `val/loss` axis. Pass more than one run and they
share every axis, one line per run, which is how a sweep is read:

```bash
./scripts/run.sh plot runs/baseline runs/min_snr --out contents/compare.png
```

Each run is labelled by its directory name, which is what makes this the other
half of [`sweep`](training.md#sweeping-a-grid): a sweep names every point's
directory after the values that distinguish it, so the whole root plots as a
legend that reads itself.

```bash
./scripts/run.sh plot runs/sweep/* --out contents/sweep.png
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `RUN...` | required | Run log directories, or `metrics.jsonl` files |
| `--out` | `contents/metrics.png` | Image to write; the extension picks the format, so `.svg` works |
| `--dpi` | 120 | Resolution for raster formats |

### TensorBoard and Weights & Biases

Two further sinks, both optional and both off by default. They need the
`tracking` extra (`uv sync --all-extras`, or
`pip install 'tinydiffusion[tracking]'`). Neither replaces `metrics.jsonl`,
which is written either way and stays what `plot` reads.

TensorBoard writes to `log_dir/tb`:

```bash
./scripts/run.sh train --config configs/mnist.toml --tensorboard
tensorboard --logdir runs/mnist/tb
```

Weights & Biases is the one that leaves the machine, which is the whole reason
to want it: a run on a remote box is watchable from a laptop, and several runs
land on shared axes without anyone copying files around.

```bash
wandb login                       # or export WANDB_API_KEY=...
./scripts/run.sh train --config configs/mnist.toml --wandb
```

The run is named after `log_dir`, so the W&B dashboard lines up with the
checkpoints on disk, and the training config is sent once at the start so the
sweep view can group and filter by hyperparameter. Nothing else goes: no
images, no checkpoints, no dataset. `WANDB_MODE=offline` records locally to
sync later, which is what a training box with no outbound network wants.

A network that drops mid-run costs the run nothing — a failed send warns and
training continues, since `metrics.jsonl` already holds everything W&B was
being told.

| Flag | Config field | Meaning |
| --- | --- | --- |
| `--log-dir` | `log_dir` | Where `metrics.jsonl` and `tb/` are written |
| `--tensorboard` | `tensorboard` | Also write TensorBoard events |
| `--wandb` | `wandb` | Also stream to Weights & Biases |
| `--wandb-project` | `wandb_project` | W&B project to log into (default `tinydiffusion`) |
| `--quiet` | `log_console` | Suppress the per-epoch table |

An unpassed flag leaves the config file's value alone, so `--tensorboard` or
`--wandb` can be turned on for one run without editing the TOML.

Under `torchrun`, only rank 0 logs. The metrics are already the group's — the
losses are all-reduced before they reach the logger — so a second rank would
add no information and one more W&B run per GPU.
