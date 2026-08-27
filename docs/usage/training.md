# Training and checkpoints

Starting a run, watching it, stopping it, and picking up where it left off.

Part of [Usage](../../USAGE.md).

- [Training](#training)
- [Checkpoints](#checkpoints)

## Training

Start with the smoke config. It is the same pipeline shrunk to finish an epoch
in well under a minute on a GPU — the point is to prove the wiring works, not
to get good digits:

```bash
./scripts/run.sh train --config configs/smoke.toml
```

Then the real run:

```bash
./scripts/run.sh train --config configs/mnist.toml
```

Each epoch writes a `sample_XXXX.png` grid to `out_dir` — generated digits
above a strip of real ones, so contrast and stroke weight are directly
comparable — and a resumable `last.pt` to `ckpt_dir`. A conditional run
generates on the real strip's own labels, so the comparison is per class: a
generated 4 sits directly above a real 4. Both the strip and the starting
noise are fixed for the whole run, including across a `--resume`, so the PNGs
read as one set of digits sharpening rather than a fresh draw each epoch. The
checkpoint holds the
model, EMA shadow weights, optimiser moments, AMP scaler state, and the config,
so a resumed run continues rather than restarts:

```bash
./scripts/run.sh train --config configs/mnist.toml --resume checkpoints/last.pt
```

Flags override the config file when passed: `--seed`, `--device`, `--epochs`,
and the logging flags in [Metrics and logging](metrics.md#metrics-and-logging).
`--config` itself is optional — omit it to run the built-in defaults, or the
settings stored in the checkpoint when [resuming](#resuming).

```bash
./scripts/run.sh train --config configs/mnist.toml --device cpu --epochs 1 --seed 7
```

### The dashboard

`tui` trains inside a terminal dashboard rather than behind a progress bar:

```bash
./scripts/run.sh tui --config configs/mnist.toml          # then press s
./scripts/run.sh tui --config configs/mnist.toml --start  # or start straight away
```

It needs the `tui` extra (`pip install 'tinydiffusion[tui]'`), which pulls in
[Textual](https://textual.textualize.io/). Without it the command says so in
one line, the way `plot` and `serve` do for theirs.

It takes the same settings `train` does — `--config`, `--resume`, `--dataset`,
`--set`, `--seed`, `--device`, `--epochs`, `--log-dir` — and resolves them the
same way, so a run started here is the run `train` would have started.

| Key | What it does |
| --- | --- |
| `s` | Start training |
| `x` | Stop at the next batch boundary, writing `interrupted.pt` first |
| `r` | Restart: stop, then start again once the worker has left |
| `l` | Show or hide the log pane |
| `c` | Clear the log |
| `f` | Focus mode — hide the sidebar, the tiles and the log, and give the room to the charts |
| `t` | Theme picker, applying each theme as the highlight moves |
| `d` / `D` | Cycle to the next or previous theme |
| `ctrl+s` | Save an SVG screenshot into the run's log directory |
| `ctrl+p` | Command palette — every action above, by name |
| `?` | List every key |
| `q` | Quit |

What it shows, live:

- a status line: whether the run is ready, training, stopping, finished or
  failed, what it is training, and how far through the current epoch it is;
- five headline tiles — smoothed training loss, validation loss, throughput,
  ETA and elapsed — big enough to read from across the room;
- the resolved plan — parameter count, device and GPU model, precision,
  conditioning, steps per epoch — as data rather than as the one-line sentence
  the plain trainer prints;
- epoch and batch progress, with throughput, epoch time and an ETA for the
  whole run;
- `train/loss` and `val/loss` per epoch, drawn as braille lines on one shared
  axis with the values labelled. Braille puts a 2x4 grid of dots in every
  character cell, so an eight-row chart is thirty-two pixels tall — enough to
  see a validation curve stop following the training curve down, which is the
  single most useful thing a training dashboard can show and the thing two
  separately-normalised sparklines cannot;
- the loss split by timestep quartile, as four bars on one scale — which
  quarter of the schedule the error is sitting in is the thing a single loss
  number cannot tell you;
- the latest sample grid, drawn in the terminal. A cell is about twice as tall
  as it is wide, so each one carries two pixels as an upper half block, and a
  32px MNIST grid is more than legible enough to watch a 7 become a 7.

The layout follows the terminal. Below about 100 columns the tiles drop away —
the sidebar's stats panel is already carrying every number on them — and below
about 72 the sidebar goes too, leaving the charts and the samples the width.

#### Themes

Thirty-odd of them: ten written for this dashboard (`tinydiffusion` and
`tinydiffusion-light`, `latent`, `ember`, `mint`, `oceanic`, `synthwave`,
`noir`, `paper`, `arctic`) alongside every theme Textual ships — Nord, Gruvbox,
Dracula, Tokyo Night, Catppuccin, Monokai, Flexoki, Solarized, Rosé Pine and
the rest. `t` opens the picker, which applies each one as the highlight moves
so the choice is made by eye; `d` and `D` walk the same list a keypress at a
time. The colours reach the charts, the bars and the tiles, not only the
borders.

Whatever is chosen is remembered in `~/.config/tinydiffusion/tui.json` and used
the next time the dashboard opens. `TINYDIFFUSION_CONFIG_DIR` moves that file,
`XDG_CONFIG_HOME` moves the directory it sits under, and a missing or
unreadable one simply means the default.

Stopping with `x` is the Ctrl+C path without the prompt: it stops at a batch
boundary, where the model, optimiser and EMA all agree, and writes
`interrupted.pt` so `train --resume` or `tui --resume` picks the run up. The
run's `metrics.jsonl` is written exactly as it always is, so `plot` works on a
run trained here; only the per-epoch console table is turned off, since stdout
belongs to the display.

Training runs on a worker thread and the display on the event loop, which is
why a slow terminal cannot stall the run: batch updates are throttled, and the
loop never waits longer than a frame on the screen.

### Overriding any config field

`--set field=value` reaches every field in
[the config reference](configuration.md#configuration) without editing a file, which is what
makes a sweep a shell loop rather than a directory of near-identical TOMLs.
It is repeatable:

```bash
./scripts/run.sh train --config configs/mnist.toml --set lr=1e-4 --set batch_size=64
```

```bash
for lr in 1e-4 2e-4 4e-4; do
  ./scripts/run.sh train --config configs/mnist.toml     --set lr=$lr --set log_dir=runs/lr-$lr --set ckpt_dir=checkpoints/lr-$lr
done
```

That loop is what [`sweep`](#sweeping-a-grid) does with the bookkeeping already
done, and the three `--set`s of directories are the bookkeeping.

Values are read exactly as the config file would read them, so the types come
out right without you saying which is which: `batch_size=64` is an integer,
`lr=1e-4` a float, `amp=false` a boolean, `channel_mult=[1,2,2]` a list.
Anything TOML cannot parse as a value is taken as a plain string, which is what
lets `dataset=cifar10` and `out_dir=runs/sweep` go unquoted.

A field name that does not exist is an error rather than a silent no-op, and
the value is validated exactly as the file's would be — `--set batch_size=0` is
refused before the dataset is touched. `--set` is applied last, so it also wins
over `--epochs` and the other named flags:

```console
$ ./scripts/run.sh train --set batch_sizes=64
error: unknown config field(s): batch_sizes
```

### Sweeping a grid

The shell loop above works, and the moment you forget one of the directory
overrides every point writes over the last one's record of itself — which is
the comparison the sweep was for. `sweep` is that loop with the directories
handled:

```bash
./scripts/run.sh sweep --config configs/mnist.toml   --axis lr=1e-4,2e-4,4e-4 --axis sample_spacing=uniform,quadratic
```

Every combination is run — two axes of three and two values are six points, and
six training runs — and each gets its own directory under `--out-root`, named
after the values that distinguish it:

```text
runs/sweep/lr=0.0001_sample_spacing=uniform/
runs/sweep/lr=0.0001_sample_spacing=quadratic/
...
```

`log_dir`, `ckpt_dir` and `out_dir` are set per point, so metrics, checkpoints
and sample grids all land inside it. Which means the whole root plots as one
figure with a legend that reads itself:

```bash
./scripts/run.sh plot runs/sweep/* --out contents/sweep.png
```

At the end it prints what each point reached, ranked:

```text
point                                  epochs   best val/loss
lr=0.0004_sample_spacing=quadratic         30         0.03812
lr=0.0002_sample_spacing=quadratic         30         0.03904
lr=0.0001_sample_spacing=uniform           30         0.04120
```

Values on an axis are read exactly as `--set` reads one, so the same types come
out. `--set` still works alongside, and is where the settings the sweep holds
*fixed* go — they are the same for every point, so they stay out of the
directory names:

```bash
./scripts/run.sh sweep --config configs/mnist.toml   --axis lr=1e-4,4e-4 --set num_epochs=10 --set amp_dtype=bf16
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` | the defaults | Config every point starts from |
| `--axis` | required | `FIELD=A,B,C`, repeatable; every combination is run |
| `--set` | — | Override a field for every point, repeatable |
| `--out-root` | `runs/sweep` | Directory the points are created under |
| `--dry-run` | off | Print the grid without training anything |
| `--skip-existing` | off | Leave a point alone if its directory already holds metrics |
| `--seed` | the config's | Random seed |
| `--device` | the config's | `cuda`, `cpu`, … |
| `--epochs` | the config's | Epochs per point |

Three things worth knowing:

- **`--dry-run` first.** It prints the grid and stops, which is the cheap way to
  find out that a sweep is larger than it looked. Points multiply.
- **One bad point does not stop the rest.** A combination that runs out of
  memory is a fact about that point, and losing five good runs to it would be
  the wrong trade; the failure is reported in the summary and the sweep exits
  non-zero. `Ctrl+C` is different — that ends the sweep, not the point.
- **`--skip-existing` resumes it.** A point whose directory already holds
  `metrics.jsonl` is left alone and its existing numbers read, so an interrupted
  sweep continues rather than restarting. Axes over `log_dir`, `ckpt_dir` or
  `out_dir` are refused: those are the thing the sweep is setting.

### Stopping a run early

`Ctrl+C` does not kill the run outright. At the next batch boundary training
pauses and asks whether to stop, and if so whether to checkpoint first:

```text
stop training? [y/N] y
save a checkpoint so training can resume later? [Y/n] y
saved checkpoints/interrupted.pt (3 epochs complete, plus a partial epoch)
resume with: tinydiffusion train --resume checkpoints/interrupted.pt
```

Answering `n` to the first question resumes training where it left off. A
second `Ctrl+C` while the questions are on screen quits immediately, and a run
without a terminal attached — a CI job, a `nohup`ed script — saves and exits
without asking.

The save goes to `interrupted.pt`, never over `last.pt`. An interrupt lands
mid-epoch, so its weights are a few batches *worse* than the ones the previous
epoch finished on — and because a mid-epoch checkpoint records the last
**completed** epoch (so resuming replays the interrupted one in full), writing
it to `last.pt` would replace a good checkpoint with a worse one carrying the
same epoch number. Keeping them apart means `last.pt` is always the newest
complete epoch and `interrupted.pt` is always the furthest the run actually
got.

## Checkpoints

A run writes up to three kinds of file into `ckpt_dir`:

| File | Written | Holds |
| --- | --- | --- |
| `last.pt` | after every completed epoch | the newest complete epoch |
| `best.pt` | when held-out loss improves | the best epoch so far, by `val/loss` |
| `interrupted.pt` | on a saved `Ctrl+C` | the furthest the run got |
| `epoch_NNNN.pt` | when `keep_last > 0` | the last `keep_last` epochs |

Each is a full training state — weights, EMA shadow weights, optimiser moments,
AMP scale, and the config it was trained with — so any of them can be passed to
`--resume`, `sample`, `eval`, `fid` or `serve`.

`best.pt` is usually the one to sample from. Diffusion runs do not improve
monotonically, and the last epoch is not reliably the best one; without a
recorded score there is no way to tell which epoch was, and `last.pt` has
already overwritten the evidence. `keep_last` is the cruder insurance: set it
to keep a rolling window of numbered epoch snapshots.

```toml
[bookkeeping]
keep_best = true   # maintain best.pt (the default)
keep_last = 3      # also keep epoch_0028.pt, epoch_0029.pt, epoch_0030.pt
```

### Resuming

`--resume` on its own continues the run the checkpoint came from: the settings
it was trained with travel inside it, so the TOML file it started from is not
needed a second time.

```bash
./scripts/run.sh train --resume checkpoints/last.pt
```

Pass `--config` as well to resume into different settings, and the usual flags
still override whichever of the two was used:

```bash
./scripts/run.sh train --resume checkpoints/last.pt --epochs 60
```

`--resume` loads weights into the model the config describes, so the two have
to agree. They are checked before anything is loaded, and a mismatch names the
setting that changed:

```text
checkpoints/last.pt was trained with a different model, so it cannot resume
into this config:
  base_channels: checkpoint 64, config 128
match the config to the checkpoint, or start a fresh run without --resume
```

Everything the weights depend on is compared — the six architecture fields, the
schedule, and the three parameterisation fields. Settings that do not change
the weights, like `batch_size`, `lr` or `num_epochs`, are yours to change
between resumes.

A checkpoint written before the config travelled with it has nothing to rebuild
from, so a bare `--resume` on one asks for the config file instead.
