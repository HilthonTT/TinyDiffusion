# Configuration

Every config field, and the settings worth reaching for when a run is too
slow, too big for one GPU, or not learning what you wanted.

Part of [Usage](../../USAGE.md).

## Configuration

Configs are TOML. Tables are cosmetic grouping only: every key is flattened
into the flat `TrainConfig` namespace, so a key must name a real field and may
appear in exactly one table. Unknown or repeated keys are errors, not silent
no-ops — a typo fails immediately instead of wasting a training run.

```toml
[model]
base_channels = 64
channel_mult = [1, 2, 2]
attn_resolutions = [16]

[diffusion]
num_timesteps = 1000
schedule = "cosine"       # or "linear", which uses beta_start/beta_end
```

### Going faster

Everything here is off or neutral by default, because each one is a change you
should measure on your own hardware rather than inherit:

```toml
[optimisation]
grad_accum = 2          # effective batch of batch_size * grad_accum
lr_schedule = "cosine"  # decay to zero over the run, after the warmup ramp

[bookkeeping]
amp_dtype = "bf16"      # no gradient scaler, no skipped steps; Ampere or newer
compile = true          # torch.compile the training step
channels_last = true    # measure this one; it can lose on a small model
```

`bf16` wants bfloat16 *hardware*, which means Ampere or newer. Older cards can
emulate it, and emulation is far slower than the fp16 it would be replacing —
on a Turing card, roughly 1210ms/step against fp16's 258ms — so a run on one
says so at startup and uses fp16 instead. There is nothing to configure: the
fallback is automatic, and the startup line reports the dtype actually used.

`full_fp16` is the other half-precision strategy, and an alternative to
autocast rather than an addition to it. Autocast leaves the weights in float32
and casts each operand on the way into a kernel; `full_fp16` puts float16 in
the convolutions themselves and hands the optimiser a flattened float32 copy —
the master parameters — to step instead. The convolutions then run end to end
with no per-operation casts, and the weights are stored once in half rather
than once in full.

```toml
[bookkeeping]
full_fp16 = true        # float16 weights, float32 master copy; needs CUDA
```

The norms, the timestep and label embeddings, the FiLM projections and the
output head all stay in float32 — they are a rounding error in both parameter
count and FLOPs, and they are exactly where half precision costs accuracy. The
gradient scaler is not optional here and is always on: with no float32 weights
to fall back on, an unscaled backward pass flushes a good share of a diffusion
model's gradients to float16's floor.

Everything downstream is unaffected. The float32 master copy is what gets
written, so a `full_fp16` run produces ordinary float32 checkpoints, the EMA is
averaged in float32, and `train` hands back a normal float32 model. The one
thing that does not carry across is AdamW's moments: they are stored per
parameter tensor, and this mode gives the optimiser one flat tensor where every
other mode gives it a few hundred. Resuming across the setting keeps the
weights, the EMA and the schedule and says on startup that it is starting the
moments fresh.

Whether it is faster than autocast depends on the card and the width — measure
it rather than assume it. `amp = true` and `amp_dtype = "fp16"` are its
prerequisites, and both are the defaults; anything else is refused rather than
ignored, and CUDA is required, with a float32 fallback and a printed line where
there is no GPU.

When the limit is memory rather than speed — a wider `base_channels` or a
larger `batch_size` than the card will hold — turn on `grad_checkpoint`. It
drops the U-Net's intermediate activations and recomputes them during the
backward pass, costing roughly a third more time per step and buying back most
of the activation memory. Nothing else changes: same weights, same loss, and a
checkpoint trained with it on resumes fine with it off. Sampling is untouched,
since there is no backward pass there to save for.

Measured on an RTX 5060 at the `configs/mnist.toml` settings (batch 128, 32px,
`base_channels = 64`), 25 steps after a warmup:

| Setting | ms/step | vs default |
| --- | --- | --- |
| fp16 (the default) | 203 | — |
| bf16 | 206 | 0.98x |
| fp16 + `channels_last` | 182 | **1.11x** |
| bf16 + `channels_last` | 183 | 1.11x |

So on that card `channels_last` is worth about 11% and `amp_dtype = "bf16"`
buys no throughput — its argument is stability, not speed: no loss scaler, and
so no skipped steps. Both are worth re-measuring on your own card and at your
own width; these numbers do not transfer.

`compile` wraps the network for the training step only. The checkpoint, the
EMA and every sampler keep the eager module, which shares its parameters — so a
compiled run writes ordinary checkpoints rather than ones whose keys all carry
a `_orig_mod.` prefix, and the first batch pays the compile cost once.

> **On Windows, `compile` needs Triton**, which the PyTorch Windows wheels do
> not ship: `pip install triton-windows`. Without it the run says so on
> startup and trains eagerly, rather than failing on the first batch several
> frames inside dynamo. It is therefore unmeasured here.

`amp_dtype = "bf16"` turns the gradient scaler off, since bfloat16 has
float32's exponent range and nothing to overflow. That also means no skipped
steps: `train/skipped_step` stays at zero, and it is not hiding anything. On a
GPU without bfloat16 the run says so and falls back to fp16.

`grad_accum` buys an effective batch larger than VRAM allows. Each group is
averaged over the batches it actually holds, including the ragged one a
non-dividing loader leaves at the end of an epoch, so the last update of an
epoch is not quietly a fraction of the others. `lr_warmup` and `lr_schedule`
count optimiser steps, so raising `grad_accum` covers proportionally more data
per step of the schedule.

The optimiser is AdamW. At the default `weight_decay = 0.0` that is Adam
exactly — decoupled decay is the only thing the two differ in — so turning it
on is an opt-in, and `betas` is there for the runs that need it.

#### The dataloader is not the bottleneck

Worth recording, because it looks like it should be: every epoch decodes the
split from PIL and resizes it again, and caching the decoded tensors is the
obvious optimisation. Measured on an RTX 2070 over the 60,000-image MNIST
training split at `batch_size = 128`, it is not worth doing.

| | Throughput |
| --- | --- |
| PIL pipeline, `num_workers = 0` | 4,652 img/s |
| PIL pipeline, `num_workers = 4` (the default) | **18,026 img/s** |
| decoded once into uint8 tensors, for comparison | 30,066 img/s |
| — what `configs/mnist.toml` can consume | **471 img/s** |
| — what `configs/smoke.toml` can consume | **2,629 img/s** |

The model is the slow half by a factor of 7 to 38. `configs/mnist.toml` is
271.8 ms per batch, or about 128 s of pure compute per epoch — which is
essentially the whole of the 1.8 minutes an epoch takes, leaving nothing for
the loader to be blamed for. The workers prefetch, so what little the data
costs overlaps the compute rather than adding to it, and even the deliberately
tiny `configs/smoke.toml` model has sevenfold headroom.

Caching the split would therefore buy no wall-clock while costing 59 MB for
MNIST — 153 MB for CIFAR-10 — a one-off decode at startup, and a question
about where the augmentation's random flip is drawn from. So the pipeline
stays as it is.

Two things follow for when you change the shape of a run. Dropping to
`num_workers = 0` gives up the overlap, which is the one setting that can make
the data visible: at 4,652 img/s it still outruns `configs/mnist.toml`
tenfold, but it is no longer free. And the headroom is a property of *this*
model and this resolution — a much smaller network, or a much larger
`batch_size`, moves the model's appetite up towards the loader's ceiling.
`time/images_per_second` in `metrics.jsonl` is the number to watch: compare it
against the table above, and if a run sits near the loader's figure rather than
the model's, the data has become the limit.

### Training on several GPUs

One process per GPU, launched by `torchrun`. Nothing in the config turns this
on: the launcher sets `RANK`, `WORLD_SIZE` and `LOCAL_RANK` in each worker's
environment, and the run reads them. Their absence is what an ordinary
single-process run looks like, so the path everything else in this document
describes is untouched.

```bash
torchrun --nproc_per_node=4 -m tinydiffusion train --config configs/cifar10.toml
```

`--nproc_per_node` is how many GPUs on this machine to use; `-m tinydiffusion`
is the same CLI the `tinydiffusion` command runs, addressed as a module because
that is what a launcher can hand to the processes it starts. Every flag and
`--set` override works exactly as it does on one GPU.

The parallelism is over the batch, not the model. Every rank holds a complete
copy of the network and draws a disjoint shard of each epoch, and the gradients
are averaged across ranks during the backward pass, so all copies step
identically. An epoch is still one pass over the dataset — four ranks each see
a quarter of it — rather than four passes.

**The batch that matters is the effective one.** Each rank contributes its own
`batch_size` to the same averaged gradient, so an optimiser step under
`--nproc_per_node=4` averages over `batch_size * grad_accum * 4` images. The
startup line reports it:

```
2.31M parameters | cifar10 32px x3 | device cuda:0 (...) | amp fp16 | ... | 512 effective | rank 0/4
```

That is a real change to the run, not just to its throughput: an effective
batch four times larger is four times fewer optimiser steps per epoch, and the
usual response is to raise `lr` — the square-root and linear scaling rules are
both defensible, and neither is applied for you. `lr_warmup` and `lr_schedule`
count optimiser steps, so they cover proportionally more data per step too.
Leaving `lr` alone is a valid choice; it is just a different run from the
single-GPU one, and worth knowing you have made.

Everything a run writes is rank 0's: `metrics.jsonl`, the sample grids,
`last.pt` and `best.pt`, the progress bar and every startup line. The weights
are identical on every rank, so there is nothing for the others to add, and
four processes appending to one metrics file would interleave into a file
nothing can parse. The logged loss is all-reduced first, so it is the loss of
the whole global batch rather than of rank 0's shard, and
`time/images_per_second` covers the group — which is the number to compare
against a single-GPU run.

The checkpoints are ordinary ones. The wrapper shares its parameters with the
eager network, exactly as `compile` does, so a four-GPU run's `best.pt` has no
`module.` prefix in its keys and `sample`, `eval`, `fid` and `serve` read it
without knowing how it was trained. A resume works in either direction: train
on four GPUs, resume on one.

Ctrl+C still works, and still asks. The launcher delivers the signal to each
process separately, so the ranks can see it a batch or two apart; the run
agrees on it at a fixed batch cadence, rank 0 asks the question, and the answer
is broadcast to the group. Without that the first rank to leave the loop would
strand the others at the next gradient all-reduce for the full hour-long
timeout.

A few things that are worth knowing before you reach for this:

- **`--nproc_per_node=1` is not a group.** It resolves to the ordinary
  single-process path, which is what you want — there is nothing for a group of
  one to synchronise.
- **The dashboard (`tui`) is single-process.** It owns a terminal, and there is
  only one to own.
- **This is single-node as written.** Multi-node needs the rendezvous flags
  `torchrun` documents (`--nnodes`, `--node_rank`, `--rdzv_endpoint`); the code
  reads `RANK` and `LOCAL_RANK` separately and so is ready for it, but it is
  untested here.
- **MNIST at these sizes will not scale.** The model is 271.8 ms per batch on
  one card and the datasets are small; multi-GPU is for `configs/cifar10.toml`
  at a real width, not for making the smoke config finish sooner.

> **Measured on CPU over gloo, not on multiple GPUs.** The developer machine
> behind these docs has one GPU. What is verified is correctness — that the
> ranks shard the data disjointly, end on bit-identical weights, and write
> exactly one set of files — by `tests/test_distributed.py`, which runs a real
> two-process group on the CPU. The NCCL path and any speedup figure are
> unmeasured. Treat the scaling as untested and check `time/images_per_second`
> on your own hardware.

### Choosing a dataset

`dataset` names an entry in the registry in `tinydiffusion/data/datasets.py`:

| Name | Channels | Native size | Classes | Flips |
| --- | --- | --- | --- | --- |
| `mnist` | 1 | 28 | 10 | no |
| `fashion_mnist` | 1 | 28 | 10 | yes |
| `cifar10` | 3 | 32 | 10 | yes |

Nothing downstream hard-codes a channel count: the U-Net's input and output
width, the shape the samplers draw, and the reference side of a FID all read it
from the spec the config names. Switching datasets is therefore the one key,
plus whatever `num_classes` and `image_size` the new one implies:

```bash
./scripts/run.sh train --config configs/cifar10.toml
./scripts/run.sh train --config configs/mnist.toml --dataset fashion_mnist
```

`num_classes` has to match the dataset's label space exactly, and a mismatch is
refused when the config is read — the labels come from the dataset, so a
smaller count would index past the embedding table the first time a batch
carried a higher one.

The "flips" column is whether a random horizontal flip preserves the label. It
does for natural images and does not for digits, so MNIST opts out. The flip is
applied to the training split only: a scored split never gets one, or a
held-out number would move for reasons that have nothing to do with the
weights.

A checkpoint records the dataset it was trained on, and `--resume` refuses to
carry a run across a change to it — the channel count is part of the shape of
every tensor in the state dict.

### Validation and `best.pt`

After each epoch the EMA weights are scored on a fixed slice of the held-out
test split, at a pinned grid of timesteps with pinned noise. Everything that
would otherwise make the number jump around is held constant, so `val/loss`
moves only with the weights — which is what makes it usable for picking a best
epoch:

```toml
[validation]
val_every = 1     # epochs between scores; 0 turns validation off
val_steps = 10    # timesteps to score at
val_batches = 4   # batches of the test split to score; 0 for all of it
```

The defaults cost a few percent of an epoch. `val_batches = 0` scores the whole
10k split, which is a better absolute number and a much slower one — and it
buys little here, since the same small slice every epoch already isolates the
weights. The score lands in `metrics.jsonl` as `val/loss`, alongside
`val/best_loss`, and drives `best.pt`.

It is scored with the *EMA* weights, because those are what the sample grids
and every downstream command draw from; scoring the live weights would pick a
best epoch nobody ever samples.

### Choosing the parameterisation

`predict`, `variance` and `objective` pick what the network predicts, where the
reverse variance comes from, and what is optimised. Their defaults —
`epsilon` / `fixed_small` / `mse` — are the DDPM baseline and build the `DDPM`
class itself; anything else builds `GaussianDiffusion`, which implements all
three as explicit choices. The combination worth trying is Nichol & Dhariwal's
improved DDPM, which samples well in far fewer steps:

```toml
[diffusion]
variance = "learned_range"   # the net emits the variance alongside the mean
objective = "rescaled_mse"   # L_simple plus a down-weighted variational term
```

A learned variance doubles the U-Net's output channels, so a checkpoint trained
this way is not interchangeable with a baseline one. Bad combinations — a
learned variance under plain `mse`, which would leave the variance head
untrained — are rejected when the config is read.

### Predicting velocity, and closing the terminal-SNR leak

`predict = "v"` has the network predict the *velocity*
`v = sqrt(abar)*eps - sqrt(1-abar)*x0` (Salimans & Ho 2022). It interpolates
between the two obvious targets — epsilon at high noise, `x_0` at low — so no
timestep is left regressing on something it cannot see the signal in.

It pairs with `zero_snr`, which rescales the beta schedule so the last step
carries no signal at all (Lin et al. 2024). A schedule that stops short leaves
`x_T` holding a trace of the image's mean brightness; training always starts
from a real image and never notices, but sampling starts from pure noise, whose
mean is zero, and the model spends the chain restoring a brightness that was
never there. The symptom is samples that are never fully black or fully white,
and it gets worse the fewer steps you take.

```toml
[diffusion]
predict = "v"
zero_snr = true
```

`zero_snr` with `predict = "epsilon"` is rejected: with no signal at `t = T`
there is no epsilon that says anything about `x_0`, which is exactly why the
rescaling is published alongside `v`. How much it buys depends on the schedule
— `linear` leaves `sqrt(abar_T)` around 0.006 over 1000 steps, while `cosine`
clamps its betas at 0.999 and lands within 5e-5 of zero on its own, so there it
mostly just makes the intent exact.

### Weighting the timesteps

Two independent knobs, both aimed at the same problem: a uniform draw of
timesteps with uniform weights spends most of the gradient on the steps that
need it least.

```toml
[diffusion]
loss_weighting = "min_snr"      # clamp each timestep's weight at min_snr_gamma
min_snr_gamma = 5.0             # the paper's value; not sensitive
timestep_sampler = "loss_second_moment"
```

`loss_weighting = "min_snr"` (Hang et al. 2023) weights each timestep by
`min(SNR(t), gamma)`, expressed in whatever space the network predicts in.
Uniform weighting of an epsilon-space MSE is implicitly `1/SNR` weighting in
`x_0` space, so the low-noise timesteps — where the model is nearly right
already — dominate and pull against the high-noise ones. The clamp stops that,
and usually reaches a given loss in noticeably fewer epochs. It applies to the
MSE term only: under a hybrid objective the variational term keeps its own
scale, and under a pure KL objective there is no MSE term to weight, so the
combination is rejected. The logged `train/loss_q*` buckets stay unweighted, so
they remain comparable with each other and across runs.

`timestep_sampler = "loss_second_moment"` (Nichol & Dhariwal 2021) attacks the
other half: which timesteps get drawn at all. It keeps a ten-deep history of
each timestep's loss, samples in proportion to its RMS, and divides the loss
back through by the sampling probability so the estimator stays unbiased. The
draw is uniform until every timestep has a full history. It earns its keep on
the variational objectives, whose per-timestep terms differ by orders of
magnitude, and does very little for plain `mse`. The history lives in memory
only, so a resumed run re-warms it over its first few hundred batches. Under a
multi-GPU run it is the *group's* history: each rank's timesteps and losses are
gathered before they are folded in, so every rank warms on the whole global
batch and they all draw from the same proposal. It costs one small collective
per step and nothing else — the history is kept on the training device and
updated without ever reading a value back to the host, so turning it on does
not reintroduce the per-batch synchronisation the loop is written to avoid.

### Class conditioning

`num_classes` opts into class-conditional training. It is off by default, and
on in both shipped configs — the one place they depart from the defaults:

```toml
[conditioning]
num_classes = 10      # MNIST's digits; has to match the dataset's label space
class_dropout = 0.1   # labels replaced by the null token during training
guidance = 2.0        # scale used at sample time
guidance_rescale = 0.0  # correction for the scale guidance inflates
```

The three work together. `num_classes` gives the U-Net a class embedding,
summed into the timestep embedding. That embedding table holds one extra row —
the null token, meaning "no class given" — and `class_dropout` is what trains
it: 10% of training labels are replaced by it, so the one network learns both a
conditional and an unconditional prediction. `guidance` then extrapolates
between them at sample time, and needs the other two to mean anything;
combinations that cannot work, such as `guidance` above 1 with
`class_dropout = 0`, are rejected when the config is read.

`guidance_rescale` corrects the scale that extrapolation inflates, and is what
keeps a high `guidance` from washing images out; see
[Asking for a particular digit](sampling.md#asking-for-a-particular-digit) for what it does and when to
reach for it. It is 0 by default, and setting it above 0 at `guidance = 1.0` is
rejected rather than silently ignored, since there is no extrapolation there to
correct.

The label costs almost nothing: one embedding row per class, and the sample
grids become class-matched — each generated digit sits directly above a real
one of the same class instead of above an unrelated digit.

Conditional and unconditional checkpoints are not interchangeable, since the
state dict differs. Turning conditioning on part way through a run means
starting it over, not `--resume`.

| Field | Default | Notes |
| --- | --- | --- |
| `dataset` | `mnist` | Also `fashion_mnist`, `cifar10`; see [Choosing a dataset](#choosing-a-dataset) |
| `data_root` | `data` | The dataset is downloaded here on first use |
| `image_size` | 32 | 32 keeps 28x28 digits intact and halves exactly |
| `batch_size` | 128 | 8 GB of VRAM has room for 256 |
| `num_workers` | 4 | 0 when debugging; see [The dataloader is not the bottleneck](#the-dataloader-is-not-the-bottleneck) |
| `base_channels` | 64 | Width; DDPM uses 128 for CIFAR-10 |
| `channel_mult` | `[1, 2, 2]` | One entry per resolution level |
| `num_res_blocks` | 2 | Residual blocks per level |
| `attn_resolutions` | `[16]` | Spatial sizes that get self-attention |
| `dropout` | 0.1 | Inside ResBlocks |
| `num_classes` | unset | Classes to condition on; 10 for MNIST |
| `class_dropout` | 0.1 | Labels replaced by the null token in training |
| `guidance` | 1.0 | Guidance scale; 1.0 is plain conditional |
| `guidance_rescale` | 0.0 | Corrects the scale guidance inflates; 0.7 above `guidance` 3 |
| `num_timesteps` | 1000 | Length of the diffusion schedule |
| `schedule` | `cosine` | `cosine` or `linear` |
| `beta_start` / `beta_end` | 1e-4 / 0.02 | Linear schedule only |
| `predict` | `epsilon` | Also `v` (velocity), `start_x`, `previous_x` |
| `variance` | `fixed_small` | Also `fixed_large`, `learned_range`, `learned` |
| `objective` | `mse` | Also `rescaled_mse` (hybrid), `kl`, `rescaled_kl` |
| `zero_snr` | `false` | Rescale the schedule to zero terminal SNR; needs `predict = "v"` or `"start_x"` |
| `loss_weighting` | `uniform` | Or `min_snr`, clamping each timestep's weight |
| `min_snr_gamma` | 5.0 | The clamp, under `min_snr` weighting |
| `timestep_sampler` | `uniform` | Or `loss_second_moment`, importance sampling the draw |
| `num_epochs` | 30 | |
| `lr` | 2e-4 | Adam |
| `lr_warmup` | 500 | Optimiser steps to ramp the LR over; 0 disables |
| `lr_schedule` | `constant` | Or `cosine`, decaying to zero after the ramp |
| `betas` | `[0.9, 0.999]` | AdamW moment decays |
| `weight_decay` | 0.0 | Decoupled; at 0 AdamW is Adam |
| `grad_accum` | 1 | Micro-batches per optimiser step |
| `grad_clip` | 1.0 | 0 disables clipping |
| `ema_decay` | 0.9999 | Sample quality depends on this |
| `ema_warmup` | 2000 | Steps over which the decay ramps in |
| `seed` | 0 | Python, NumPy and torch RNGs; with the epoch index, fixes the batch order |
| `deterministic` | `false` | Force deterministic CUDA kernels and disable the cuDNN autotuner, at a throughput cost |
| `amp` | `true` | Autocast; ignored off CUDA |
| `amp_dtype` | `fp16` | Or `bf16`, which runs unscaled but needs Ampere or newer; older cards fall back to fp16 rather than emulate it |
| `full_fp16` | `false` | float16 weights with a float32 master copy, instead of autocast; needs CUDA and `amp_dtype = "fp16"` |
| `compile` | `false` | `torch.compile` the training step |
| `channels_last` | `false` | Worth measuring rather than assuming |
| `grad_checkpoint` | `false` | Recompute activations in the backward pass: roughly a third more compute for a large cut in memory |
| `sample_every` | 1 | Epochs between sample grids; 0 disables |
| `num_samples` | 16 | Images per grid |
| `sampler` | `ddim` | Or `dpmpp` (DPM-Solver++(2M)), which needs about a third of the steps |
| `sample_steps` | 50 | Denoising steps for those grids; 15-20 is plenty for `dpmpp` |
| `sample_spacing` | `uniform` | Or `quadratic`, which packs the steps near `t = 0`; free, and worth it at low `sample_steps` |
| `out_dir` | `contents` | Sample grids |
| `ckpt_dir` | `checkpoints` | Checkpoints |
| `device` | auto | `cuda` when available, else `cpu` |
| `log_dir` | `runs/mnist` | `metrics.jsonl`, and `tb/` when TensorBoard is on |
| `log_console` | `true` | Per-epoch metrics table |
| `log_jsonl` | `true` | Append `metrics.jsonl` |
| `tensorboard` | `false` | TensorBoard events; needs the `tracking` extra |

`configs/mnist.toml` lists every field with its default, `[conditioning]`
aside;
`TrainConfig` in `src/tinydiffusion/training/config.py` is the source of truth.
