# Sampling

Turning a checkpoint into images: samplers, step counts, precision,
guidance, and latent walks.

Part of [Usage](../../USAGE.md).

## Sampling

```bash
./scripts/run.sh sample --checkpoint checkpoints/last.pt --num-images 16 --out contents/grid.png
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to sample from |
| `--num-images` | 8 | How many images to generate |
| `--batch-size` | all of them at once | Images drawn at a time; sets peak memory |
| `--steps` | the checkpoint's `sample_steps` | Denoising steps; fewer is faster, coarser |
| `--sampler` | the checkpoint's `sampler` | `ddim`, `dpmpp`, `heun` or `plms`; see [Choosing a sampler](#choosing-a-sampler) |
| `--spacing` | the checkpoint's `sample_spacing` | `uniform` or `quadratic`; see [Spacing the steps](#spacing-the-steps) |
| `--eta` | 0.0 | 0 is deterministic DDIM, 1 is ancestral DDPM. Only `ddim` accepts anything else |
| `--labels` | one image per class | Classes to generate, e.g. `7` or `0,1,2` |
| `--guidance` | the checkpoint's `guidance` | Classifier-free guidance scale |
| `--guidance-rescale` | the checkpoint's `guidance_rescale` | Corrects the scale guidance inflates; 0.7 above `--guidance 3` |
| `--out` | `contents/samples.png` | Where to write the grid |
| `--save-individual` | off | Also write each image on its own beside the grid |
| `--seed` | 0 | Seed applied before sampling |
| `--device` | auto | `cuda`, `cpu`, `cuda:1`, … |
| `--precision` | `fp32` | `fp32`, `tf32`, `fp16` or `bf16`; see [Half precision](#half-precision) |

Checkpoints embed the config they were trained with, so this reconstructs the
architecture from the `.pt` alone — the TOML that produced it is not needed.
Sampling always uses the EMA weights, which is what the training grids are
drawn from.

### Generating more images than fit at once

A sampler runs one reverse chain over the whole batch it is handed, so the
memory a draw needs follows `--num-images` directly, and asking an 8 GB card
for a few hundred images at once is an out-of-memory error rather than a slow
run. `--batch-size` splits the draw instead:

```bash
./scripts/run.sh sample --checkpoint checkpoints/last.pt \
  --num-images 256 --batch-size 32 --save-individual --out contents/set.png
```

The split does not change which images come out. Every image's starting latent
is drawn before the first batch and handed out in order, so image `i` gets the
latent and the label it would have had unsplit — `--batch-size` is a memory
knob, not a sampling one. Two things sit underneath that. `--eta 0` (the
default) is deterministic, but a positive `--eta` draws per-step noise per
batch and so does follow the split; and on a GPU the convolutions choose their
algorithm by batch shape, which leaves two splits agreeing to a pixel or so of
rounding rather than byte-for-byte.

`--save-individual` writes each image beside the grid, named after it:
`set.png` gives `set_0000.png` through `set_0255.png`. The grid is for looking
at; the individual files are what anything downstream actually reads.

### Choosing a sampler

Four samplers, all usable with any checkpoint — which one to draw with is a
runtime choice, not something baked into the weights:

| `--sampler` | What it is | Order | Network calls per step | Steps it wants |
| --- | --- | --- | --- | --- |
| `ddim` | DDIM (Song et al. 2020), a step along the probability-flow ODE | 1st | 1 | 50 |
| `dpmpp` | DPM-Solver++(2M) (Lu et al. 2022), multistep | 2nd | 1 | 15-20 |
| `heun` | Heun's method, as EDM (Karras et al. 2022) uses it | 2nd | 2 | 15-20 |
| `plms` | PLMS (Liu et al. 2022), Adams-Bashforth on the noise estimate | 4th | 1 | 20+ |

They differ in what they spend to beat DDIM's first-order step, and the whole
comparison is only meaningful at a fixed number of *network evaluations* —
which is `--steps` for three of them and twice `--steps` for `heun`.

`dpmpp` integrates the linear part of the ODE in closed form and approximates
only the `x_0` prediction, reusing the previous step's network evaluation rather
than paying for a second one. A step costs exactly what a DDIM step costs, and
ten to twenty of them land about where fifty DDIM steps do:

```bash
./scripts/run.sh sample --checkpoint checkpoints/last.pt --sampler dpmpp --steps 20
```

`heun` gets its second order the other way: it takes the DDIM step
provisionally, evaluates the network again at where it landed, and re-takes the
step along the average of the two directions. That is two calls a step rather
than one — so compare it at half the step count — and it buys something
`dpmpp` cannot, which is being correct from the *first* step. `dpmpp` has no
history to extrapolate from until its second, and a very short chain is mostly
first steps:

```bash
./scripts/run.sh sample --checkpoint checkpoints/last.pt --sampler heun --steps 10
```

`plms` remembers instead of re-evaluating: it fits a cubic through the last four
noise estimates, which are already paid for, and extrapolates. Fourth order at
one call a step, and the cheapest order on offer — but the first three steps
have no history, so the order ramps 1, 2, 3, 4 and those early steps are exactly
where a short chain is worst. Below about 15 steps the ramp eats the benefit:

```bash
./scripts/run.sh sample --checkpoint checkpoints/last.pt --sampler plms --steps 25
```

Only `ddim` is stochastic. The other three integrate the probability-flow ODE,
which has no noise term to scale, so `--eta` above 0 is refused rather than
ignored. `--eta 1` remains available under `ddim`, where it reproduces ancestral
DDPM sampling.

Which of the four wins is a property of the model, not of the list. `fid --kid`
at a fixed evaluation budget is the comparison that settles it — see
[Measuring sample quality](evaluation.md#measuring-sample-quality).

`sampler` is also a config field, so a run's per-epoch grids are drawn the same
way, and a checkpoint remembers what it was trained to be sampled with.
Sampling settings move FID, so hold `--sampler` and `--steps` fixed across the
checkpoints being compared.

### Spacing the steps

`--steps` says how many timesteps of the 1000-step training schedule to visit.
`--spacing` says *which*:

| `--spacing` | Where the steps go | Defined on |
| --- | --- | --- |
| `uniform` | Evenly across the schedule | the index |
| `quadratic` | Packed towards `t = 0`, the low-noise end | the index |
| `karras` | Evenly across the *noise level* | the schedule's sigmas |

All three take the same number of network evaluations, so this is free to try.
It matters when `--steps` is small: the last few steps are where a short chain
has the least room to correct itself, and spending more of the budget there is
what the DDIM paper found better on CIFAR-10 at low step counts.

```bash
./scripts/run.sh sample --checkpoint checkpoints/last.pt --steps 15 --spacing quadratic
```

At 50 steps or more they are hard to tell apart, so the default stays
`uniform`. Like `--sampler` it is also a config field — `sample_spacing` — so
the per-epoch grids and the [sampling server](serving.md#serving-a-checkpoint-over-http)
follow whatever the run was configured with, and it moves FID like any other
sampling setting.

#### `karras`, and what it actually buys

`uniform` and `quadratic` both space the steps by *index*: they pick every 50th
timestep, or something denser at one end. `karras` (Karras et al. 2022) spaces
them by *noise level* instead. Rewriting the forward process as
`x_0 + sigma * eps` gives each timestep a sigma, and the EDM ramp puts the
steps evenly along `sigma^(1/7)` — evenly in the thing the denoiser actually
sees, rather than in an index that is only a proxy for it.

Measured on the shipped MNIST model, 1,000 images, comparing at equal network
evaluations. KID is quoted with its spread, which is what says the gaps are
real rather than noise:

| NFEs | `uniform` | `quadratic` | `karras` |
| --- | --- | --- | --- |
| 12 | 0.01360 ± 0.00117 | **0.00452** ± 0.00048 | 0.00963 ± 0.00083 |
| ~20 | 0.00720 ± 0.00084 | **0.00273** ± 0.00042 | 0.00471 ± 0.00060 |

So `karras` is a clear improvement on `uniform` — roughly halving the KID at
both budgets, by several times either error bar — and is beaten by `quadratic`
on this model. That last part is a fact about a cosine-schedule MNIST model and
not a general claim; the reason `fid --kid` exists is that you can check
instead of believing.

One wart, and it is a real one: **`karras` does not honour `--steps` on a
cosine schedule.** The ramp is anchored to the schedule's largest sigma, and
cosine ends at `abar ≈ 2e-9` — a sigma of about 20,000, against the 80 the ramp
was designed around. Part of the ramp therefore lands inside the handful of
timesteps above `t = 936`, and rounding back to whole timesteps collapses them:

```
--steps 20 --spacing karras   ->  12 steps actually taken
--steps 40 --spacing karras   ->  22
```

Ask for roughly double what you want. A `linear` schedule ends at a sigma of
about 157 and does not have the problem — there, 20 steps are 20.

Note that `sample_spacing` is unrelated to `timestep_sampler`, which decides
which timesteps a *training* batch is drawn at; see
[Weighting the timesteps](configuration.md#weighting-the-timesteps).

### Half precision

`--precision` decides what the network runs in while it draws. It is accepted
by every command that samples — `sample`, `interpolate`, `fid` and `serve` —
and the default is `fp32` on all four, which is bit-for-bit what they did
before the flag existed.

| `--precision` | What it does | Needs |
| --- | --- | --- |
| `fp32` | float32 throughout. The default | — |
| `tf32` | float32 storage, reduced-mantissa matmul and convolution kernels | Ampere or newer |
| `fp16` | Half precision through autocast, in NHWC | tensor cores |
| `bf16` | As `fp16`, with float32's exponent range | Ampere or newer |

Anything but `fp32` falls back to it off CUDA, and `bf16` falls back to `fp16`
on a card that only emulates bfloat16 — the same fallback, and the same printed
line, that a training run makes. Nothing is silent: the message names what it
is running instead.

Sampling is where a diffusion model's arithmetic actually is. A `fid` over
10,000 images at 50 steps with guidance is a million network evaluations, and
until this flag existed every one of them ran in float32.

Measured on a Turing card (RTX 2070) at the `configs/mnist.toml` geometry —
6.95M parameters, 32px, attention at 16, batch 128, 50 DDIM steps, guidance 2:

| `--precision` | Throughput | vs `fp32` |
| --- | --- | --- |
| `fp32` | 10.9 img/s | — |
| `tf32` | 11.7 img/s | 1.08x |
| `fp16` | 16.4 img/s | **1.51x** |

`bf16` is not listed because that card has no bfloat16 hardware and falls back
to `fp16`; `tf32` gains what it does there for the same reason, since TF32 units
arrived with Ampere. Both are worth re-measuring on your own card.

Half precision brings the memory format with it — the wrapper runs the network
in NHWC, because tensor cores read that layout and cuDNN handed an NCHW
half-precision tensor transposes it per convolution instead. That is most of
the difference: on the same card, the forward pass alone goes 1.12x faster in
NCHW `fp16` and 2.36x faster in NHWC `fp16`. It is not a separate switch,
because on its own it is a pessimisation — NHWC float32 is markedly *slower*
than NCHW float32 — so it travels with the dtype and only with the dtype.

What it costs is small but not nothing. Drawing 64 images from the same seed
at 50 steps, `fp16` moves a pixel by 0.09 of a 255-level on average and 18
levels at the worst single pixel — invisible in a grid, and not zero. So it is
a measurement setting as much as a speed one: **hold it fixed across the
checkpoints you are comparing**, exactly as you would `--sampler` or `--steps`.
`fid` records it and prints it in the report whenever it is not `fp32`, so a
score cannot quietly claim to be comparable with one drawn at another
precision.

Two things deliberately stay in float32 however this is set. Classifier-free
guidance extrapolates and rescales in float32 — the rescale takes a standard
deviation over a whole image, which is not a thing to ask of ten mantissa bits
— because the wrapper goes under the conditioning rather than over it. And
`fid`'s feature extractor is untouched: it is the instrument the score is
measured with, the cached reference features on the real half were computed
with it, and it is one network evaluation per image against sampling's
hundred.

`serve` resolves the setting once at startup rather than per request, and
reports it on `/api/status`. A caller cannot choose it: the speed-for-accuracy
trade is the operator's, and two identical requests should not come back
different.

### Walking between two latents

`sample` draws unrelated images. `interpolate` draws a path between two of
them, and the path is the interesting part: every sampler here is a
deterministic function of its starting latent, so a smooth walk through latent
space comes out as a smooth walk through image space — where the model has
learned one. Where it has not, the strip snaps from one digit to another
partway across, which is the model telling you it has two modes and nothing in
between.

```bash
./scripts/run.sh interpolate --checkpoint checkpoints/last.pt --labels 7 --steps 10
```

That holds the class fixed and moves only the latent, so the strip shows what
varies *within* a 7 — stroke weight, slant, whether the bar is crossed. The two
ends are named by seed, so a walk you liked is one you can get back:

```bash
./scripts/run.sh interpolate --checkpoint checkpoints/last.pt --labels 3 \
  --seed-start 4 --seed-end 9 --out contents/threes.png
```

The path is *spherical*, not a straight line, and that is not a detail. A
latent is a draw from an isotropic Gaussian, whose mass sits almost entirely in
a thin shell at radius `sqrt(d)` — 55.4 for a 1×32×32 image. The midpoint of a
straight line between two such draws has an expected norm around `0.71 ×
sqrt(d)`, well inside the shell and nowhere the model was ever trained, so a
linear walk washes out in the middle and comes back. Slerp travels along the
shell, and every point on it is as plausible a latent as the two ends.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to sample from |
| `--steps` | 8 | Points along the walk, counting both ends |
| `--denoise-steps` | the checkpoint's | Denoising steps per image |
| `--sampler` | the checkpoint's | `ddim` or `dpmpp` |
| `--spacing` | the checkpoint's | `uniform`, `quadratic` or `karras` |
| `--labels` | class 0 | Class held fixed across the walk |
| `--guidance` | the checkpoint's | Classifier-free guidance scale |
| `--guidance-rescale` | the checkpoint's | Guidance rescale factor |
| `--seed-start` | 0 | Seed for the latent the walk starts at |
| `--seed-end` | 1 | Seed for the latent it ends at |
| `--out` | `contents/interpolation.png` | Output |
| `--device` | auto | `cuda`, `cpu`, … |
| `--precision` | `fp32` | `fp32`, `tf32`, `fp16` or `bf16`; see [Half precision](#half-precision) |

Sampling is deterministic here — `eta` is pinned at 0 — because a walk whose
points each drew their own noise would be varying for two reasons at once and
showing neither.

### Asking for a particular digit

`configs/mnist.toml` trains class-conditionally, so the checkpoint knows which
digit is which:

```bash
./scripts/run.sh sample --checkpoint checkpoints/last.pt --labels 7 --num-images 8
./scripts/run.sh sample --checkpoint checkpoints/last.pt --labels 0,1,2 --guidance 4
```

`--labels` takes a comma-separated list, repeated in order until the grid is
full: `7` fills it with sevens, `0,1,2` cycles the three. Leaving it out gives
one image per class, laid out one class per column.

`--guidance` is the classifier-free guidance scale. At 1.0 you get the plain
conditional prediction. Above that, each step extrapolates away from the
model's unconditional prediction — digits get cleaner and more emphatically
class-typical, variety drops, and every step costs a second forward pass. 2–4
is the useful range on MNIST; past about 6 strokes start to blow out.

`--guidance-rescale` is what pushes that ceiling back. Extrapolation makes the
prediction *larger*, not just better aimed: its standard deviation grows with
the scale, and since the model was trained on targets of a fixed scale, the
recovered image ends up driven to the extremes of the range — the blow-out
above. Lin et al. 2023 §3.4 rescale the guided prediction back onto the
conditional one's standard deviation, then blend, since going all the way is
itself too strong:

```bash
./scripts/run.sh sample --checkpoint checkpoints/last.pt --guidance 5 --guidance-rescale 0.7
```

0.7 is the published blend and a good starting point. 0 is plain guidance and
the default, since the correction only matters once the scale is high enough to
need it. It is worth most on a `predict = "v"` model trained with `zero_snr`,
where the terminal step carries no signal to anchor the scale against.

All three flags are errors against an unconditional checkpoint, which has no
class space to name.
