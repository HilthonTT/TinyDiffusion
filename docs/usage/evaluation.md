# Evaluating a checkpoint

Held-out loss, the variational bound, and the sample-quality scores — FID,
sFID, KID, precision and recall, the Inception Score — that loss cannot tell
you.

Part of [Usage](../../USAGE.md).

- [Evaluating a checkpoint](#evaluating-a-checkpoint)
- [Bits per dimension](#bits-per-dimension)
- [Measuring sample quality](#measuring-sample-quality)

## Evaluating a checkpoint

Sampling shows you what the model draws; `eval` puts a number on it, by scoring
noise-prediction loss over the 10k held-out MNIST test split:

```bash
./scripts/run.sh eval --checkpoint checkpoints/last.pt
```

```
checkpoints/last.pt | test split | 10000 images | ema weights
loss 0.09984

     t     loss
     0   0.39324
    40   0.06916
    80   0.04649
   119   0.03946
   159   0.03071
   199   0.02001
```

The training loss is drawn at a *random* timestep per image, so it is far too
noisy to compare two checkpoints with. This pins the timesteps to a fixed grid
and reseeds before every batch, so the only thing that varies between two runs
is the weights — run it twice on the same checkpoint and you get the same
number to the last digit.

A conditional checkpoint is scored on the true labels, and never with
guidance: the objective it was trained on is the conditional prediction, so
scoring anything else would measure something the run never optimised.

Read the per-timestep column as well as the headline. Loss is always highest at
`t = 0`, where `x_t` is nearly clean and there is almost no noise left to
identify, and falls as `t` grows. A checkpoint that improves only at high `t`
has learned the easy end of the schedule and little else.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to score |
| `--split` | `test` | `test` (10k held out) or `train` (60k) |
| `--timesteps` | 10 | How many timesteps to score at |
| `--batch-size` | the checkpoint's | Larger is faster |
| `--data-root` | the checkpoint's | Dataset directory |
| `--no-ema` | off | Score the raw weights instead of the EMA |
| `--bpd` | off | Also evaluate the variational bound; see [Bits per dimension](#bits-per-dimension) |
| `--bpd-images` | 128 | Images the bound is estimated over |
| `--seed` | 0 | Fixes the noise; change it to resample |
| `--device` | auto | `cuda`, `cpu`, … |

Scoring both splits is how you see overfitting — a test loss that stalls or
rises while the train loss keeps falling:

```bash
./scripts/run.sh eval --checkpoint checkpoints/last.pt --split test
./scripts/run.sh eval --checkpoint checkpoints/last.pt --split train
```

One caveat: this is a proxy. Lower held-out loss means the network predicts
noise better, which correlates with sample quality but does not measure it
directly. For that, use `fid` below — and keep looking at the grids.

## Bits per dimension

The held-out loss is whatever *this* run was trained on, which is what makes it
useless for comparing a v-prediction model against an epsilon one, or either
against a published number. The variational bound is not: every diffusion model
defines the same bound on the negative log-likelihood of real data, and
`--bpd` evaluates it.

```bash
./scripts/run.sh eval --checkpoint checkpoints/last.pt --bpd
```

```
checkpoints/last.pt | test split | 10000 images | ema weights
loss 0.09984
bpd 3.61204 (prior 0.00003) over 128 images

     t     loss
...
```

Two numbers, and the second is the more interesting one. `prior` is the gap
between `q(x_T | x_0)` and the standard normal the chain starts from. It depends
on the schedule alone — no amount of training moves it — so a large share of the
total means the forward process has not finished destroying the signal by `x_T`,
which is exactly what
[`zero_snr`](configuration.md#choosing-the-parameterisation) exists to fix.

Three things to know before reading the number:

- **It needs the generalised process.** The default parameterisation is served
  by the plain DDPM implementation, which has no bound to walk; the command says
  so rather than guessing. Train with a non-default `predict`, `variance` or
  `objective` — `variance = "learned_range"` with `objective = "rescaled_mse"`
  is the configuration the bound is normally quoted for, and the one Nichol &
  Dhariwal's improved DDPM is.
- **It is expensive.** The bound walks *every* timestep of the training
  schedule, so it costs `num_timesteps` network evaluations per image — a
  thousand at the default — against the couple of dozen the loss spends. Hence
  `--bpd-images`, which defaults to 128; a few hundred is enough for the third
  decimal, and the whole split is neither affordable nor necessary.
- **It disagrees with FID, and that is not a bug.** Likelihood and sample
  quality are different questions, and a model can win on one and lose on the
  other. Reporting both is the point.

## Measuring sample quality

`eval` scores the training objective; `fid` scores the thing you actually care
about. It draws samples, pushes them and an equal number of real images through
a pretrained Inception-v3, and measures the distance between the two clouds of
activations.

FID is the default and always reported. Four more are available on request, and
each answers something FID cannot: `--kid` is unbiased, so it survives the small
sample counts a single GPU can afford; `--precision-recall` splits a bad score
into the two different problems it might be; `--sfid` sees the spatial
incoherence FID's pooled features average away; and `--inception-score` reads
the samples without reference to the real data at all.

```bash
./scripts/run.sh fid --checkpoint checkpoints/last.pt
```

```
checkpoints/last.pt | train split | ema weights
fid 18.472

10000 generated vs 10000 real images
50 ddim steps (uniform spacing) | guidance 2
```

The Inception weights (~100 MB) download on first use into the usual torch hub
cache; see [What gets downloaded, and where](install.md#what-gets-downloaded-and-where).

### KID, when 10,000 samples is too many

FID fits a Gaussian to each cloud, and a 2048-dimensional covariance estimated
from fewer than ~2048 images is singular. The error that introduces does not
average out: it is a bias, always upwards, and its size depends on the sample
count. A FID over 1,000 images is therefore not a noisier estimate of the FID
over 50,000 — it is a different number, and the two cannot be compared.

KID has no Gaussian in it. It is a kernel distance between the two sets in the
unbiased form, so its expected value does not move with the sample count, and a
score over 1,000 images means the same thing as one over 50,000. It also comes
with a spread, which FID cannot offer at all:

```bash
./scripts/run.sh fid --checkpoint checkpoints/last.pt --num-images 2000 --kid
```

```
checkpoints/last.pt | train split | ema weights
fid 34.118
kid 0.02170 +- 0.00184 (100 subsets of 1000)

2000 generated vs 2000 real images
50 ddim steps (uniform spacing) | guidance 2

warning: fewer than 2048 images per side, so the covariance is singular and
the fid is biased upwards. Compare it only with scores taken at the same image
count.
```

The warning is about the FID on the line above it, and not about the KID: at
2,000 images the first number is mostly reporting its own sample count and the
second is not. Without `--kid` the report says so and points here.

The spread is the useful part. It is how much one subset of 1,000 images
disagrees with another, so two checkpoints whose KIDs differ by less than it
have not been told apart — which is exactly the judgement a bare FID invites
you to get wrong. Hold `--kid-subset-size` fixed across the checkpoints you
compare, since the spread is a spread over subsets of that size.

This is the metric to reach for during a run, where 10,000 samples through the
full sampling chain is a long wait for one number.

### Precision and recall, when the score is bad and you need to know why

A single number cannot distinguish a model that makes beautiful images of three
digits from one that makes all ten badly. They call for opposite fixes, and FID
and KID both score them the same.

Precision and recall estimate the two clouds' *manifolds* instead — a ball
around each feature vector reaching its k-th nearest neighbour — and ask how
much of each lands inside the other:

- **precision** is the fraction of generated images inside the real manifold:
  how much of what the model makes is realistic.
- **recall** is the fraction of real images inside the generated manifold: how
  much of the real data the model reaches.

```bash
./scripts/run.sh fid --checkpoint checkpoints/last.pt --num-images 2000 --precision-recall
```

```
checkpoints/last.pt | train split | ema weights
fid 34.118
precision 0.681 | recall 0.412 (k=3)

2000 generated vs 2000 real images
50 ddim steps (uniform spacing) | guidance 2

warning: fewer than 2048 images per side, so the covariance is singular and
this score is biased upwards. Compare it only with scores taken at the same
image count.
--kid is unbiased at this count and does not have that problem.
```

Both fractions are honest at this count — it is the FID above them the warning
is about. The two flags combine, and at these counts they usually should.

Guidance moves them in opposite directions, which is the clearest thing either
number does: sweeping `--guidance` and watching precision climb while recall
falls shows you the trade being made, where the FID minimum only tells you
where it balances.

The cost is quadratic in `--num-images` — every generated image is measured
against every real one — so this is a flag for the low thousands, not for
50,000.

### sFID, when the parts are right and the whole is not

FID's features are *pooled* — averaged over the image — which is what makes them
a summary of what an image contains and blind to where. A model that draws
perfect strokes and assembles them into something that is not a digit scores
well on a metric that cannot see the arrangement.

sFID (Nash et al. 2021) is the same distance taken in an *unpooled* reading of
the same network: the first seven channels of an intermediate Inception feature
map, kept spatially, for 7 x 17 x 17 = 2023 dimensions against FID's 2048. It
rides along on the Inception pass FID is already making, so it costs almost no
time:

```bash
./scripts/run.sh fid --checkpoint checkpoints/last.pt --sfid
```

```
checkpoints/last.pt | train split | ema weights
fid 18.472
sfid 9.317
```

Read the two together — a good FID beside a bad sFID is the finding, and says
the samples have the right ingredients in the wrong places. The absolute values
are not comparable with each other, only each with itself across checkpoints.
It caches its reference half separately, under the same key plus `_spatial`,
so the first `--sfid` run re-reads the real images even if a plain `fid` has
already scored the same set.

### The Inception Score, and why it is here

The Inception Score never looks at a real image. It asks the classifier whether
each sample is confidently *some* ImageNet class and whether the samples between
them cover *many*, and reports the exponentiated KL between the two. High means
individually decisive and collectively varied.

```bash
./scripts/run.sh fid --checkpoint checkpoints/last.pt --inception-score
```

```
checkpoints/last.pt | train split | ema weights
fid 18.472
inception score 2.114 +- 0.087 (10 splits of 1000)
```

Not looking at the real data is the whole of its appeal and the whole of its
problem: it cannot tell you whether the samples resemble *your* dataset, only
whether they resemble ImageNet. **On MNIST that is close to meaningless** —
handwritten digits are not an ImageNet class, and a model that has learned them
perfectly scores whatever Inception happens to think a 7 is. It earns its keep
on natural images, and it is here because it is free on a pass that is already
running Inception over every sample.

The spread is over `--is-splits` disjoint chunks of the sample set, so it says
whether the number is stable, not whether two models differ. The score depends
on the chunk size, so hold `--is-splits` and `--num-images` fixed across
checkpoints.

### The reference features are cached

Half of every score is the real images, and that half does not depend on the
checkpoint or on any sampling setting: for a given dataset, split, resolution
and image count it is the same images through the same network every time. So
it is computed once and kept, under `<data_root>/fid_cache`.

This is what makes a sweep affordable. Without it, five guidance scales at
`--num-images 10000` push 50,000 real images through Inception-v3 to compute
one number five times:

```bash
for g in 1 2 3 4 5; do
  ./scripts/run.sh fid --checkpoint checkpoints/last.pt --guidance $g
done
```

Only the first of those pays for the real half; the rest read it back and go
straight to generating.

Everything that moves the statistics is part of the entry's name, so changing
`--split`, `--num-images`, `image_size` or the feature network is a miss rather
than a stale read, and a file that cannot be read is treated as absent. Nothing
you can do to the cache changes a score — only how long it takes.

An entry is about 33 MB (the covariance is 2048 x 2048 in float64, which is the
precision the accumulation needs). Delete `<data_root>/fid_cache` whenever you
want the space back; the next score simply recomputes. `--no-cache` skips the
cache for one run without deleting anything, which is the flag to reach for if
you ever want to confirm an entry against a fresh pass.

`--kid` and `--precision-recall` need the feature vectors themselves rather
than their moments, so they write a second entry alongside — same name, plus
`_features`. That one grows with the image count: about 8 KB an image, so 80 MB
at `--num-images 10000` against the moments' flat 33 MB. It is only written when
one of those flags actually asked for it, so a FID-only sweep keeps paying the
smaller price. A moments entry cannot stand in for a feature one, so the first
`--kid` run re-reads the real images even if a plain `fid` has already been
scored over the same set.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to score |
| `--num-images` | 10000 | Samples drawn, and real images compared against |
| `--split` | `train` | Real split to compare against |
| `--batch-size` | the checkpoint's | Larger is faster |
| `--data-root` | the checkpoint's | Dataset directory |
| `--steps` | the checkpoint's | Denoising steps per sample |
| `--sampler` | the checkpoint's | `ddim`, `dpmpp`, `heun` or `plms`; hold it fixed across compared checkpoints |
| `--spacing` | the checkpoint's | `uniform`, `quadratic` or `karras`; hold it fixed too |
| `--eta` | 0.0 | 0 is DDIM, 1 is ancestral DDPM |
| `--guidance` | the checkpoint's | Classifier-free guidance scale |
| `--guidance-rescale` | the checkpoint's | Guidance rescale factor; sweep it jointly with `--guidance` |
| `--no-ema` | off | Sample the raw weights instead of the EMA |
| `--no-cache` | off | Recompute the real images' features instead of reusing them |
| `--kid` | off | Also report the Kernel Inception Distance, with its spread |
| `--kid-subsets` | 100 | Subsets the KID is averaged over |
| `--kid-subset-size` | 1000 | Images per KID subset, per side; hold it fixed across compared checkpoints |
| `--precision-recall` | off | Also report manifold precision and recall |
| `--neighbours` | 3 | k for the precision/recall manifolds |
| `--sfid` | off | Also report the spatial FID |
| `--inception-score` | off | Also report the Inception Score, with its spread |
| `--is-splits` | 10 | Chunks the Inception Score is averaged over |
| `--seed` | 0 | Fixes the samples; change it to redraw |
| `--device` | auto | `cuda`, `cpu`, … |
| `--precision` | `fp32` | `fp32`, `tf32`, `fp16` or `bf16`; see [Half precision](sampling.md#half-precision) |

Read the number as a comparison, never as an absolute:

- **It only compares like with like.** `--num-images`, `--split`, `--steps` and
  the extractor each move the score on their own, so hold them fixed across the
  checkpoints you are comparing.
- **Small sample counts inflate it.** The Inception feature space is 2048-dimensional,
  so a covariance estimated from fewer than ~2048 images per side is singular and
  the score is biased upwards by an amount that depends on the count. Below that
  the report says so, and points at `--kid`, which does not have the problem.
  Use 10k when the FID needs to mean anything, and `--kid` below that.
- **It is not the published FID.** Those numbers come from the original
  TensorFlow Inception graph; torchvision's port differs enough to shift the
  absolute value. The ordering it induces over checkpoints is what carries over.
- **A conditional model is sampled with a uniform class mix**, while real MNIST
  is not — the first 10k training images run from 8.6% (5s) to 11.3% (1s). That
  prior mismatch adds a small constant to the score. It is identical for every
  checkpoint scored the same way, so comparisons are unaffected; it is one more
  reason not to read the absolute number.

It is also slow — every score runs the full DDIM chain `--num-images` times —
which is why it is a command you run at the end of a run rather than a metric
logged per epoch.

Guidance is worth sweeping rather than leaving at the checkpoint's default. It
trades diversity for fidelity, and FID punishes both, so the minimum usually
sits somewhere above 1:

```bash
for g in 1 1.5 2 3 5; do
  ./scripts/run.sh fid --checkpoint checkpoints/last.pt --num-images 2000 --guidance $g
done
```
