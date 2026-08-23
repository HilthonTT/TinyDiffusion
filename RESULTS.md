# Results

What the shipped configs actually score, on hardware that exists, with the
commands that produced each number. Nothing here is copied from a paper.

Every figure is reproducible from this repository: the command is given in
full, the checkpoint it was taken from is named, and the seed is fixed. Where a
run was cut short of what its config asks for, the table says so rather than
implying the config's number.

## The machine

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Laptop (8 GB) |
| Platform | Windows 11, build 26200 |
| PyTorch | 2.13.0+cu132 |
| Python | 3.14.7 |

One GPU. Every multi-GPU claim in [USAGE.md](USAGE.md#training-on-several-gpus)
is a correctness claim verified on CPU over gloo, not a speedup measured here —
see [What is not measured](#what-is-not-measured).

## MNIST

`configs/mnist.toml`, class-conditional on the ten digits, 6.95M parameters.

```bash
tinydiffusion train --config configs/mnist.toml \
  --set 'ckpt_dir="runs/results-mnist/checkpoints"' \
  --set 'out_dir="runs/results-mnist/contents"' \
  --log-dir runs/results-mnist
```

> **Trained for 8 of the config's 30 epochs.** The run was stopped early. The
> scores below are therefore what 8 epochs buys, not what the config is capable
> of — validation loss was still falling when it stopped, so treat these as a
> floor rather than a converged result.

### Sample quality

```bash
tinydiffusion fid --checkpoint runs/results-mnist/checkpoints/best.pt \
  --num-images 5000 --split train --kid --precision-recall --seed 0
```

| Metric | Value |
| --- | --- |
| FID | **15.194** |
| KID | **0.00573 ± 0.00055** (100 subsets of 1000) |
| Precision | 0.669 |
| Recall | 0.634 |

5000 generated against 5000 real training images, EMA weights, 50 DDIM steps on
uniform spacing, guidance 2.0, seed 0.

Reading them together: precision and recall are close, so at 8 epochs the model
is neither collapsed onto a few clean digits nor spraying coverage it cannot
render — it is simply not finished. The FID is dominated by that, and by the
5000-image sample count, which carries real bias of its own. KID does not, which
is why it is quoted alongside; the ± is a spread over subsets, so two
checkpoints under a point of FID apart can still be told from noise.

### Training curve

Batch 128, AMP fp16, `num_workers = 4`, seed 0.

| Epoch | train/loss | val/loss | seconds | img/s |
| --- | --- | --- | --- | --- |
| 1 | 0.2722 | 0.0544 | 112 | 534 |
| 2 | 0.0333 | 0.0457 | 83 | 719 |
| 3 | 0.0306 | 0.0425 | 84 | 716 |
| 4 | 0.0288 | 0.0408 | 87 | 692 |
| 5 | 0.0284 | 0.0403 | 91 | 660 |
| 6 | 0.0276 | 0.0402 | 83 | 718 |
| 7 | 0.0273 | 0.0400 | 111 | 540 |
| 8 | 0.0270 | 0.0399 | 195 | 308 |

About 85 seconds per epoch at ~700 img/s when the card is otherwise idle.
Epochs 7 and 8 are slower because a scoring job was competing for the same GPU
— a useful reminder that `time/images_per_second` measures the machine, not the
code, and that these columns are only comparable against each other when
nothing else is running.

Best validation loss 0.03985 at epoch 8, which is the checkpoint `best.pt`
holds and the one scored above. The gap between epochs 6 and 8 is 0.0003 —
flattening, but not flat.

## What is not measured

Stated plainly, because a results file that quietly omits these is worse than
one that has none:

- **CIFAR-10.** `configs/cifar10.toml` ships and trains, but no scored run
  exists. It is the config that would actually exercise the model, and it wants
  far more than 8 epochs.
- **A converged MNIST run.** 30 epochs is what the config asks for; 8 is what
  was run.
- **Multi-GPU throughput.** The scaling claim is untested — one GPU here. What
  *is* verified is that a two-rank group shards the data disjointly, ends on
  bit-identical weights and writes exactly one set of files
  (`tests/test_distributed.py`, real subprocesses over gloo).
- **Sampler and spacing comparisons.** USAGE.md argues DPM-Solver++ at 15–20
  steps reaches DDIM's 50-step quality, and that `quadratic` and `karras`
  spacing move a score. Both are reasoned rather than measured on a scored
  checkpoint; `fid --sampler dpmpp --steps 20` against the same checkpoint is
  the experiment.
- **A guidance sweep.** FID usually bottoms out above a scale of 1, and 2.0 was
  inherited from the config rather than chosen by measurement.

## Reproducing these

```bash
uv sync --all-extras --dev
tinydiffusion train --config configs/mnist.toml --epochs 8
tinydiffusion fid --checkpoint checkpoints/best.pt \
  --num-images 5000 --split train --kid --precision-recall --seed 0
```

Expect the third decimal to move. The training seed is fixed and the sampling
seed is fixed, but cuDNN's autotuner picks kernels on the day
(`deterministic = true` in the config pins them, at a throughput cost), and the
scores are taken over 5000 images rather than the 10,000 `fid` defaults to.
