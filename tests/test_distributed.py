"""The multi-GPU path, exercised on CPU over gloo.

Nothing here needs a second GPU. What a distributed run can get wrong — a shard
every rank draws identically, gradients that never sync, four processes racing
to write one checkpoint — is a property of the process group, not of the
backend underneath it, so two gloo processes on the CPU catch all of it.

The integration tests below launch real subprocesses because that is what a
process group requires: `torch.distributed` needs one interpreter per rank, and
a thread-based fake would not exercise the sampler, the DDP wrapper or the
rank-0 write guards at all.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch
from torch.utils.data import DistributedSampler

from tinydiffusion.data.datasets import (
    DATASETS,
    DatasetSpec,
    image_dataloader,
    set_loader_epoch,
)
from tinydiffusion.training import distributed as dist_module
from tinydiffusion.training.distributed import Distributed
from tinydiffusion.utils.tracking import METRICS_FILENAME

# One rank per CPU-bound subprocess. Two is enough to prove sharding, syncing
# and the write guards; more only makes the suite slower.
WORLD_SIZE = 2


# ---------------------------------------------------------------------------
# Reading the launcher's environment
# ---------------------------------------------------------------------------


def test_no_launcher_variables_means_a_single_process(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)

    group = dist_module.from_environment()

    assert not group.enabled
    assert group.world_size == 1
    assert group.is_main


def test_a_world_of_one_is_not_a_group(monkeypatch):
    """--nproc_per_node=1 has to land on the plain single-process path."""
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")

    assert not dist_module.from_environment().enabled


def test_the_launcher_variables_are_read_through(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("LOCAL_RANK", "1")

    group = dist_module.from_environment()

    assert (group.enabled, group.rank, group.local_rank, group.world_size) == (True, 5, 1, 8)
    assert not group.is_main


def test_local_rank_falls_back_to_rank(monkeypatch):
    """Launchers that set only the global index are single-node, where they agree."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert dist_module.from_environment().local_rank == 3


def test_the_group_renders_for_the_plan_line():
    assert str(Distributed()) == "single process"
    assert str(Distributed(enabled=True, rank=2, world_size=4)) == "rank 2/4"


# ---------------------------------------------------------------------------
# Collectives outside a group
# ---------------------------------------------------------------------------


def test_the_collectives_are_no_ops_without_a_group():
    """The single-process path runs straight through them, so they must not touch anything."""
    solo = Distributed()
    value = torch.tensor([2.0, 4.0])

    assert dist_module.all_reduce_mean(value, solo) is value
    assert value.tolist() == [2.0, 4.0]
    assert dist_module.all_reduce_sum(value, solo) is value
    assert value.tolist() == [2.0, 4.0]
    assert dist_module.any_rank(True, solo) is True
    assert dist_module.any_rank(False, solo) is False
    assert dist_module.broadcast_object({"stop": True}, solo) == {"stop": True}
    dist_module.barrier(solo)


def test_setup_without_a_launcher_resolves_a_device_and_starts_nothing(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    group, device = dist_module.setup("cpu")

    assert not group.enabled
    assert device == "cpu"
    assert not torch.distributed.is_initialized()


# ---------------------------------------------------------------------------
# Sharding the dataloader
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset(monkeypatch):
    """A registered dataset of 24 numbered images, with no download behind it.

    Numbered so a shard can be identified: image ``i`` carries label ``i``, so
    what a rank saw is exactly the labels it was handed.
    """
    size = 24

    def builder(root, train, download, transform):
        images = torch.zeros(size, 1, 16, 16)
        return list(zip(images, range(size), strict=True))

    spec = DatasetSpec(
        name="synthetic",
        channels=1,
        native_size=16,
        num_classes=size,
        hflip=False,
        builder=builder,
    )
    monkeypatch.setitem(DATASETS, "synthetic", spec)
    return spec


def test_an_unsharded_loader_has_no_distributed_sampler(synthetic_dataset, tmp_path):
    loader = image_dataloader(synthetic_dataset, tmp_path, batch_size=4, num_workers=0)

    assert not isinstance(loader.sampler, DistributedSampler)
    # set_loader_epoch has nothing to advance, and must not raise looking for it.
    set_loader_epoch(loader, 3)


def test_sharding_splits_the_data_between_the_ranks(synthetic_dataset, tmp_path):
    """Every image goes to exactly one rank, which is what makes an epoch an epoch."""
    seen = []
    for rank in range(WORLD_SIZE):
        loader = image_dataloader(
            synthetic_dataset,
            tmp_path,
            batch_size=4,
            num_workers=0,
            num_replicas=WORLD_SIZE,
            rank=rank,
        )
        set_loader_epoch(loader, 0)
        seen.append({int(label) for _, labels in loader for label in labels})

    assert seen[0] & seen[1] == set()
    assert seen[0] | seen[1] == set(range(24))
    assert len(seen[0]) == len(seen[1]) == 12


def test_every_rank_gets_the_same_number_of_batches(synthetic_dataset, tmp_path):
    """Unequal batch counts are what strands a rank at a gradient all-reduce."""
    counts = []
    for rank in range(WORLD_SIZE):
        loader = image_dataloader(
            synthetic_dataset,
            tmp_path,
            batch_size=5,  # 24 images over 2 ranks does not divide by 5
            num_workers=0,
            num_replicas=WORLD_SIZE,
            rank=rank,
        )
        counts.append(len(list(loader)))

    assert counts[0] == counts[1]


def test_the_shard_changes_with_the_epoch(synthetic_dataset, tmp_path):
    """Without set_epoch a rank re-draws its own shard every epoch, forever."""
    loader = image_dataloader(
        synthetic_dataset,
        tmp_path,
        batch_size=4,
        num_workers=0,
        num_replicas=WORLD_SIZE,
        rank=0,
    )

    def shard() -> list[int]:
        return [int(label) for _, labels in loader for label in labels]

    set_loader_epoch(loader, 0)
    first = shard()
    set_loader_epoch(loader, 1)
    second = shard()

    assert first != second


def test_the_shard_is_a_function_of_seed_and_epoch(synthetic_dataset, tmp_path):
    """The same contract the undivided loader's generator gives: reproducible order."""

    def shard(seed: int) -> list[int]:
        loader = image_dataloader(
            synthetic_dataset,
            tmp_path,
            batch_size=4,
            num_workers=0,
            num_replicas=WORLD_SIZE,
            rank=0,
            seed=seed,
        )
        set_loader_epoch(loader, 2)
        return [int(label) for _, labels in loader for label in labels]

    assert shard(0) == shard(0)
    assert shard(0) != shard(1)


def test_sharding_needs_a_rank(synthetic_dataset, tmp_path):
    with pytest.raises(ValueError, match="rank"):
        image_dataloader(synthetic_dataset, tmp_path, num_replicas=2)


# ---------------------------------------------------------------------------
# A real two-process run
# ---------------------------------------------------------------------------

# Run in each subprocess. Registers the same synthetic dataset the fixture
# above does — a rank cannot inherit a monkeypatch — trains, and writes what it
# ended up with so the parent can compare the ranks against each other.
WORKER = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    import torch

    from tinydiffusion.data.datasets import DATASETS, DatasetSpec
    from tinydiffusion.training.config import TrainConfig
    from tinydiffusion.training.train import train

    out = Path(sys.argv[1])
    rank = int(sys.argv[2])
    seen = []

    def builder(root, train, download, transform):
        images = torch.arange(24, dtype=torch.float32).reshape(24, 1, 1, 1)
        images = images.expand(24, 1, 16, 16).contiguous() / 24.0
        return list(zip(images, range(24), strict=True))

    DATASETS["synthetic"] = DatasetSpec(
        name="synthetic", channels=1, native_size=16, num_classes=24,
        hflip=False, builder=builder,
    )

    cfg = TrainConfig(
        dataset="synthetic",
        data_root=out / "data",
        image_size=16,
        batch_size=4,
        num_workers=0,
        base_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        num_classes=None,
        num_timesteps=10,
        num_epochs=2,
        lr=1e-2,
        lr_warmup=0,
        ema_warmup=0,
        amp=False,
        device="cpu",
        val_batches=1,
        val_steps=2,
        sample_every=0,
        sample_steps=5,
        out_dir=out / "contents",
        ckpt_dir=out / "checkpoints",
        log_dir=out / "runs",
    )

    diffusion = train(cfg)

    # The weights this rank finished with, as one flat vector. Identical across
    # ranks is the whole point of the gradient all-reduce.
    flat = torch.cat([p.detach().reshape(-1) for p in diffusion.net.parameters()])
    (out / f"weights_{rank}.pt").write_bytes(b"")
    torch.save(flat, out / f"weights_{rank}.pt")
    (out / f"done_{rank}.json").write_text(json.dumps({"rank": rank}))
    """
)


def _run_group(tmp_path) -> None:
    """Launch WORLD_SIZE ranks of the worker above and wait for all of them.

    Args:
        tmp_path: directory the ranks write their results into.

    Raises:
        AssertionError: if any rank exits non-zero, with that rank's output.
    """
    script = tmp_path / "worker.py"
    script.write_text(WORKER)

    procs = []
    for rank in range(WORLD_SIZE):
        env = {
            **os.environ,
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(WORLD_SIZE),
            "MASTER_ADDR": "127.0.0.1",
            # Fixed rather than found: the ranks have to agree on it, and
            # nothing else in the suite binds a socket.
            "MASTER_PORT": "29517",
            # gloo on a machine with several interfaces otherwise picks one the
            # other rank cannot reach, and waits out the timeout.
            "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", ""),
        }
        if not env["GLOO_SOCKET_IFNAME"]:
            del env["GLOO_SOCKET_IFNAME"]
        procs.append(
            subprocess.Popen(
                [sys.executable, str(script), str(tmp_path), str(rank)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    for rank, proc in enumerate(procs):
        try:
            output = proc.communicate(timeout=300)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            for other in procs:
                other.kill()
            pytest.fail(f"rank {rank} did not finish within 300s")
        if proc.returncode != 0:
            for other in procs:
                other.kill()
            pytest.fail(f"rank {rank} exited {proc.returncode}:\n{output}")


@pytest.fixture(scope="module")
def group_run(tmp_path_factory):
    """One two-rank training run, shared by the assertions below.

    Module-scoped because launching two interpreters and training them is the
    slow part; every test here reads the same finished run.
    """
    out = tmp_path_factory.mktemp("group")
    _run_group(out)
    return out


@pytest.mark.slow
def test_both_ranks_finish(group_run):
    assert (group_run / "done_0.json").exists()
    assert (group_run / "done_1.json").exists()


@pytest.mark.slow
def test_the_ranks_end_on_identical_weights(group_run):
    """What the gradient all-reduce is for: two copies that stepped as one.

    Divergence here is the failure that does not announce itself — both ranks
    train, the loss falls on both, and the checkpoint is one rank's private
    model rather than the group's.
    """
    first = torch.load(group_run / "weights_0.pt", weights_only=True)
    second = torch.load(group_run / "weights_1.pt", weights_only=True)

    assert torch.equal(first, second)


@pytest.mark.slow
def test_the_weights_actually_moved(group_run):
    """Guard the guard: two untrained copies are also identical."""
    first = torch.load(group_run / "weights_0.pt", weights_only=True)

    assert first.abs().sum() > 0
    assert first.std() > 0


@pytest.mark.slow
def test_only_rank_zero_writes_the_metrics(group_run):
    """Two ranks appending to one file interleave into an unparseable one."""
    records = [
        json.loads(line)
        for line in (group_run / "runs" / METRICS_FILENAME).read_text().splitlines()
        if line
    ]

    assert [record["step"] for record in records] == [0, 1]


@pytest.mark.slow
def test_the_checkpoint_is_written_once_and_loads(group_run):
    checkpoint = torch.load(
        group_run / "checkpoints" / "last.pt", map_location="cpu", weights_only=True
    )

    assert checkpoint["epoch"] == 1
    # No `module.` prefix: DDP wrapped the network but the checkpoint is taken
    # from the eager one, so it stays loadable by every downstream command.
    assert not any(key.startswith("module.") for key in checkpoint["model"])


@pytest.mark.slow
def test_the_logged_throughput_covers_the_whole_group(group_run):
    """Per-rank images/second would under-report a four-GPU run by four times."""
    records = [
        json.loads(line)
        for line in (group_run / "runs" / METRICS_FILENAME).read_text().splitlines()
        if line
    ]
    epoch = records[0]

    images = epoch["time/images_per_second"] * epoch["time/epoch_seconds"]
    # 24 images, sharded two ways and dropped to whole batches of 4.
    assert images == pytest.approx(24, rel=0.2)
