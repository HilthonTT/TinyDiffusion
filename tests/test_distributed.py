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

import contextlib
import json
import os
import socket
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

# How long a rank gets before the parent gives up on it. Generous for the real
# training run, which builds a model and trains two epochs per rank.
RUN_TIMEOUT = 300

# The rendezvous probe below only has to reach an all-reduce, so it gets far
# less rope. This is the number that decides how long a machine that cannot
# form a group at all takes to say so.
PROBE_TIMEOUT = 20


def _free_port() -> int:
    """A TCP port that is free right now.

    A hard-coded port is flaky by construction: a previous rank's socket in
    TIME_WAIT, a second copy of the suite, or anything else on the machine that
    happened to want it are all "address already in use" on rank 0. Asking the
    kernel for one narrows the race to the microseconds between closing this
    socket and a rank binding it.

    Returns:
        A port number nothing was listening on a moment ago.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _loopback_interfaces() -> list[str]:
    """Names ``GLOO_SOCKET_IFNAME`` might accept for this platform's loopback.

    Every rank here runs on one machine, so loopback is both sufficient and the
    only interface guaranteed to be able to reach the others. The Windows name
    is localised, which is why this is a list of candidates to try rather than
    an answer.

    Returns:
        Interface names, best guess first. Possibly empty.
    """
    if sys.platform == "win32":
        return ["Loopback Pseudo-Interface 1"]
    if sys.platform == "darwin":
        return ["lo0"]
    return ["lo"]


# Run by `_working_environment` to find out whether a group can be formed at
# all, before the expensive one is launched. Deliberately the smallest thing
# that exercises both halves: the store rendezvous, and gloo's own transport.
PROBE = textwrap.dedent(
    """
    import os, sys
    import torch, torch.distributed as dist

    dist.init_process_group(backend="gloo", world_size=int(os.environ["WORLD_SIZE"]),
                            rank=int(sys.argv[1]))
    total = torch.tensor([1.0])
    dist.all_reduce(total)
    assert total.item() == int(os.environ["WORLD_SIZE"]), total
    dist.destroy_process_group()
    """
)


def _launch(script, args_per_rank, env, timeout):
    """Run one subprocess per rank and collect their exit codes and output.

    Args:
        script: path to the worker script.
        args_per_rank: extra argv for each rank, one list per rank.
        env: environment shared by every rank, minus the rank-specific parts.
        timeout: seconds to wait for each rank before killing the lot.

    Returns:
        List of ``(returncode, output)`` per rank. A killed rank reports None.
    """
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), *args_per_rank[rank]],
            env={**env, "RANK": str(rank), "LOCAL_RANK": str(rank)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in range(WORLD_SIZE)
    ]

    results: list[tuple[int | None, str]] = []
    for proc in procs:
        try:
            output = proc.communicate(timeout=timeout)[0]
            results.append((proc.returncode, output))
        except subprocess.TimeoutExpired:
            # One rank stuck means the rest are stuck waiting on it, so the
            # whole group goes rather than each timing out in turn.
            for other in procs:
                other.kill()
            results.append((None, "timed out"))

    # Reap whatever the kills left behind, so no rank outlives the test and
    # holds the rendezvous port.
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.communicate(timeout=5)
    return results


def _base_environment() -> dict[str, str]:
    """The environment every rank shares, bar the interface setting.

    Returns:
        A copy of this process's environment with the launcher variables and a
        freshly found rendezvous port added.
    """
    return {
        **os.environ,
        "WORLD_SIZE": str(WORLD_SIZE),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(_free_port()),
    }


@pytest.fixture(scope="session")
def gloo_environment(tmp_path_factory):
    """An environment two ranks can actually form a process group in.

    Left to itself, gloo resolves this machine's hostname and binds the first
    address that comes back. On a developer machine that is quite often a VPN
    or hypervisor adapter — a Tailscale or NordLynx address, a Hyper-V vSwitch
    — which both ranks bind and neither can reach the other over. The symptom
    is not an error: it is the whole group sitting in `init_process_group`
    until something kills it, which is a 300-second hang per test that wanted a
    group.

    Naming a loopback interface fixes it, but the name is platform-specific and
    localised on Windows, so this probes instead of assuming: gloo's own
    default first, since that is what CI uses and what works on an ordinary
    Linux runner, then each loopback candidate. A machine where none of them
    works skips rather than hangs, and says which knob to reach for.

    Returns:
        The environment to launch ranks with, port and all.

    Raises:
        pytest.skip.Exception: if no candidate can form a group here.
    """
    script = tmp_path_factory.mktemp("probe") / "probe.py"
    script.write_text(PROBE)

    # An operator who has set this has already made the choice; do not probe
    # around it, and do not silently override it.
    if os.environ.get("GLOO_SOCKET_IFNAME"):
        return _base_environment()

    tried = []
    for interface in [None, *_loopback_interfaces()]:
        env = _base_environment()
        if interface is None:
            env.pop("GLOO_SOCKET_IFNAME", None)
        else:
            env["GLOO_SOCKET_IFNAME"] = interface
        results = _launch(script, [[str(r)] for r in range(WORLD_SIZE)], env, PROBE_TIMEOUT)
        if all(code == 0 for code, _ in results):
            return env
        tried.append(f"{interface or 'gloo default'}: {results[0][1].strip().splitlines()[-1:]}")

    pytest.skip(
        "no two-rank gloo group could be formed on this machine; set GLOO_SOCKET_IFNAME "
        "to an interface the ranks can reach each other over. Tried — " + "; ".join(tried)
    )


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


def _run_group(tmp_path, environment) -> None:
    """Launch WORLD_SIZE ranks of the worker above and wait for all of them.

    Args:
        tmp_path: directory the ranks write their results into.
        environment: the probed environment from `gloo_environment`. A fresh
            port is taken for this run rather than reusing the probe's, which
            has ranks in TIME_WAIT on it.

    Raises:
        AssertionError: if any rank exits non-zero or runs out of time, with
            that rank's output.
    """
    script = tmp_path / "worker.py"
    script.write_text(WORKER)

    env = {**environment, "MASTER_PORT": str(_free_port())}
    args = [[str(tmp_path), str(rank)] for rank in range(WORLD_SIZE)]
    results = _launch(script, args, env, RUN_TIMEOUT)

    for rank, (code, output) in enumerate(results):
        if code is None:
            pytest.fail(f"rank {rank} did not finish within {RUN_TIMEOUT}s")
        if code != 0:
            pytest.fail(f"rank {rank} exited {code}:\n{output}")


@pytest.fixture(scope="module")
def group_run(tmp_path_factory, gloo_environment):
    """One two-rank training run, shared by the assertions below.

    Module-scoped because launching two interpreters and training them is the
    slow part; every test here reads the same finished run.
    """
    out = tmp_path_factory.mktemp("group")
    _run_group(out, gloo_environment)
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


# ---------------------------------------------------------------------------
# Sharing the adaptive timestep proposal
# ---------------------------------------------------------------------------

# Each rank folds in a batch only it can see, then writes the history it ended
# up with. Without the gather each rank remembers its own half; with it, both
# remember all of it and agree on the proposal.
RESAMPLER_WORKER = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    import torch

    from tinydiffusion.diffusion.timesteps import LossSecondMomentResampler
    from tinydiffusion.training.distributed import all_gather_cat, setup, shutdown

    out = Path(sys.argv[1])
    rank = int(sys.argv[2])
    shared = sys.argv[3] == "shared"

    group, _ = setup("cpu")
    sampler = LossSecondMomentResampler(4, history=2)
    if shared:
        sampler.gather = lambda value: all_gather_cat(value, group)

    # Rank 0 only ever sees timesteps 0 and 1; rank 1 only 2 and 3. A rank that
    # keeps its own history can never warm up, because half the schedule never
    # reaches it.
    steps = torch.tensor([0, 1] if rank == 0 else [2, 3])
    for _ in range(2):
        sampler.update(steps, torch.full((2,), 1.0 + rank))

    torch.save(
        {"counts": sampler._counts.clone(), "weights": sampler.weights().clone(),
         "warm": sampler.warm},
        out / f"resampler_{rank}.pt",
    )
    shutdown()
    """
)


@pytest.fixture(scope="module")
def resampler_runs(tmp_path_factory, gloo_environment):
    """Both variants of the two-rank resampler run, shared by the tests below.

    Returns:
        Mapping from ``"shared"``/``"local"`` to that run's output directory.
    """
    runs = {}
    for mode in ("shared", "local"):
        out = tmp_path_factory.mktemp(f"resampler-{mode}")
        script = out / "worker.py"
        script.write_text(RESAMPLER_WORKER)
        env = {**gloo_environment, "MASTER_PORT": str(_free_port())}
        args = [[str(out), str(rank), mode] for rank in range(WORLD_SIZE)]
        results = _launch(script, args, env, RUN_TIMEOUT)
        for rank, (code, output) in enumerate(results):
            assert code == 0, f"{mode} rank {rank} exited {code}:\n{output}"
        runs[mode] = out
    return runs


@pytest.mark.slow
def test_a_shared_proposal_sees_every_ranks_timesteps(resampler_runs):
    """The point of the gather: one history built from the whole global batch."""
    first = torch.load(resampler_runs["shared"] / "resampler_0.pt", weights_only=True)
    second = torch.load(resampler_runs["shared"] / "resampler_1.pt", weights_only=True)

    # Every timestep has its full history, though no rank drew more than half.
    assert first["counts"].tolist() == [2, 2, 2, 2]
    assert first["warm"]
    # And both ranks agree, so they are drawing from the same proposal.
    assert torch.equal(first["counts"], second["counts"])
    assert torch.equal(first["weights"], second["weights"])


@pytest.mark.slow
def test_without_the_gather_each_rank_keeps_half_the_history(resampler_runs):
    """Guard the guard: the test above passes trivially if sharding is broken."""
    first = torch.load(resampler_runs["local"] / "resampler_0.pt", weights_only=True)
    second = torch.load(resampler_runs["local"] / "resampler_1.pt", weights_only=True)

    assert first["counts"].tolist() == [2, 2, 0, 0]
    assert second["counts"].tolist() == [0, 0, 2, 2]
    # Neither ever warms, so both fall back to a uniform draw forever.
    assert not first["warm"] and not second["warm"]


def test_gathering_outside_a_group_hands_the_tensor_straight_back():
    value = torch.tensor([1.0, 2.0])

    assert dist_module.all_gather_cat(value, Distributed()) is value
