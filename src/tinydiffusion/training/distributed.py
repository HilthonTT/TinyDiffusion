"""Data-parallel training across several GPUs, over ``torch.distributed``.

The unit of parallelism here is the batch, not the model: every rank holds a
complete copy of the network and sees a disjoint shard of each epoch, and
``DistributedDataParallel`` all-reduces the gradients during the backward pass
so all copies step identically. Nothing about the model changes — a run on
eight GPUs writes the same checkpoints, in the same format, as a run on one.

A run becomes distributed because the launcher says so, never because a config
field asked for it. ``torchrun`` sets ``RANK``, ``WORLD_SIZE`` and
``LOCAL_RANK`` in each worker's environment; :func:`from_environment` reads
them, and their absence is what "one process, train normally" looks like. That
keeps the single-GPU path — which is every test, the TUI, and most users — free
of any distributed machinery at all::

    torchrun --nproc_per_node=4 -m tinydiffusion train --config configs/cifar10.toml

The two rules that keep a group from deadlocking are worth stating once, since
every awkward shape in the training loop follows from them:

1. **Collectives are lockstep.** Every rank must reach the same collective the
   same number of times, in the same order. A branch that runs on rank 0 only
   and contains an all-reduce hangs the other seven.
2. **Only rank 0 writes.** Checkpoints, sample grids, ``metrics.jsonl`` and the
   progress bar are all rank-0 work. Eight processes writing one file is a
   corrupt file, and eight progress bars is an unreadable terminal.

:attr:`Distributed.is_main` answers (2), and :func:`any_rank` and
:func:`broadcast_object` are what a rank-0 decision uses to become everyone's
decision without breaking (1).
"""

import datetime as dt
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from tinydiffusion.utils.device import resolve_device

__all__ = [
    "TIMEOUT",
    "Distributed",
    "all_reduce_mean",
    "all_reduce_sum",
    "any_rank",
    "barrier",
    "broadcast_object",
    "from_environment",
    "setup",
    "shutdown",
]


TIMEOUT = dt.timedelta(minutes=60)
"""How long a rank waits at a collective before declaring the group dead.

Longer than the 30-minute default, which is short enough to matter on the first
epoch: rank 0 downloading a dataset while the other ranks sit in the opening
all-reduce is a normal way to start a run, not a hung one.
"""


@dataclass(frozen=True)
class Distributed:
    """Where one process sits in the training group.

    A single-process run holds the disabled instance :func:`from_environment`
    returns when the launcher variables are absent: rank 0 of a world of 1,
    which makes every rank-0 guard in the training loop true and every
    collective below a no-op. The loop therefore reads the same on one GPU as
    on eight, rather than branching on whether it is distributed at all.

    Attributes:
        enabled: whether a process group is running. False for a single
            process, and the switch every collective below checks first.
        rank: this process's index in the whole group, in ``[0, world_size)``.
            Unique across machines.
        local_rank: this process's index on *this machine*, which is what picks
            the GPU. Equal to ``rank`` on a single-node run, and not otherwise.
        world_size: how many processes are training together.
    """

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        """Whether this process is the one that writes and prints.

        Returns:
            True on rank 0, including for a single-process run.
        """
        return self.rank == 0

    def __str__(self) -> str:
        """Render the group for the run's plan line.

        Returns:
            Something like ``"rank 2/4"``, or ``"single process"``.
        """
        if not self.enabled:
            return "single process"
        return f"rank {self.rank}/{self.world_size}"


def from_environment() -> Distributed:
    """Read this process's place in the group from the launcher's variables.

    ``torchrun`` sets ``RANK``, ``WORLD_SIZE`` and ``LOCAL_RANK``; the older
    ``torch.distributed.launch`` and most cluster schedulers set the same
    three. Nothing here starts a process group — see :func:`setup` — so this is
    safe to call in any process, including one that will never be distributed.

    A ``WORLD_SIZE`` of 1 is treated as no group at all. It is what a
    ``--nproc_per_node=1`` launch produces, and there is nothing for a group of
    one to synchronise: skipping the setup keeps that case on exactly the code
    path a plain ``tinydiffusion train`` takes.

    Returns:
        The group this process belongs to, disabled when the environment
        describes a single process.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return Distributed()
    return Distributed(
        enabled=True,
        rank=int(os.environ.get("RANK", "0")),
        # Falls back to RANK for launchers that set only the global index,
        # which is the same number on the single-node runs those are used for.
        local_rank=int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))),
        world_size=world_size,
    )


def setup(requested_device: str | None = None) -> tuple[Distributed, str]:
    """Join the process group, and say which device this rank owns.

    Called in a process the launcher did not mark as distributed, this
    initialises nothing and hands back the device the single-process path would
    have chosen anyway — so the training loop can call it unconditionally.

    Args:
        requested_device: the config's device string, or None to pick
            automatically. Under a group it is honoured only for its *type*: a
            four-process run cannot put all four ranks on ``cuda:0``, so the
            index comes from ``local_rank`` instead.

    Returns:
        The group, and the device string this rank should train on.

    Raises:
        RuntimeError: if the process group cannot be started.
    """
    group = from_environment()
    if not group.enabled:
        # The single-process answer, unchanged: whatever the config asked for,
        # with the CPU fallback resolve_device applies when CUDA was requested
        # and none is visible.
        return group, resolve_device(requested_device)

    # The backend follows the hardware, not the request: NCCL is the only one
    # that moves gradients between GPUs at a useful rate, and it does not touch
    # CPU tensors at all. gloo covers the CPU case, which is what the tests and
    # a machine without a second GPU exercise.
    wants_cpu = requested_device is not None and torch.device(requested_device).type == "cpu"
    cuda = torch.cuda.is_available() and not wants_cpu
    if cuda:
        # Before init_process_group: NCCL binds its communicator to the current
        # device, and every rank leaving that at cuda:0 is the classic way to
        # get a hang on the first all-reduce.
        torch.cuda.set_device(group.local_rank)
        device = f"cuda:{group.local_rank}"
    else:
        device = "cpu"

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if cuda else "gloo",
            timeout=TIMEOUT,
            world_size=group.world_size,
            rank=group.rank,
        )
    return group, device


def shutdown() -> None:
    """Leave the process group, if this process is in one.

    Worth doing rather than leaving to interpreter exit: NCCL warns about a
    communicator torn down without being destroyed, and on some driver versions
    the abandoned communicator holds its GPU memory until the process is
    reaped — which a script that trains and then evaluates will notice.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(group: Distributed) -> None:
    """Wait until every rank has arrived.

    Args:
        group: the training group. A no-op when it is disabled.
    """
    if group.enabled and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: torch.Tensor, group: Distributed) -> torch.Tensor:
    """Average a tensor across every rank, in place.

    Used for the logged metrics rather than for anything the model depends on:
    gradients are averaged by ``DistributedDataParallel`` itself, so this is
    what makes the *reported* loss a mean over the whole global batch instead
    of over whichever shard of it this rank happened to see.

    Args:
        value: tensor to average. Must have the same shape and dtype on every
            rank, and live on this rank's device under NCCL.
        group: the training group.

    Returns:
        The tensor, averaged when there is a group and untouched when there is
        not.
    """
    return all_reduce_sum(value, group).div_(group.world_size) if group.enabled else value


def all_reduce_sum(value: torch.Tensor, group: Distributed) -> torch.Tensor:
    """Total a tensor across every rank, in place.

    The right reduction for a running count — an epoch's per-quartile loss
    totals, say, where each rank has bucketed only its own shard and the sum
    over all of them is the epoch. Averaging those would divide by the world
    size twice over.

    Args:
        value: tensor to total. Must have the same shape and dtype on every
            rank, and live on this rank's device under NCCL.
        group: the training group.

    Returns:
        The tensor, totalled when there is a group and untouched when there is
        not.
    """
    if not group.enabled or not dist.is_initialized():
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def any_rank(flag: bool, group: Distributed, *, device: str = "cpu") -> bool:
    """Whether `flag` is set on *any* rank, as agreed by all of them.

    This is how a per-process event — a Ctrl+C the launcher delivered to some
    ranks a batch earlier than others — becomes one decision the whole group
    acts on at the same batch. Acting on the local flag where it was seen would
    have ranks leave the loop at different iterations, and the first collective
    a straggler reached would then wait out :data:`TIMEOUT` against a rank that
    is never coming.

    Every rank must call this at the same point, which is why the training loop
    asks at a fixed batch cadence rather than whenever the flag changes.

    Args:
        flag: this process's local answer.
        group: the training group.
        device: device to reduce on. Must be this rank's CUDA device under
            NCCL, which cannot reduce a CPU tensor.

    Returns:
        The logical OR over every rank, identical on all of them.
    """
    if not group.enabled or not dist.is_initialized():
        return flag
    # A one-element sum is the portable spelling: ReduceOp.BOR is unsupported
    # on NCCL, and any non-zero total means at least one rank set the flag.
    total = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return bool(total.item() > 0)


def broadcast_object(obj: Any, group: Distributed, *, src: int = 0) -> Any:
    """Send one rank's Python object to every other rank.

    The counterpart to :func:`any_rank`: once the group agrees that *something*
    happened, this is how rank 0 — the only process attached to a terminal, and
    so the only one that can ask the user anything — tells the others what was
    decided.

    Args:
        obj: the object to send. Read on rank `src` and ignored elsewhere, so
            the other ranks may pass None. Must be picklable.
        group: the training group.
        src: the rank whose value wins.

    Returns:
        `src`'s object, on every rank.
    """
    if not group.enabled or not dist.is_initialized():
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]
