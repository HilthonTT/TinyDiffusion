"""How a training step picks the timesteps it scores at.

The default is what DDPM does: draw t uniformly and weight every draw equally.
That is unbiased for L_simple, whose per-timestep terms are all of a similar
size, but the variational bound's are not — the terms near t=0 are orders of
magnitude larger than the rest, so a uniform draw spends almost every sample on
timesteps that contribute nothing and the gradient is dominated by whichever
low-t term happened to land in the batch.

:class:`LossSecondMomentResampler` is Nichol & Dhariwal's fix
(https://arxiv.org/abs/2102.09672, section 3.2): keep a short history of each
timestep's loss, sample in proportion to its RMS, and divide the loss back
through by the sampling probability so the estimator stays unbiased.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch

__all__ = [
    "LossSecondMomentResampler",
    "TimestepSampler",
    "UniformSampler",
    "timestep_sampler",
]


@runtime_checkable
class TimestepSampler(Protocol):
    """A strategy for drawing training timesteps.

    Implementations return importance weights alongside the timesteps: the
    training loss is multiplied by them, so a non-uniform proposal still
    estimates the same expectation.
    """

    def sample(
        self, batch_size: int, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw timesteps and their importance weights.

        Args:
            batch_size: how many timesteps to draw.
            device: device the returned tensors should live on.

        Returns:
            Tuple of ``(t, weights)``, both shape ``(batch_size,)``. `t` is a
            long tensor of timesteps and `weights` a float tensor to multiply
            the per-sample loss by.
        """
        ...

    def update(self, t: torch.Tensor, losses: torch.Tensor) -> None:
        """Record the losses a draw produced.

        Args:
            t: the timesteps that were scored.
            losses: shape ``(B,)`` per-sample loss, *unweighted* — the proposal
                is built from the objective's own scale, not from whatever the
                last proposal did to it.
        """
        ...


class UniformSampler:
    """Draw every timestep with equal probability. What DDPM does.

    Args:
        num_timesteps: length of the diffusion schedule.
    """

    def __init__(self, num_timesteps: int) -> None:
        self.num_timesteps = num_timesteps

    def sample(
        self, batch_size: int, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw uniform timesteps and unit weights.

        Args:
            batch_size: how many timesteps to draw.
            device: device the returned tensors should live on.

        Returns:
            Tuple of ``(t, weights)``, the weights all 1.
        """
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        return t, torch.ones(batch_size, device=device)

    def update(self, t: torch.Tensor, losses: torch.Tensor) -> None:
        """Ignore the losses; a uniform proposal has nothing to adapt.

        Args:
            t: the timesteps that were scored.
            losses: their per-sample losses.
        """


class LossSecondMomentResampler:
    """Sample timesteps in proportion to the RMS of their recent losses.

    The history is kept in memory only: a resumed run re-warms it over its
    first few hundred batches, during which the draw is uniform, which costs
    far less than carrying the state through every checkpoint.

    It is also kept *on the device the losses arrive on*, and every operation
    below is written to keep it there. The training loop goes to some trouble
    to queue a batch's work and move on without waiting for it — see
    :data:`~tinydiffusion.training.train.DRAIN_EVERY` — and a proposal that
    read its own history back to the host would undo that for every run that
    turned this sampler on, once per batch. So there is no ``.item()``, no
    ``.tolist()`` and no Python-level branch on a tensor anywhere between
    :meth:`sample` and :meth:`update`; :attr:`warm` is the one accessor that
    synchronises, and nothing on the hot path calls it.

    Args:
        num_timesteps: length of the diffusion schedule.
        history: losses remembered per timestep. Every timestep needs this many
            before the proposal turns on.
        uniform_prob: floor mixed into the proposal, so no timestep can be
            starved of the samples that would update its estimate.
        gather: how to collect a per-sample tensor from every rank of a
            distributed run, or None to keep the history local to this process.
            See :meth:`update`.

    Raises:
        ValueError: if `history` is not positive, or `uniform_prob` falls
            outside [0, 1).
    """

    def __init__(
        self,
        num_timesteps: int,
        history: int = 10,
        uniform_prob: float = 1e-3,
        gather: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if history < 1:
            raise ValueError(f"history must be positive, got {history}")
        if not 0.0 <= uniform_prob < 1.0:
            raise ValueError(f"uniform_prob must lie in [0, 1), got {uniform_prob}")

        self.num_timesteps = num_timesteps
        self.history = history
        self.uniform_prob = uniform_prob
        self.gather = gather
        # One row per timestep, plus a scratch row at index `num_timesteps`
        # that `_placement` aims the writes it wants to discard at. Built on
        # the CPU and moved to meet the first batch; see `_align`.
        self._losses = torch.zeros(num_timesteps + 1, history, dtype=torch.float64)
        self._counts = torch.zeros(num_timesteps, dtype=torch.long)
        # Where the next write for each timestep goes. The history is a ring
        # rather than a shifted window: same contents, but each update writes
        # one slot instead of rewriting the whole row.
        self._position = torch.zeros(num_timesteps, dtype=torch.long)

    def _align(self, device: torch.device | str) -> None:
        """Move the history to `device` if it is not already there.

        Called from both :meth:`sample` and :meth:`update`, so the history
        follows the model rather than having to be told where it went.

        Args:
            device: where the next draw or update is happening.
        """
        # Unconditional because `Tensor.to` already is one: it hands back the
        # same tensor when there is nothing to change. Guarding it with a
        # comparison would mean resolving `cuda` against `cuda:0` by hand, and
        # getting that wrong copies three buffers on every batch.
        self._losses = self._losses.to(device)
        self._counts = self._counts.to(device)
        self._position = self._position.to(device)

    @property
    def warm(self) -> bool:
        """Whether every timestep has a full history yet.

        Reading this synchronises with the device, which is why
        :meth:`weights` works out the same fact as a tensor instead. It is here
        for tests and for anything reporting on a run, not for the training
        loop.

        Returns:
            True once the adaptive proposal is in use.
        """
        return bool((self._counts == self.history).all())

    def weights(self) -> torch.Tensor:
        """The sampling probability of each timestep.

        Returns:
            Shape ``(num_timesteps,)`` float64 tensor summing to 1, on the
            device the history currently lives on.
        """
        rows = self._losses[: self.num_timesteps]
        uniform = torch.full(
            (self.num_timesteps,),
            1.0 / self.num_timesteps,
            dtype=rows.dtype,
            device=rows.device,
        )
        # The square root of the second moment, i.e. the RMS over the history.
        rms = rows.square().mean(dim=1).sqrt()
        total = rms.sum()
        # Both conditions stay tensors and the choice is made by `torch.where`,
        # because `if not self.warm` would be a host read on the hot path. The
        # second one covers a history of exactly zero everywhere, which happens
        # in tests and nowhere else; the clamp only keeps the arithmetic finite
        # in the branch that is then thrown away.
        usable = (self._counts == self.history).all() & (total > 0)
        adaptive = rms / total.clamp(min=torch.finfo(rows.dtype).tiny)
        adaptive = adaptive * (1.0 - self.uniform_prob) + self.uniform_prob / self.num_timesteps
        return torch.where(usable, adaptive, uniform)

    def sample(
        self, batch_size: int, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw timesteps from the current proposal, with matching weights.

        The draw happens on `device` rather than on the host, so a CUDA run
        consumes its CUDA generator here where it used to consume the CPU one.
        That is a different random stream for the same seed — the same change
        :class:`UniformSampler` already embodies, since its ``randint`` has
        always been device-side.

        Args:
            batch_size: how many timesteps to draw.
            device: device the returned tensors should live on.

        Returns:
            Tuple of ``(t, weights)``, where ``weights = 1 / (T * p(t))`` — all
            1 while the proposal is still uniform.
        """
        self._align(device)
        probs = self.weights()
        # With replacement: the batch is a Monte Carlo draw, and forbidding
        # repeats would bias it away from exactly the peaked timesteps the
        # proposal exists to concentrate on.
        index = torch.multinomial(probs, batch_size, replacement=True)
        weights = 1.0 / (self.num_timesteps * probs[index])
        return index, weights.float()

    def _placement(self, steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Work out where each of a batch's losses belongs in the ring.

        A batch draws with replacement, so the same timestep can appear several
        times, and the sequential loop this replaces gave each occurrence the
        next slot along. Reproducing that without a loop needs each element's
        rank among the duplicates that share its timestep, which is what the
        sort below computes: sort by timestep, and an element's rank is its
        distance from the start of its run.

        A timestep drawn more than `history` times in one batch would wrap the
        ring inside a single update and collide with itself. Those writes are
        aimed at the scratch row instead — with only the last `history`
        occurrences kept, which is exactly what the sequential loop would have
        left behind — so every surviving ``(row, slot)`` pair is distinct and
        the write order does not matter.

        Args:
            steps: shape ``(B,)`` timesteps, already on the history's device.

        Returns:
            Tuple of ``(row, slot, occurrences)``: the row of `_losses` to
            write each loss to, the slot within that row, and how many times
            each timestep appeared in this batch.
        """
        count = steps.shape[0]
        device = steps.device
        index = torch.arange(count, device=device)

        order = torch.argsort(steps, stable=True)
        ordered = steps[order]
        opens = torch.ones(count, dtype=torch.bool, device=device)
        opens[1:] = ordered[1:] != ordered[:-1]
        # Running maximum of "index of the last run that opened", which for a
        # sorted sequence is the first index holding this element's timestep.
        run_start = torch.cummax(torch.where(opens, index, torch.zeros_like(index)), dim=0).values
        rank = torch.empty_like(index)
        rank[order] = index - run_start

        occurrences = torch.zeros(self.num_timesteps, dtype=torch.long, device=device)
        occurrences.index_add_(0, steps, torch.ones_like(steps))
        # Keep the last `history` occurrences of each timestep and discard the
        # rest, which is what a ring of that size would hold anyway.
        keep = occurrences[steps] - rank <= self.history
        row = torch.where(keep, steps, torch.full_like(steps, self.num_timesteps))
        slot = (self._position[steps] + rank) % self.history
        return row, slot, occurrences

    def update(self, t: torch.Tensor, losses: torch.Tensor) -> None:
        """Push each loss onto its timestep's history, oldest out first.

        Entirely device-side: no value crosses to the host, so this costs the
        training loop a few small kernels rather than a synchronisation per
        batch.

        Under a distributed run each rank sees only its own shard of the global
        batch, and would otherwise warm its own private proposal on ``1 /
        world_size`` of the data. A `gather` collects the whole group's
        timesteps and losses first, so every rank builds the same history from
        the same evidence — and, because that is a collective, it runs on every
        call and on every rank whether or not this batch had anything to add.

        Args:
            t: shape ``(B,)`` timesteps that were scored.
            losses: shape ``(B,)`` per-sample loss.
        """
        steps = t.detach().reshape(-1).to(torch.long)
        values = losses.detach().reshape(-1)
        if self.gather is not None:
            # Before `_align`, and unconditionally: a rank that skipped this
            # because its own batch was empty would hang the rest of the group.
            steps = self.gather(steps)
            values = self.gather(values)
        if steps.numel() == 0:
            return

        self._align(steps.device)
        row, slot, occurrences = self._placement(steps)
        self._losses.index_put_((row, slot), values.to(self._losses.dtype))
        self._position.add_(occurrences).remainder_(self.history)
        self._counts.add_(occurrences).clamp_(max=self.history)


def timestep_sampler(name: str, num_timesteps: int) -> TimestepSampler:
    """Build the timestep sampler a config names.

    Args:
        name: ``"uniform"`` or ``"loss_second_moment"``.
        num_timesteps: length of the diffusion schedule.

    Returns:
        The sampler.

    Raises:
        ValueError: if `name` matches neither.
    """
    match name:
        case "uniform":
            return UniformSampler(num_timesteps)
        case "loss_second_moment":
            return LossSecondMomentResampler(num_timesteps)
        case _:
            raise ValueError(
                f"unknown timestep_sampler {name!r}, expected 'uniform' or 'loss_second_moment'"
            )
