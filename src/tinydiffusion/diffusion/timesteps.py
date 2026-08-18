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

    Args:
        num_timesteps: length of the diffusion schedule.
        history: losses remembered per timestep. Every timestep needs this many
            before the proposal turns on.
        uniform_prob: floor mixed into the proposal, so no timestep can be
            starved of the samples that would update its estimate.

    Raises:
        ValueError: if `history` is not positive, or `uniform_prob` falls
            outside [0, 1).
    """

    def __init__(
        self,
        num_timesteps: int,
        history: int = 10,
        uniform_prob: float = 1e-3,
    ) -> None:
        if history < 1:
            raise ValueError(f"history must be positive, got {history}")
        if not 0.0 <= uniform_prob < 1.0:
            raise ValueError(f"uniform_prob must lie in [0, 1), got {uniform_prob}")

        self.num_timesteps = num_timesteps
        self.history = history
        self.uniform_prob = uniform_prob
        # Kept on the CPU whatever the model trains on: it is touched twice per
        # batch with a handful of scalars, and a device round-trip per update
        # would cost more than the arithmetic.
        self._losses = torch.zeros(num_timesteps, history, dtype=torch.float64)
        self._counts = torch.zeros(num_timesteps, dtype=torch.long)

    @property
    def warm(self) -> bool:
        """Whether every timestep has a full history yet.

        Returns:
            True once the adaptive proposal is in use.
        """
        return bool((self._counts == self.history).all())

    def weights(self) -> torch.Tensor:
        """The sampling probability of each timestep.

        Returns:
            Shape ``(num_timesteps,)`` float64 tensor summing to 1.
        """
        uniform = torch.full((self.num_timesteps,), 1.0 / self.num_timesteps, dtype=torch.float64)
        if not self.warm:
            return uniform

        # The square root of the second moment, i.e. the RMS over the history.
        probs = self._losses.square().mean(dim=1).sqrt()
        total = probs.sum()
        if total <= 0:
            # Every remembered loss is exactly zero, which happens in tests and
            # nowhere else. Fall back rather than divide by it.
            return uniform
        probs = probs / total
        return probs * (1.0 - self.uniform_prob) + self.uniform_prob / self.num_timesteps

    def sample(
        self, batch_size: int, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw timesteps from the current proposal, with matching weights.

        Args:
            batch_size: how many timesteps to draw.
            device: device the returned tensors should live on.

        Returns:
            Tuple of ``(t, weights)``, where ``weights = 1 / (T * p(t))`` — all
            1 while the proposal is still uniform.
        """
        probs = self.weights()
        # With replacement: the batch is a Monte Carlo draw, and forbidding
        # repeats would bias it away from exactly the peaked timesteps the
        # proposal exists to concentrate on.
        index = torch.multinomial(probs, batch_size, replacement=True)
        weights = 1.0 / (self.num_timesteps * probs[index])
        return index.to(device), weights.float().to(device)

    def update(self, t: torch.Tensor, losses: torch.Tensor) -> None:
        """Push each loss onto its timestep's history, oldest out first.

        Args:
            t: shape ``(B,)`` timesteps that were scored.
            losses: shape ``(B,)`` per-sample loss.
        """
        steps = t.detach().to("cpu", torch.long)
        values = losses.detach().to("cpu", torch.float64)
        for step, value in zip(steps.tolist(), values.tolist(), strict=True):
            if self._counts[step] == self.history:
                # Full: shift the window along and drop the oldest entry.
                self._losses[step, :-1] = self._losses[step, 1:].clone()
                self._losses[step, -1] = value
            else:
                self._losses[step, self._counts[step]] = value
                self._counts[step] += 1


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
