"""The DDPM forward/reverse process (Ho et al. 2020, https://arxiv.org/abs/2006.11239)."""

from typing import Literal

import torch
import torch.nn as nn

from tinydiffusion.diffusion.schedules import ddpm_schedules, linear_beta_schedule
from tinydiffusion.utils.modules import eval_mode


class DDPM(nn.Module):
    """Wraps an epsilon-prediction network with the DDPM forward/reverse process.

    Args:
        eps_model: network mapping (x_t, t) -> predicted noise. `t` is passed as
            an integer tensor of shape (B,) with values in [0, num_timesteps-1].
        betas: (beta_start, beta_end) for the linear schedule, or a pre-built
            1-D tensor of length num_timesteps.
        num_timesteps: number of diffusion steps.
        criterion: loss between true and predicted noise. Defaults to MSE.
        clip_denoised: clamp the implied x_0 to [-1, 1] each reverse step.
        variance: "small" uses beta_tilde_t, "large" uses beta_t. DDPM found
            both workable; beta_tilde is the true posterior variance.
    """

    # Registered from `ddpm_schedules` in __init__; declared here so the schedule
    # surface is visible on the class and typed as tensors rather than modules.
    betas: torch.Tensor
    alpha_t: torch.Tensor
    alphabar_t: torch.Tensor
    alphabar_prev: torch.Tensor
    sqrtab: torch.Tensor
    sqrtmab: torch.Tensor
    oneover_sqrta: torch.Tensor
    mab_over_sqrtmab: torch.Tensor
    sqrt_recip_ab: torch.Tensor
    sqrt_recipm1_ab: torch.Tensor
    posterior_var: torch.Tensor
    posterior_logvar: torch.Tensor
    posterior_mean_c0: torch.Tensor
    posterior_mean_ct: torch.Tensor

    def __init__(
        self,
        eps_model: nn.Module,
        betas: tuple[float, float] | torch.Tensor = (1e-4, 0.02),
        num_timesteps: int = 1000,
        criterion: nn.Module | None = None,
        clip_denoised: bool = True,
        variance: Literal["small", "large"] = "small",
    ) -> None:
        super().__init__()
        self.eps_model = eps_model
        self.num_timesteps = num_timesteps
        self.clip_denoised = clip_denoised
        self.variance = variance
        self.criterion = criterion if criterion is not None else nn.MSELoss()

        if isinstance(betas, torch.Tensor):
            beta_t = betas.float()
            if beta_t.numel() != num_timesteps:
                raise ValueError(
                    f"betas has {beta_t.numel()} entries, expected num_timesteps={num_timesteps}"
                )
        else:
            beta_t = linear_beta_schedule(betas[0], betas[1], num_timesteps)

        for name, buffer in ddpm_schedules(beta_t).items():
            self.register_buffer(name, buffer, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample t and eps, build x_t, and score the network's noise estimate.

        Args:
            x: (B, C, H, W) clean images in [-1, 1].

        Returns:
            Scalar training loss.
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        return self.loss_at(x, t)

    def loss_at(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        """Score the noise estimate at explicit timesteps.

        Training draws `t` at random, which makes any single loss value too
        noisy to compare between checkpoints. Evaluation pins both `t` and the
        noise instead, and this is the shared path for both.

        Args:
            x: (B, C, H, W) clean images in [-1, 1].
            t: (B,) integer timesteps in [0, num_timesteps-1].
            noise: the epsilon to add, or None to draw a fresh one.
            model: network to score. Pass `ema.module` to use EMA weights.

        Returns:
            Scalar loss.
        """
        net = model if model is not None else self.eps_model
        eps = torch.randn_like(x) if noise is None else noise
        x_t = self._extract(self.sqrtab, t, x) * x + self._extract(self.sqrtmab, t, x) * eps
        return self.criterion(net(x_t, t), eps)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        size: tuple[int, ...],
        device: torch.device | str,
        model: nn.Module | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Draw samples by running the reverse chain from x_T to x_0.

        Args:
            num_samples: batch size to generate.
            size: shape of one sample, e.g. (1, 28, 28).
            device: device to generate on.
            model: network to sample from. Pass `ema.module` to use EMA weights.
            return_trajectory: also keep every intermediate x_t.

        Returns:
            (num_samples, *size), or (num_timesteps + 1, num_samples, *size) when
            `return_trajectory` is set.
        """
        net = model if model is not None else self.eps_model
        with eval_mode(net):
            x = torch.randn(num_samples, *size, device=device)
            traj: list[torch.Tensor] = [x] if return_trajectory else []

            for i in reversed(range(self.num_timesteps)):
                t = torch.full((num_samples,), i, device=device, dtype=torch.long)
                eps = net(x, t)

                if self.clip_denoised:
                    # Recover x_0, clamp it, then rebuild the posterior mean.
                    x0 = (
                        self._extract(self.sqrt_recip_ab, t, x) * x
                        - self._extract(self.sqrt_recipm1_ab, t, x) * eps
                    ).clamp(-1.0, 1.0)
                    mean = (
                        self._extract(self.posterior_mean_c0, t, x) * x0
                        + self._extract(self.posterior_mean_ct, t, x) * x
                    )
                else:
                    mean = self._extract(self.oneover_sqrta, t, x) * (
                        x - self._extract(self.mab_over_sqrtmab, t, x) * eps
                    )

                if i > 0:
                    var = self.posterior_var if self.variance == "small" else self.betas
                    x = mean + self._extract(var, t, x).sqrt() * torch.randn_like(x)
                else:
                    x = mean  # no noise on the final step

                if return_trajectory:
                    traj.append(x)

        return torch.stack(traj) if return_trajectory else x

    @staticmethod
    def _extract(buf: torch.Tensor, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Gather buf[t] and reshape to broadcast against ref: (B, 1, 1, ...)."""
        return buf.gather(0, t).reshape(-1, *([1] * (ref.dim() - 1)))
