from typing import Literal, Tuple, Optional

import torch
import torch.nn as nn

from schedules import linear_beta_schedule, ddpm_schedules

class DDPM(nn.Module):
    """Wraps an epsilon-prediction network with the DDPM forward/reverse process.
 
    Args:
        eps_model: network mapping (x_t, t) -> predicted noise. `t` is passed as
            an integer tensor of shape (B,) with values in [0, n_T-1].
        betas: (beta_start, beta_end) for the linear schedule, or a pre-built
            1-D tensor of length n_T.
        n_T: number of diffusion steps.
        criterion: loss between true and predicted noise.
        clip_denoised: clamp the implied x_0 to [-1, 1] each reverse step.
        variance: "small" uses beta_tilde_t, "large" uses beta_t. DDPM found
            both workable; beta_tilde is the true posterior variance.
    """

    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float] | torch.Tensor = (1e-4, 0.02),
        n_T: int = 1000,
        criterion: Optional[nn.Module] = None,
        clip_denoised: bool = True,
        variance: Literal["small", "large"] = "small",
    ) -> None:
        super().__init__()
        self.eps_model = eps_model
        self.n_T = n_T
        self.clip_denoised = clip_denoised
        self.variance = variance
        self.criterion = criterion if criterion is not None else nn.MSELoss()

        if isinstance(betas, torch.Tensor):
            beta_t = betas.float()
            if beta_t.numel() != n_T:
                raise ValueError(f"betas has {beta_t.numel()} entries, expected n_T={n_T}")
        else:
            beta_t = linear_beta_schedule(betas[0], betas[1], n_T)
 
        for k, v in ddpm_schedules(beta_t).items():
            self.register_buffer(k, v, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample t and eps, build x_t, and score the network's noise estimate."""
        t = torch.randint(0, self.n_T, (x.shape[0],), device=x.device)
        eps = torch.randn_like(x)
        x_t = self._extract(self.sqrtab, t, x) * x + self._extract(self.sqrtmab, t, x) * eps
        return self.criterion(self.eps_model(x_t, t), eps)

    @torch.no_grad()
    def sample(
        self,
        n_sample: int,
        size: Tuple[int, ...],
        device: torch.device | str,
        model: Optional[nn.Module] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Draw samples by running the reverse chain from x_T to x_0.
 
        Pass `model=ema.module` to sample from the EMA weights.
        """
        net = model if model is not None else self.eps_model
        was_training = net.training
        net.eval()
        try:
            x = torch.randn(n_sample, *size, device=device)
            traj = [x] if return_trajectory else None
 
            for i in reversed(range(self.n_T)):
                t = torch.full((n_sample,), i, device=device, dtype=torch.long)
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
        finally:
            net.train(was_training)
 
        return torch.stack(traj) if return_trajectory else x
 
    @staticmethod
    def _extract(buf: torch.Tensor, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Gather buf[t] and reshape to broadcast against ref: (B, 1, 1, ...)."""
        return buf.gather(0, t).reshape(-1, *([1] * (ref.dim() - 1)))
