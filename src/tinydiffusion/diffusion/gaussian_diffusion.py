"""The generalised Gaussian diffusion process.

:class:`~tinydiffusion.diffusion.ddpm.DDPM` fixes three choices that the papers
treat as free: the network predicts epsilon, the reverse variance is the fixed
posterior beta-tilde, and the loss is plain MSE. That combination is the DDPM
baseline (Ho et al. 2020) and is all most runs need.

This module makes those three choices explicit as :class:`ModelMeanType`,
:class:`ModelVarType` and :class:`LossType`, and adds the variational bound
they are measured against. Learning the reverse variance and training on the
hybrid objective is the main contribution of Nichol & Dhariwal 2021
(https://arxiv.org/abs/2102.09672); it is what lets a model sample well in far
fewer steps.

Adapted from openai/improved-diffusion, rewritten to be torch-native, to keep
its schedule in registered buffers, and to take integer timesteps.
"""

from enum import StrEnum
from typing import Self

import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddpm import DDPM, LossTerms
from tinydiffusion.diffusion.losses import (
    discretized_gaussian_log_likelihood,
    mean_flat,
    normal_kl,
)
from tinydiffusion.diffusion.schedules import ddpm_schedules, linear_beta_schedule
from tinydiffusion.utils.modules import eval_mode

__all__ = [
    "Diffusion",
    "GaussianDiffusion",
    "LossType",
    "ModelMeanType",
    "ModelVarType",
]

_LOG_2 = 0.6931471805599453
"""Nats per bit, for reporting the bound in bits-per-dimension."""


class ModelMeanType(StrEnum):
    """What the network's output is interpreted as.

    Attributes:
        EPSILON: the noise added to x_0. The DDPM default, and the easiest to
            train because the target has unit variance at every timestep.
        START_X: the clean image x_0 directly.
        PREVIOUS_X: the previous latent x_{t-1}. Present for completeness;
            it trains poorly and no current model uses it.
    """

    EPSILON = "epsilon"
    START_X = "start_x"
    PREVIOUS_X = "previous_x"


class ModelVarType(StrEnum):
    """How the reverse-process variance is obtained.

    Attributes:
        FIXED_SMALL: the true posterior variance beta-tilde. Optimal when x_0
            is known; slightly over-confident when it is not.
        FIXED_LARGE: beta_t, the upper bound. DDPM found both usable.
        LEARNED: the network emits the log-variance directly. Unstable, since
            nothing bounds it.
        LEARNED_RANGE: the network emits a value in [-1, 1] that interpolates
            between the two fixed choices. This is the Nichol & Dhariwal
            formulation and the one to use.
    """

    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED = "learned"
    LEARNED_RANGE = "learned_range"

    @property
    def is_learned(self) -> bool:
        """Whether this option needs the network to emit 2C channels.

        Returns:
            True for the two learned variants.
        """
        return self in (ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE)


class LossType(StrEnum):
    """Which objective to train against.

    Attributes:
        MSE: the simplified L_simple objective from DDPM.
        RESCALED_MSE: L_simple plus a down-weighted variational term, used to
            train a learned variance without letting it disturb the mean. This
            is the hybrid objective.
        KL: the full variational bound, in bits per dimension.
        RESCALED_KL: the bound rescaled by num_timesteps, which makes its
            gradient magnitude comparable to MSE.
    """

    MSE = "mse"
    RESCALED_MSE = "rescaled_mse"
    KL = "kl"
    RESCALED_KL = "rescaled_kl"

    @property
    def is_variational(self) -> bool:
        """Whether this objective is the bound itself rather than MSE.

        Returns:
            True for the two KL variants.
        """
        return self in (LossType.KL, LossType.RESCALED_KL)


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion with configurable parameterisation and objective.

    The buffer set is shared with :class:`~tinydiffusion.diffusion.ddpm.DDPM`
    via :func:`~tinydiffusion.diffusion.schedules.ddpm_schedules`, so both
    classes work with :func:`~tinydiffusion.diffusion.ddim.ddim_sample` and
    with the same checkpoints.

    Args:
        model: network mapping ``(x_t, t)`` to its prediction. `t` is an
            integer tensor of shape ``(B,)`` in ``[0, num_timesteps-1]``. When
            `model_var_type` is learned the network must emit ``2 * C``
            channels: the mean prediction, then the variance parameters.
        betas: ``(beta_start, beta_end)`` for a linear schedule, or a
            pre-built 1-D tensor of length `num_timesteps`.
        num_timesteps: number of diffusion steps.
        model_mean_type: how to interpret the network's mean output.
        model_var_type: how to obtain the reverse variance.
        loss_type: which objective to train against.
        clip_denoised: clamp the implied x_0 to [-1, 1] while sampling.

    Raises:
        ValueError: if `betas` has the wrong length, or the loss and variance
            settings cannot be trained together.
    """

    betas: torch.Tensor
    alphabar_t: torch.Tensor
    alphabar_prev: torch.Tensor
    sqrtab: torch.Tensor
    sqrtmab: torch.Tensor
    sqrt_recip_ab: torch.Tensor
    sqrt_recipm1_ab: torch.Tensor
    posterior_var: torch.Tensor
    posterior_mean_c0: torch.Tensor
    posterior_mean_ct: torch.Tensor
    posterior_logvar_clipped: torch.Tensor
    log_betas: torch.Tensor
    fixed_large_logvar: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        betas: tuple[float, float] | torch.Tensor = (1e-4, 0.02),
        num_timesteps: int = 1000,
        model_mean_type: ModelMeanType = ModelMeanType.EPSILON,
        model_var_type: ModelVarType = ModelVarType.FIXED_SMALL,
        loss_type: LossType = LossType.MSE,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.clip_denoised = clip_denoised

        if loss_type is LossType.RESCALED_MSE and not model_var_type.is_learned:
            raise ValueError(
                "RESCALED_MSE adds a variational term to train a learned variance, "
                f"but model_var_type is {model_var_type}; use LossType.MSE instead"
            )

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

        # The shared schedule clamps the posterior log-variance at 1e-20 before
        # taking a log, which is fine for sampling because t=0 draws no noise.
        # The bound's decoder term uses it as a log-scale, though, where
        # exp(23) makes the 1/255 discretisation window span ~4e7 sigma and the
        # CDF difference underflows. Substituting t=1's value keeps it finite.
        posterior_var = self.posterior_var
        clipped = torch.cat([posterior_var[1:2], posterior_var[1:]])
        self.register_buffer("posterior_logvar_clipped", clipped.log(), persistent=False)
        self.register_buffer("log_betas", beta_t.log(), persistent=False)
        # FIXED_LARGE uses beta_t everywhere except t=0, where beta_0 would give
        # a worse decoder likelihood than the posterior variance at t=1.
        fixed_large = torch.cat([posterior_var[1:2], beta_t[1:]])
        self.register_buffer("fixed_large_logvar", fixed_large.log(), persistent=False)

    @property
    def net(self) -> nn.Module:
        """The wrapped network, under the name every process shares.

        Returns:
            The network passed to the constructor.
        """
        return self.model

    @property
    def out_channels_per_image_channel(self) -> int:
        """How many output channels the network owes per input channel.

        Returns:
            2 when the variance is learned, 1 otherwise.
        """
        return 2 if self.model_var_type.is_learned else 1

    # ------------------------------------------------------------------
    # forward process q
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw x_t from q(x_t | x_0), the closed-form forward process.

        Args:
            x_start: ``(B, C, H, W)`` clean images in [-1, 1].
            t: ``(B,)`` integer timesteps.
            noise: the epsilon to add, or None to draw a fresh one.

        Returns:
            The noised images, same shape as `x_start`.
        """
        eps = torch.randn_like(x_start) if noise is None else noise
        return _expand(self.sqrtab, t, x_start) * x_start + _expand(self.sqrtmab, t, x_start) * eps

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean and variance of q(x_{t-1} | x_t, x_0).

        This is the target the reverse process is fitted against: the true
        posterior, available in closed form only because x_0 is known during
        training.

        Args:
            x_start: the clean images.
            x_t: the noised images at timestep `t`.
            t: ``(B,)`` integer timesteps.

        Returns:
            Tuple of ``(mean, variance, clipped_log_variance)``.
        """
        mean = (
            _expand(self.posterior_mean_c0, t, x_t) * x_start
            + _expand(self.posterior_mean_ct, t, x_t) * x_t
        )
        var = _expand(self.posterior_var, t, x_t)
        log_var = _expand(self.posterior_logvar_clipped, t, x_t)
        return mean, var, log_var

    # ------------------------------------------------------------------
    # reverse process p
    # ------------------------------------------------------------------

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        model: nn.Module | None = None,
        model_output: torch.Tensor | None = None,
        clip_denoised: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate p(x_{t-1} | x_t) and the implied x_0.

        Args:
            x: ``(B, C, H, W)`` latents at timestep `t`.
            t: ``(B,)`` integer timesteps.
            model: network to evaluate. Defaults to the wrapped model; pass
                ``ema.module`` to use EMA weights.
            model_output: a precomputed network output, used by the training
                loss to avoid a second forward pass.
            clip_denoised: override the instance default.

        Returns:
            Tuple of ``(mean, variance, log_variance, pred_xstart)``.
        """
        clip = self.clip_denoised if clip_denoised is None else clip_denoised
        if model_output is None:
            net = model if model is not None else self.model
            model_output = net(x, t)

        channels = x.shape[1]
        if self.model_var_type.is_learned:
            model_output, var_values = torch.split(model_output, channels, dim=1)
            if self.model_var_type is ModelVarType.LEARNED:
                log_var = var_values
            else:
                # var_values in [-1, 1] interpolates in log space between the
                # two fixed choices, which bounds what the network can ask for.
                min_log = _expand(self.posterior_logvar_clipped, t, x)
                max_log = _expand(self.log_betas, t, x)
                frac = (var_values + 1.0) / 2.0
                log_var = frac * max_log + (1.0 - frac) * min_log
            var = log_var.exp()
        elif self.model_var_type is ModelVarType.FIXED_LARGE:
            log_var = _expand(self.fixed_large_logvar, t, x)
            var = log_var.exp()
        else:
            var = _expand(self.posterior_var, t, x)
            log_var = _expand(self.posterior_logvar_clipped, t, x)

        if self.model_mean_type is ModelMeanType.PREVIOUS_X:
            pred_xstart = self._predict_xstart_from_xprev(x, t, model_output)
            pred_xstart = pred_xstart.clamp(-1.0, 1.0) if clip else pred_xstart
            mean = model_output
        else:
            if self.model_mean_type is ModelMeanType.START_X:
                pred_xstart = model_output
            else:
                pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
            pred_xstart = pred_xstart.clamp(-1.0, 1.0) if clip else pred_xstart
            mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)

        # The decoder term needs a per-pixel log-scale, and the learned
        # branches already produce one; broadcast the fixed ones to match so
        # both paths hand back the same shape.
        return mean, var.expand_as(x), log_var.expand_as(x), pred_xstart

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """Invert the forward process to recover x_0 from a noise estimate.

        Args:
            x_t: latents at timestep `t`.
            t: ``(B,)`` integer timesteps.
            eps: the predicted noise.

        Returns:
            The implied x_0.
        """
        return (
            _expand(self.sqrt_recip_ab, t, x_t) * x_t - _expand(self.sqrt_recipm1_ab, t, x_t) * eps
        )

    def _predict_xstart_from_xprev(
        self, x_t: torch.Tensor, t: torch.Tensor, xprev: torch.Tensor
    ) -> torch.Tensor:
        """Recover x_0 from a prediction of x_{t-1}.

        Args:
            x_t: latents at timestep `t`.
            t: ``(B,)`` integer timesteps.
            xprev: the predicted x_{t-1}.

        Returns:
            The implied x_0.
        """
        coef1 = _expand(self.posterior_mean_c0, t, x_t)
        coef2 = _expand(self.posterior_mean_ct, t, x_t)
        return (xprev - coef2 * x_t) / coef1

    def predict_eps_from_xstart(
        self, x_t: torch.Tensor, t: torch.Tensor, pred_xstart: torch.Tensor
    ) -> torch.Tensor:
        """Recover the noise implied by an x_0 prediction.

        Needed after clipping x_0, so the sampler's direction term stays
        consistent with the clipped value.

        Args:
            x_t: latents at timestep `t`.
            t: ``(B,)`` integer timesteps.
            pred_xstart: the predicted x_0.

        Returns:
            The implied epsilon.
        """
        return (_expand(self.sqrt_recip_ab, t, x_t) * x_t - pred_xstart) / _expand(
            self.sqrt_recipm1_ab, t, x_t
        )

    # ------------------------------------------------------------------
    # objectives
    # ------------------------------------------------------------------

    def vb_terms_bpd(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        model: nn.Module | None = None,
        model_output: torch.Tensor | None = None,
        clip_denoised: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One term of the variational bound, in bits per dimension.

        For t > 0 this is KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t)). At t=0 the
        KL is not the right quantity — the target is a point mass — so the
        term becomes the discretised decoder likelihood instead.

        Args:
            x_start: the clean images.
            x_t: the noised images at timestep `t`.
            t: ``(B,)`` integer timesteps.
            model: network to evaluate.
            model_output: precomputed network output.
            clip_denoised: clamp the implied x_0. Left off for the bound,
                where clipping would make the number no longer a bound.

        Returns:
            Tuple of ``(per_sample_bits, pred_xstart)``.
        """
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_start, x_t, t)
        mean, _, log_var, pred_xstart = self.p_mean_variance(
            x_t, t, model=model, model_output=model_output, clip_denoised=clip_denoised
        )

        kl = mean_flat(normal_kl(true_mean, true_log_var, mean, log_var)) / _LOG_2
        decoder_nll = (
            -mean_flat(
                discretized_gaussian_log_likelihood(x_start, means=mean, log_scales=0.5 * log_var)
            )
            / _LOG_2
        )
        return torch.where(t == 0, decoder_nll, kl), pred_xstart

    def training_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-sample losses for one batch of timesteps.

        Args:
            x_start: ``(B, C, H, W)`` clean images in [-1, 1].
            t: ``(B,)`` integer timesteps.
            noise: the epsilon to add, or None to draw a fresh one.
            model: network to score. Pass ``ema.module`` to use EMA weights.

        Returns:
            Mapping with a ``"loss"`` key holding a shape ``(B,)`` tensor, plus
            ``"mse"`` and ``"vb"`` when those terms are computed separately.
            Keeping the losses per-sample is what lets the caller bucket them
            by timestep instead of averaging the signal away.
        """
        net = model if model is not None else self.model
        eps = torch.randn_like(x_start) if noise is None else noise
        x_t = self.q_sample(x_start, t, noise=eps)
        terms: dict[str, torch.Tensor] = {}

        if self.loss_type.is_variational:
            loss, _ = self.vb_terms_bpd(x_start, x_t, t, model=net)
            if self.loss_type is LossType.RESCALED_KL:
                loss = loss * self.num_timesteps
            terms["loss"] = loss
            return terms

        model_output = net(x_t, t)

        if self.model_var_type.is_learned:
            channels = x_start.shape[1]
            mean_out, var_values = torch.split(model_output, channels, dim=1)
            # Detach the mean so the bound trains only the variance head. Left
            # attached, the KL term degrades the mean that L_simple is fitting.
            frozen = torch.cat([mean_out.detach(), var_values], dim=1)
            vb, _ = self.vb_terms_bpd(x_start, x_t, t, model_output=frozen)
            if self.loss_type is LossType.RESCALED_MSE:
                # The 1/1000 keeps the term's scale independent of the schedule
                # length, matching the reference implementation.
                vb = vb * self.num_timesteps / 1000.0
            terms["vb"] = vb
            model_output = mean_out

        target = self._loss_target(x_start, x_t, t, eps)
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"] + terms["vb"] if "vb" in terms else terms["mse"]
        return terms

    def _loss_target(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """The regression target implied by `model_mean_type`.

        Args:
            x_start: the clean images.
            x_t: the noised images.
            t: ``(B,)`` integer timesteps.
            eps: the noise that was added.

        Returns:
            A tensor shaped like `x_start`.
        """
        match self.model_mean_type:
            case ModelMeanType.EPSILON:
                return eps
            case ModelMeanType.START_X:
                return x_start
            case ModelMeanType.PREVIOUS_X:
                return self.q_posterior_mean_variance(x_start, x_t, t)[0]

    def forward(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        """Sample timesteps uniformly and return the mean training loss.

        Signature-compatible with :class:`~tinydiffusion.diffusion.ddpm.DDPM`,
        so an existing training loop needs no change.

        Args:
            x: ``(B, C, H, W)`` clean images in [-1, 1].
            model: network to score. Defaults to the wrapped model.

        Returns:
            Scalar loss.
        """
        return self.loss_terms(x, model=model).loss

    def loss_terms(self, x: torch.Tensor, model: nn.Module | None = None) -> LossTerms:
        """Take one training step's loss, keeping the per-image breakdown.

        The counterpart of :meth:`~tinydiffusion.diffusion.ddpm.DDPM.loss_terms`,
        so :func:`~tinydiffusion.training.train.train` drives either
        process through the same call. The per-image term is the MSE alone
        whenever there is one: it is the quantity
        :func:`~tinydiffusion.utils.tracking.timestep_quartile_losses` is
        comparable across, and the variational term has its own scale.

        Args:
            x: ``(B, C, H, W)`` clean images in [-1, 1].
            model: network to score. Defaults to the wrapped model.

        Returns:
            The scalar loss, the per-image loss, and the sampled timesteps.
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        terms = self.training_losses(x, t, model=model)
        per_sample = terms.get("mse", terms["loss"])
        return LossTerms(loss=terms["loss"].mean(), per_sample=per_sample.detach(), timesteps=t)

    def loss_at(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        """Score the objective at explicit timesteps.

        Pinning `t` and the noise is what makes two checkpoints comparable; see
        :meth:`~tinydiffusion.diffusion.ddpm.DDPM.loss_at`. The number is this
        process's own objective, so it is only comparable between checkpoints
        that share a `loss_type`.

        Args:
            x: ``(B, C, H, W)`` clean images in [-1, 1].
            t: ``(B,)`` integer timesteps.
            noise: the epsilon to add, or None to draw a fresh one.
            model: network to score. Pass ``ema.module`` to use EMA weights.

        Returns:
            Scalar loss.
        """
        return self.training_losses(x, t, noise=noise, model=model)["loss"].mean()

    # ------------------------------------------------------------------
    # sampling and evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        size: tuple[int, ...],
        device: torch.device | str,
        model: nn.Module | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Run the reverse chain from x_T to x_0.

        Args:
            num_samples: batch size to generate.
            size: shape of one sample, e.g. ``(1, 32, 32)``.
            device: device to generate on.
            model: network to sample from. Pass ``ema.module`` for EMA weights.
            return_trajectory: also keep every intermediate x_t.

        Returns:
            ``(num_samples, *size)``, or ``(num_timesteps + 1, num_samples,
            *size)`` when `return_trajectory` is set.
        """
        net = model if model is not None else self.model
        with eval_mode(net):
            x = torch.randn(num_samples, *size, device=device)
            traj: list[torch.Tensor] = [x] if return_trajectory else []

            for step in reversed(range(self.num_timesteps)):
                t = torch.full((num_samples,), step, device=device, dtype=torch.long)
                mean, _, log_var, _ = self.p_mean_variance(x, t, model=net)
                # No noise on the final step: x_0 is the mean itself.
                noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
                x = mean + (0.5 * log_var).exp() * noise
                if return_trajectory:
                    traj.append(x)

        return torch.stack(traj) if return_trajectory else x

    @torch.no_grad()
    def prior_bpd(self, x_start: torch.Tensor) -> torch.Tensor:
        """KL between q(x_T | x_0) and the standard normal prior.

        This term cannot be optimised — it depends only on the schedule — but
        it belongs in the total bound, and a large value means the forward
        process has not fully destroyed the signal by x_T.

        Args:
            x_start: ``(B, C, H, W)`` clean images.

        Returns:
            Shape ``(B,)`` tensor of bits per dimension.
        """
        t = torch.full((x_start.shape[0],), self.num_timesteps - 1, device=x_start.device)
        mean = _expand(self.sqrtab, t, x_start) * x_start
        log_var = (1.0 - _expand(self.alphabar_t, t, x_start)).log()
        kl = normal_kl(mean, log_var, torch.zeros_like(mean), torch.zeros_like(log_var))
        return mean_flat(kl) / _LOG_2

    @torch.no_grad()
    def calc_bpd_loop(
        self,
        x_start: torch.Tensor,
        *,
        model: nn.Module | None = None,
        clip_denoised: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the whole variational bound, one timestep at a time.

        This is the number to quote when comparing against published
        likelihoods, and it costs one forward pass per timestep — minutes, not
        seconds. Sample quality and bits-per-dim do not always agree, which is
        precisely why both get reported.

        Args:
            x_start: ``(B, C, H, W)`` clean images in [-1, 1].
            model: network to evaluate. Defaults to the wrapped model.
            clip_denoised: clamp the implied x_0 at each step.

        Returns:
            Mapping with ``"total_bpd"``, ``"prior_bpd"`` and ``"vb"``, the
            last holding the per-timestep terms with shape ``(B, T)``.
        """
        net = model if model is not None else self.model
        batch = x_start.shape[0]
        terms: list[torch.Tensor] = []

        with eval_mode(net):
            for step in reversed(range(self.num_timesteps)):
                t = torch.full((batch,), step, device=x_start.device, dtype=torch.long)
                x_t = self.q_sample(x_start, t)
                term, _ = self.vb_terms_bpd(x_start, x_t, t, model=net, clip_denoised=clip_denoised)
                terms.append(term)

        vb = torch.stack(terms[::-1], dim=1)
        prior = self.prior_bpd(x_start)
        return {"total_bpd": vb.sum(dim=1) + prior, "prior_bpd": prior, "vb": vb}

    @classmethod
    def improved(
        cls,
        model: nn.Module,
        betas: tuple[float, float] | torch.Tensor,
        num_timesteps: int = 1000,
    ) -> Self:
        """Build the Nichol & Dhariwal configuration.

        Epsilon prediction with a learned variance range, trained on the
        hybrid objective. The network must emit ``2 * C`` channels.

        Args:
            model: the network.
            betas: schedule, ideally the cosine one.
            num_timesteps: number of diffusion steps.

        Returns:
            A configured instance.
        """
        return cls(
            model,
            betas=betas,
            num_timesteps=num_timesteps,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.LEARNED_RANGE,
            loss_type=LossType.RESCALED_MSE,
        )


def _expand(buf: torch.Tensor, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Gather ``buf[t]`` and reshape it to broadcast against `ref`.

    Args:
        buf: 1-D schedule buffer of length num_timesteps.
        t: ``(B,)`` integer timesteps.
        ref: tensor whose trailing dimensions determine the broadcast shape.

    Returns:
        Shape ``(B, 1, 1, ...)`` tensor.
    """
    return buf.gather(0, t).reshape(-1, *([1] * (ref.dim() - 1)))


type Diffusion = DDPM | GaussianDiffusion
"""Either forward/reverse process.

The two share `net`, `num_timesteps`, the schedule buffers, `loss_terms`,
`loss_at` and `sample`, which is the whole surface the training loop, the
samplers and the evaluator use. Declared here rather than beside
:class:`~tinydiffusion.diffusion.ddpm.DDPM` because this is the module that can
see both classes without an import cycle.
"""
