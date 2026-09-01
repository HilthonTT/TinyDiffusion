"""The four choices that decide what a diffusion process trains against.

:class:`~tinydiffusion.diffusion.ddpm.DDPM` fixes all four; the generalised
process in :mod:`~tinydiffusion.diffusion.gaussian_diffusion` takes them as
arguments. They are :class:`~enum.StrEnum`, so a config file, a ``--set``
override and a checkpoint all round-trip them as the plain strings they read
as — ``"epsilon"``, ``"fixed_small"``, ``"mse"``, ``"min_snr"``.

They live apart from the process itself because they are its vocabulary rather
than its behaviour: the config, the CLI and the checkpoint all name these
without needing the class that acts on them.
"""

from enum import StrEnum

__all__ = [
    "LossType",
    "LossWeighting",
    "ModelMeanType",
    "ModelVarType",
]


class ModelMeanType(StrEnum):
    """What the network's output is interpreted as.

    Attributes:
        EPSILON: the noise added to x_0. The DDPM default, and the easiest to
            train because the target has unit variance at every timestep.
        START_X: the clean image x_0 directly.
        V: the velocity ``v = sqrt(abar) * eps - sqrt(1 - abar) * x_0``
            (Salimans & Ho 2022, https://arxiv.org/abs/2202.00512). It
            interpolates between the two above — epsilon at high noise, x_0 at
            low — so no timestep is left with a target the network cannot see
            the signal in. This is what makes a zero-terminal-SNR schedule and
            short sampling chains work: at t=T, epsilon prediction says nothing
            about x_0, and v prediction still does.
        PREVIOUS_X: the previous latent x_{t-1}. Present for completeness;
            it trains poorly and no current model uses it.
    """

    EPSILON = "epsilon"
    START_X = "start_x"
    V = "v"
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


class LossWeighting(StrEnum):
    """How the per-timestep MSE terms are weighted against each other.

    Attributes:
        UNIFORM: every timestep counts the same, which is what L_simple does.
        MIN_SNR: weight each timestep by ``min(SNR(t), gamma)``, expressed in
            whatever space the network predicts in (Hang et al. 2023,
            https://arxiv.org/abs/2303.09556). Uniform weighting of an
            epsilon-space MSE is implicitly ``1/SNR`` weighting in x_0 space,
            so the low-noise timesteps — where the model is already nearly
            right — dominate the gradient and fight the high-noise ones.
            Clamping the weight at gamma stops that, and typically reaches a
            given loss in a fraction of the steps.
    """

    UNIFORM = "uniform"
    MIN_SNR = "min_snr"
