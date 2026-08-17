"""Building the diffusion process a config describes.

Separate from the training loop on purpose: sampling, evaluation, FID and the
server all have to rebuild a checkpoint's model before they can load its
weights, and none of them should have to import a training loop to do it.
"""

from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.diffusion.gaussian_diffusion import (
    Diffusion,
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from tinydiffusion.diffusion.schedules import cosine_beta_schedule, linear_beta_schedule
from tinydiffusion.models.unet import UNet
from tinydiffusion.training.config import TrainConfig

__all__ = ["build_model"]


def build_model(cfg: TrainConfig) -> Diffusion:
    """Construct the U-Net and wrap it in the diffusion process.

    The parameterisation fields pick the process:
    :class:`~tinydiffusion.diffusion.ddpm.DDPM` for the default
    epsilon/fixed-small/MSE combination it was written for, and
    :class:`~tinydiffusion.diffusion.gaussian_diffusion.GaussianDiffusion` for
    anything else. A learned variance also doubles the network's output
    channels, since it emits the variance parameters alongside the mean.
    ``num_classes`` gives the U-Net a label embedding; the process itself is
    unchanged, since conditioning is carried by the network alone.

    Args:
        cfg: run configuration.

    Returns:
        An untrained diffusion process on the CPU.
    """
    mean_type, var_type, loss_type = cfg.diffusion_types()
    channels = cfg.dataset_spec().channels
    net = UNet(
        in_channels=channels,
        out_channels=channels * (2 if var_type.is_learned else 1),
        base_channels=cfg.base_channels,
        channel_mult=cfg.channel_mult,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions,
        dropout=cfg.dropout,
        image_size=cfg.image_size,
        num_classes=cfg.num_classes,
    )
    if cfg.schedule == "cosine":
        betas = cosine_beta_schedule(cfg.num_timesteps)
    else:
        betas = linear_beta_schedule(cfg.beta_start, cfg.beta_end, cfg.num_timesteps)

    if (mean_type, var_type, loss_type) == (
        ModelMeanType.EPSILON,
        ModelVarType.FIXED_SMALL,
        LossType.MSE,
    ):
        return DDPM(eps_model=net, betas=betas, num_timesteps=cfg.num_timesteps)

    return GaussianDiffusion(
        net,
        betas=betas,
        num_timesteps=cfg.num_timesteps,
        model_mean_type=mean_type,
        model_var_type=var_type,
        loss_type=loss_type,
    )
