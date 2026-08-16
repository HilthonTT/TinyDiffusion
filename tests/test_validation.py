import pytest
import torch

from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.models.unet import UNet
from tinydiffusion.training.validation import eval_timesteps, validation_loss


@pytest.fixture
def diffusion(wake):
    net = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        channel_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        image_size=8,
    )
    return DDPM(eps_model=wake(net), num_timesteps=20)


@pytest.fixture
def batches():
    return [(torch.randn(3, 1, 8, 8), torch.zeros(3, dtype=torch.long)) for _ in range(2)]


def test_eval_timesteps_span_the_schedule():
    steps = eval_timesteps(100, 5)
    assert steps.tolist() == [0, 25, 50, 74, 99]


@pytest.mark.parametrize("num_steps", [0, 11])
def test_eval_timesteps_reject_impossible_counts(num_steps):
    with pytest.raises(ValueError, match="num_steps"):
        eval_timesteps(10, num_steps)


def test_the_same_weights_always_score_the_same(diffusion, batches):
    first = validation_loss(diffusion, batches, model=diffusion.net, num_steps=4)
    second = validation_loss(diffusion, batches, model=diffusion.net, num_steps=4)
    assert first == pytest.approx(second)


def test_scoring_does_not_disturb_the_global_rng(diffusion, batches):
    """Reseeding here would rewind the training loop's own randomness."""
    torch.manual_seed(7)
    expected = torch.randn(4)

    torch.manual_seed(7)
    validation_loss(diffusion, batches, model=diffusion.net, num_steps=4)
    assert torch.equal(expected, torch.randn(4))


def test_different_weights_score_differently(diffusion, batches, wake):
    before = validation_loss(diffusion, batches, model=diffusion.net, num_steps=4)
    with torch.no_grad():
        for param in diffusion.net.parameters():
            param.add_(torch.randn_like(param) * 0.5)
    after = validation_loss(diffusion, batches, model=diffusion.net, num_steps=4)
    assert before != pytest.approx(after)


def test_scoring_leaves_the_model_in_the_mode_it_found_it(diffusion, batches):
    diffusion.net.train()
    validation_loss(diffusion, batches, model=diffusion.net, num_steps=2)
    assert diffusion.net.training


def test_an_empty_slice_is_rejected(diffusion):
    with pytest.raises(ValueError, match="at least one batch"):
        validation_loss(diffusion, [], model=diffusion.net)


def test_a_conditional_model_is_scored_on_its_labels(wake):
    net = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        channel_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        image_size=8,
        num_classes=4,
    )
    diffusion = DDPM(eps_model=wake(net), num_timesteps=20)
    x = torch.randn(4, 1, 8, 8)

    zeros = [(x, torch.zeros(4, dtype=torch.long))]
    ones = [(x, torch.ones(4, dtype=torch.long))]
    scored = validation_loss(diffusion, zeros, model=net, num_classes=4, num_steps=3)
    other = validation_loss(diffusion, ones, model=net, num_classes=4, num_steps=3)

    assert scored != pytest.approx(other)
