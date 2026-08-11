import copy

import torch
import torch.nn as nn

class EMA:
    """Exponential moving average of model weights.
 
    DDPM's published sample quality depends on this. Sample from `ema.module`,
    not from the live model.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup: int = 0):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        # Ramp the decay in so early averages are not dominated by random init.
        d = min(self.decay, (1 + self.step) / (10 + self.step)) if self.step <= self.warmup else self.decay
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1.0 - d)
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)
