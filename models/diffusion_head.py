"""Conditional diffusion over one continuous VAE column (cosine VP, v prediction)."""
import math
import torch
from torch import nn


class DiffusionHead(nn.Module):
    def __init__(self, context_dim, latent_dim, width=256, depth=3, latent_scale=1.0):
        super().__init__()
        if width < 2 or depth < 1 or not math.isfinite(latent_scale) or latent_scale <= 0:
            raise ValueError('Invalid diffusion width, depth or latent scale')
        self.latent_dim = latent_dim
        self.latent_scale = latent_scale
        self.context = nn.Linear(context_dim, width)
        self.input = nn.Linear(latent_dim, width)
        self.time = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, latent_dim))

    @staticmethod
    def coefficients(t):
        angle = t.float().unsqueeze(-1) * (math.pi / 2)
        return angle.cos(), angle.sin()

    def forward(self, noisy, t, context):
        frequencies = torch.exp(torch.arange(16, device=t.device).float() * (-math.log(10000) / 15))
        phases = t.float().unsqueeze(-1) * 1000 * frequencies
        time = torch.cat([phases.sin(), phases.cos()], dim=-1)
        dtype = self.input.weight.dtype
        h = self.input(noisy.to(dtype)) + self.context(context.to(dtype)) + self.time(time.to(dtype))
        for block in self.blocks:
            h = h + block(h)
        return self.output(h)

    def training_loss(self, target, context):
        # All schedule arithmetic and loss reductions stay fp32 under autocast.
        clean = target.float() / self.latent_scale
        t = torch.rand(clean.shape[:-1], device=clean.device) * .998 + .001
        alpha, sigma = self.coefficients(t)
        noise = torch.randn_like(clean)
        noisy = alpha * clean + sigma * noise
        velocity = self(noisy, t, context).float()
        expected = alpha * noise - sigma * clean
        loss = (velocity - expected).square().mean()
        estimate = (alpha * noisy - sigma * velocity) * self.latent_scale
        return loss, estimate

    @torch.no_grad()
    def sample(self, context, steps=20, unconditional=None, cfg_scale=1.0):
        if steps < 1:
            raise ValueError('diffusion sampling steps must be positive')
        x = torch.randn(*context.shape[:-1], self.latent_dim, device=context.device, dtype=torch.float32)
        # Deterministic DDIM updates; randomness comes from the initial Gaussian.
        times = torch.linspace(1, 0, steps + 1, device=context.device)
        for current, following in zip(times[:-1], times[1:]):
            t = current.expand(context.shape[:-1])
            alpha, sigma = self.coefficients(t)
            v = self(x, t, context).float()
            if unconditional is not None and cfg_scale != 1:
                u = self(x, t, unconditional).float()
                v = u + cfg_scale * (v - u)
            clean = alpha * x - sigma * v
            noise = sigma * x + alpha * v
            next_alpha, next_sigma = self.coefficients(following.expand_as(t))
            x = next_alpha * clean + next_sigma * noise
        return (x * self.latent_scale).to(context.dtype)
