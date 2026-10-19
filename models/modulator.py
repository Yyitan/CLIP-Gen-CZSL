import torch.nn as nn


class CompositionModulator(nn.Module):
    """FiLM-style composition modulator.

    Attribute acts as a transformation applied to the object: the attribute
    embedding generates scale (gamma) and shift (beta) parameters that modulate
    the object embedding, making the condition context-aware rather than a naive
    concatenation.
    """
    def __init__(self, dim=768, d_c=512):
        super().__init__()
        self.to_gamma = nn.Linear(dim, dim)
        self.to_beta  = nn.Linear(dim, dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, d_c),
            nn.GELU(),
            nn.Linear(d_c, d_c),
        )

    def forward(self, t_a, t_o):
        gamma = self.to_gamma(t_a)
        beta  = self.to_beta(t_a)
        modulated = gamma * t_o + beta
        return self.proj(modulated)   # [B, d_c]
