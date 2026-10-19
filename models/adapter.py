import torch.nn as nn


class FeatureAdapter(nn.Module):
    """Lightweight residual adapter on top of frozen CLIP features.

    ratio controls how much the adapter shifts the original feature.
    Small ratio preserves CLIP semantics; larger ratio allows CZSL-specific
    fine-grained calibration.
    """
    def __init__(self, dim=768, hidden=1024, ratio=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.ratio = ratio

    def forward(self, x):
        return (1.0 - self.ratio) * x + self.ratio * self.net(x)
