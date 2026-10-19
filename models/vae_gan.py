import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """VAE encoder: (real feature, condition) -> (mu, logvar).

    logvar is clamped to a safe range to prevent the KL term from diverging.
    """
    def __init__(self, dim=768, d_c=512, d_z=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + d_c, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),      nn.LeakyReLU(0.2),
        )
        self.fc_mu     = nn.Linear(1024, d_z)
        self.fc_logvar = nn.Linear(1024, d_z)

    def forward(self, x, c):
        h = self.net(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10.0, 10.0)


class Generator(nn.Module):
    """Conditional generator: (z, condition) -> synthetic feature.

    Output is L2-normalised so synthetic features live on the same unit
    hypersphere as the (normalised) CLIP image features. This keeps the
    WGAN-GP / reconstruction objectives bounded and training stable.
    """
    def __init__(self, dim=768, d_c=512, d_z=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z + d_c, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),      nn.LeakyReLU(0.2),
            nn.Linear(1024, dim),
        )

    def forward(self, z, c):
        out = self.net(torch.cat([z, c], dim=-1))
        return F.normalize(out, dim=-1)


class Discriminator(nn.Module):
    """WGAN-GP discriminator: (feature, condition) -> scalar score."""
    def __init__(self, dim=768, d_c=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + d_c, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),       nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=-1))


def reparameterize(mu, logvar):
    std = (0.5 * logvar).exp()
    return mu + std * torch.randn_like(std)
