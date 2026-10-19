"""
All loss functions for the cVAE-GAN generator training.

  L_G = L_gan_G + λ_r*L_recon + λ_kl*L_kl + λ_clip*L_clip
        + λ_dis*L_dis + λ_cyc*L_cycle + λ_div*L_div
  L_D = L_gan_D  (WGAN-GP)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── KL divergence ────────────────────────────────────────────────────────────

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ── WGAN-GP ──────────────────────────────────────────────────────────────────

def gradient_penalty(discriminator, real, fake, c, device):
    B = real.size(0)
    alpha = torch.rand(B, 1, device=device).expand_as(real)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interp, c)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )[0]
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def wgan_d_loss(discriminator, real, fake, c, device, lambda_gp=10.0):
    loss = discriminator(fake.detach(), c).mean() - discriminator(real, c).mean()
    gp   = gradient_penalty(discriminator, real, fake.detach(), c, device)
    return loss + lambda_gp * gp


def wgan_g_loss(discriminator, fake, c):
    return -discriminator(fake, c).mean()


# ── CLIP alignment (contrastive) ─────────────────────────────────────────────

def clip_alignment_loss(x_hat, pair_text_feats, pair_idx, tau=0.07):
    """
    x_hat:           [B, 768] synthetic features (should already be L2-normed)
    pair_text_feats: [n_pairs, 768] all composition text embeddings
    pair_idx:        [B] ground-truth pair index for each sample
    """
    x_hat = F.normalize(x_hat, dim=-1)
    logits = x_hat @ pair_text_feats.t() / tau      # [B, n_pairs]
    return F.cross_entropy(logits, pair_idx)


# ── Disentangle consistency ──────────────────────────────────────────────────

def disentangle_loss(dis_heads, x_hat, attr_idx, obj_idx):
    attr_logits, obj_logits = dis_heads(x_hat)
    return (F.cross_entropy(attr_logits, attr_idx) +
            F.cross_entropy(obj_logits, obj_idx))


# ── Cycle consistency ─────────────────────────────────────────────────────────

def cycle_loss(generator, modulator, adapter,
               z, t_a, t_a_prime, t_o,
               x_hat, dis_heads, lambda_obj=0.5):
    """
    Swap attribute a -> a', generate x_hat', then swap back.
    Two constraints:
      1. x_hat'' (back to a) should be close to x_hat (feature-level cycle).
      2. The object identity in x_hat' should match x_hat (obj head).
    """
    c_prime  = modulator(adapter(t_a_prime), adapter(t_o))
    x_hat_p  = generator(z, c_prime)                  # swap attr

    c_back   = modulator(adapter(t_a), adapter(t_o))
    x_hat_pp = generator(z, c_back)                   # swap back

    loss_feat = F.mse_loss(x_hat_pp, x_hat.detach())

    _, obj_logits_p  = dis_heads(x_hat_p)
    _, obj_logits    = dis_heads(x_hat.detach())
    obj_idx_pred = obj_logits.argmax(dim=-1)
    loss_obj = F.cross_entropy(obj_logits_p, obj_idx_pred)

    return loss_feat + lambda_obj * loss_obj


# ── Diversity ────────────────────────────────────────────────────────────────

def diversity_loss(generator, z1, z2, c, eps=1e-8):
    x1 = generator(z1, c)
    x2 = generator(z2, c)
    feat_dist = (x1 - x2).norm(dim=-1)
    noise_dist = (z1 - z2).norm(dim=-1).clamp(min=eps)
    return -(feat_dist / noise_dist).mean()
