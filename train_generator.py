import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, "./lyt/czsl/CLIP_Gen_CZSL")
from models import FeatureAdapter, CompositionModulator, Encoder, Generator, Discriminator
from models.classifier import DisentangleHeads
from losses import (kl_loss, wgan_d_loss, wgan_g_loss,
                    clip_alignment_loss, disentangle_loss,
                    cycle_loss, diversity_loss)
from utils import load_args, set_seed

PROJECT_ROOT = "./lyt/czsl/CLIP_Gen_CZSL"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",            required=True)
    parser.add_argument("--dataset",           default="ut-zappos")
    parser.add_argument("--seed",              type=int,   default=0)
    parser.add_argument("--feat_dim",          type=int,   default=768)
    parser.add_argument("--d_c",               type=int,   default=512)
    parser.add_argument("--d_z",               type=int,   default=512)
    parser.add_argument("--adapter_ratio",     type=float, default=0.2)
    parser.add_argument("--epochs",            type=int,   default=100)
    parser.add_argument("--batch_size",        type=int,   default=1024)
    parser.add_argument("--lr_g",              type=float, default=1e-4)
    parser.add_argument("--lr_d",              type=float, default=1e-4)
    parser.add_argument("--n_critic",          type=int,   default=5)
    parser.add_argument("--lambda_gp",         type=float, default=10.0)
    parser.add_argument("--lambda_r",          type=float, default=1.0)
    parser.add_argument("--lambda_kl",         type=float, default=0.1)
    parser.add_argument("--lambda_clip",       type=float, default=0.3)
    parser.add_argument("--lambda_dis",        type=float, default=0.5)
    parser.add_argument("--lambda_cyc",        type=float, default=0.2)
    parser.add_argument("--lambda_div",        type=float, default=0.1)
    parser.add_argument("--clip_tau",          type=float, default=0.07)
    parser.add_argument("--use_cycle",         action="store_true", default=False)
    parser.add_argument("--cycle_start_ratio", type=float, default=0.4)
    parser.add_argument("--save_every",        type=int,   default=10)
    parser.add_argument("--output_dir",        default=None)
    return parser.parse_args()


def load_cache(dataset_name):
    cache_dir = os.path.join(PROJECT_ROOT, "clip_feats", dataset_name)
    assert os.path.exists(cache_dir), (
        f"Cache not found at {cache_dir}. Run extract_features.py first."
    )
    feats   = torch.load(os.path.join(cache_dir, "image_feats_train.pt"),  weights_only=True)
    labels  = torch.load(os.path.join(cache_dir, "image_labels_train.pt"), weights_only=True)
    meta    = torch.load(os.path.join(cache_dir, "meta.pt"),               weights_only=False)
    t_attrs = torch.load(os.path.join(cache_dir, "text_feats_attrs.pt"),   weights_only=True)
    t_objs  = torch.load(os.path.join(cache_dir, "text_feats_objs.pt"),    weights_only=True)
    t_pairs = torch.load(os.path.join(cache_dir, "text_feats_pairs.pt"),   weights_only=True)
    return feats, labels, meta, t_attrs, t_objs, t_pairs


def build_pair_idx_tensor(labels, meta):
    pair2idx  = meta["pair2idx"]
    attr_list = meta["attrs"]
    obj_list  = meta["objs"]
    pair_idx  = []
    for i in range(labels.size(0)):
        attr = attr_list[labels[i, 0].item()]
        obj  = obj_list[labels[i, 1].item()]
        pair_idx.append(pair2idx[(attr, obj)])
    return torch.tensor(pair_idx, dtype=torch.long)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print(f"Loading cache for {args.dataset} ...")
    feats, labels, meta, t_attrs, t_objs, t_pairs = load_cache(args.dataset)
    pair_idx = build_pair_idx_tensor(labels, meta)

    n_attrs = len(meta["attrs"])
    n_objs  = len(meta["objs"])
    n_pairs = len(meta["pairs"])
    print(f"  {feats.shape[0]} train samples | "
          f"{n_attrs} attrs | {n_objs} objs | {n_pairs} pairs")

    t_attrs = t_attrs.to(device)
    t_objs  = t_objs.to(device)
    t_pairs = t_pairs.to(device)

    attr_idx = labels[:, 0]
    obj_idx  = labels[:, 1]
    dset = TensorDataset(feats, attr_idx, obj_idx, pair_idx)
    loader = DataLoader(dset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    adapter   = FeatureAdapter(args.feat_dim, ratio=args.adapter_ratio).to(device)
    modulator = CompositionModulator(args.feat_dim, args.d_c).to(device)
    encoder   = Encoder(args.feat_dim, args.d_c, args.d_z).to(device)
    generator = Generator(args.feat_dim, args.d_c, args.d_z).to(device)
    disc      = Discriminator(args.feat_dim, args.d_c).to(device)
    dis_heads = DisentangleHeads(args.feat_dim, n_attrs, n_objs).to(device)

    g_params = (list(adapter.parameters()) + list(modulator.parameters()) +
                list(encoder.parameters()) + list(generator.parameters()) +
                list(dis_heads.parameters()))
    opt_g = torch.optim.Adam(g_params, lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "output", args.dataset, "gen")
    os.makedirs(out_dir, exist_ok=True)

    cycle_start_epoch = int(args.epochs * args.cycle_start_ratio)
    ckpt = None

    for epoch in range(args.epochs):
        use_cycle_now = args.use_cycle and (epoch >= cycle_start_epoch)
        adapter.train(); modulator.train(); encoder.train()
        generator.train(); disc.train(); dis_heads.train()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}/{args.epochs}")
        running = {}

        for step, (x, a_idx, o_idx, p_idx) in enumerate(pbar):
            x     = x.to(device)
            a_idx = a_idx.to(device)
            o_idx = o_idx.to(device)
            p_idx = p_idx.to(device)
            B     = x.size(0)

            x_a = F.normalize(adapter(x), dim=-1)   # real features on unit sphere
            t_a = t_attrs[a_idx]
            t_o = t_objs[o_idx]
            c   = modulator(t_a, t_o)

            # -- discriminator (n_critic steps) --
            # GAN fakes come from the PRIOR z~N(0,I) (the distribution we
            # actually sample from at synthesis time), so the critic learns to
            # penalise prior-sampling mode collapse. This is the tf-VAEGAN split.
            for _ in range(args.n_critic):
                with torch.no_grad():
                    z_p   = torch.randn(B, args.d_z, device=device)
                    x_hat = generator(z_p, c.detach())
                opt_d.zero_grad()
                loss_d = wgan_d_loss(disc, x_a.detach(), x_hat,
                                     c.detach(), device, args.lambda_gp)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
                opt_d.step()

            # -- generator --
            opt_g.zero_grad()

            # VAE branch: reconstruction + KL on the POSTERIOR z_enc
            mu, logvar = encoder(x_a, c)
            z_enc = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
            x_hat_enc = generator(z_enc, c)
            loss_recon = ((x_hat_enc - x_a) ** 2).mean()
            loss_kl    = kl_loss(mu, logvar)

            # GAN / semantic branch: PRIOR z~N(0,I) -> the synthesis distribution
            z_p   = torch.randn(B, args.d_z, device=device)
            x_hat = generator(z_p, c)
            loss_ganG  = wgan_g_loss(disc, x_hat, c)
            loss_clip  = clip_alignment_loss(x_hat, t_pairs, p_idx,
                                             tau=args.clip_tau)
            loss_dis   = disentangle_loss(dis_heads, x_hat, a_idx, o_idx)

            loss_cyc = torch.tensor(0.0, device=device)
            if use_cycle_now:
                perm      = torch.randperm(B, device=device)
                t_a_prime = t_attrs[a_idx[perm]]
                loss_cyc  = cycle_loss(
                    generator, modulator, adapter,
                    z_p.detach(), t_a, t_a_prime, t_o,
                    x_hat, dis_heads
                )

            z1 = torch.randn(B, args.d_z, device=device)
            z2 = torch.randn(B, args.d_z, device=device)
            loss_div = diversity_loss(generator, z1, z2, c.detach())

            loss_g = (loss_ganG
                      + args.lambda_r    * loss_recon
                      + args.lambda_kl   * loss_kl
                      + args.lambda_clip * loss_clip
                      + args.lambda_dis  * loss_dis
                      + args.lambda_cyc  * loss_cyc
                      + args.lambda_div  * loss_div)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(g_params, 5.0)
            opt_g.step()

            for k, v in [("D", loss_d), ("G", loss_ganG), ("recon", loss_recon),
                          ("kl", loss_kl), ("clip", loss_clip),
                          ("dis", loss_dis), ("cyc", loss_cyc), ("div", loss_div)]:
                running[k] = running.get(k, 0.0) * 0.95 + v.item() * 0.05
            pbar.set_postfix({k: f"{v:.3f}" for k, v in running.items()})

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = {
                "epoch":     epoch + 1,
                "adapter":   adapter.state_dict(),
                "modulator": modulator.state_dict(),
                "encoder":   encoder.state_dict(),
                "generator": generator.state_dict(),
                "dis_heads": dis_heads.state_dict(),
                "args":      vars(args),
            }
            path = os.path.join(out_dir, f"gen_epoch{epoch+1}.pt")
            torch.save(ckpt, path)
            print(f"  Saved {path}")

    final_path = os.path.join(out_dir, "gen_final.pt")
    torch.save(ckpt, final_path)
    print(f"Generator training done. Final: {final_path}")


if __name__ == "__main__":
    args = get_args()
    if args.config:
        load_args(args.config, args)
    train(args)
