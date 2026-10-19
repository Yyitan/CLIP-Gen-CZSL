"""
Stage B: Synthesise unseen features with trained generator, then train
the composition classifier on real-seen + synthetic-unseen features.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, "./lyt/czsl/CLIP_Gen_CZSL")
from models import FeatureAdapter, CompositionModulator, Generator
from models.classifier import CompositionClassifier
from utils import load_args, set_seed

PROJECT_ROOT = "./lyt/czsl/CLIP_Gen_CZSL"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True)
    parser.add_argument("--dataset",        default="ut-zappos")
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--feat_dim",       type=int,   default=768)
    parser.add_argument("--d_c",            type=int,   default=512)
    parser.add_argument("--d_z",            type=int,   default=512)
    parser.add_argument("--adapter_ratio",  type=float, default=0.2)
    parser.add_argument("--gen_checkpoint", default=None,
                        help="path to gen_final.pt from Stage A")
    parser.add_argument("--k_syn",          type=int,   default=300,
                        help="synthetic samples per unseen composition")
    parser.add_argument("--epochs_clf",     type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=2048)
    parser.add_argument("--lr_clf",         type=float, default=1e-3)
    parser.add_argument("--open_world",     action="store_true", default=False)
    parser.add_argument("--feasibility_threshold", type=float, default=None)
    parser.add_argument("--output_dir",     default=None)
    return parser.parse_args()


def load_cache(dataset_name):
    cache_dir = os.path.join(PROJECT_ROOT, "clip_feats", dataset_name)
    feats_tr  = torch.load(os.path.join(cache_dir, "image_feats_train.pt"),  weights_only=True)
    labels_tr = torch.load(os.path.join(cache_dir, "image_labels_train.pt"), weights_only=True)
    meta      = torch.load(os.path.join(cache_dir, "meta.pt"),               weights_only=False)
    t_attrs   = torch.load(os.path.join(cache_dir, "text_feats_attrs.pt"),   weights_only=True)
    t_objs    = torch.load(os.path.join(cache_dir, "text_feats_objs.pt"),    weights_only=True)
    return feats_tr, labels_tr, meta, t_attrs, t_objs


def get_unseen_pairs(meta, open_world=False, feasibility=None, threshold=None):
    """Return list of (attr_idx, obj_idx, pair_idx) for unseen compositions."""
    train_set = set(meta["train_pairs"])
    attr2idx  = meta["attr2idx"]
    obj2idx   = meta["obj2idx"]
    pair2idx  = meta["pair2idx"]

    if open_world and feasibility is not None and threshold is not None:
        candidates = [(a, o) for a in meta["attrs"] for o in meta["objs"]
                      if (a, o) not in train_set]
        feas_flat  = feasibility.view(-1)
        n_attrs    = len(meta["attrs"])
        result = []
        for a, o in candidates:
            a_i = attr2idx[a]; o_i = obj2idx[o]
            if feas_flat[a_i * len(meta["objs"]) + o_i].item() >= threshold:
                p_i = pair2idx.get((a, o), -1)
                if p_i >= 0:
                    result.append((a_i, o_i, p_i))
        return result
    else:
        # closed world: unseen = test_pairs \ train_pairs
        unseen = [p for p in meta["test_pairs"] if p not in train_set]
        return [(attr2idx[a], obj2idx[o], pair2idx[(a, o)])
                for a, o in unseen if (a, o) in pair2idx]


@torch.no_grad()
def synthesise(generator, modulator, adapter, t_attrs, t_objs,
               unseen_list, k, d_z, device):
    """Generate k synthetic features per unseen (attr, obj) pair."""
    generator.eval(); modulator.eval(); adapter.eval()
    syn_feats, syn_labels = [], []
    for a_i, o_i, p_i in tqdm(unseen_list, desc="Synthesising"):
        t_a = t_attrs[a_i].unsqueeze(0).to(device)
        t_o = t_objs[o_i].unsqueeze(0).to(device)
        c   = modulator(t_a, t_o).expand(k, -1)
        z   = torch.randn(k, d_z, device=device)
        x   = generator(z, c)
        syn_feats.append(x.cpu())
        syn_labels.append(torch.full((k,), p_i, dtype=torch.long))
    return torch.cat(syn_feats), torch.cat(syn_labels)


def build_seen_pair_idx(feats, labels, meta):
    """Map (attr_idx, obj_idx) labels to global pair indices for seen data."""
    pair2idx  = meta["pair2idx"]
    attr_list = meta["attrs"]
    obj_list  = meta["objs"]
    pair_idx  = []
    for i in range(labels.size(0)):
        attr = attr_list[labels[i, 0].item()]
        obj  = obj_list[labels[i, 1].item()]
        pair_idx.append(pair2idx[(attr, obj)])
    return torch.tensor(pair_idx, dtype=torch.long)


def train_classifier(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print(f"Loading cache for {args.dataset} ...")
    feats_tr, labels_tr, meta, t_attrs, t_objs = load_cache(args.dataset)
    n_pairs = len(meta["pairs"])

    # Load trained generator
    gen_path = args.gen_checkpoint or os.path.join(
        PROJECT_ROOT, "output", args.dataset, "gen", "gen_final.pt")
    print(f"Loading generator from {gen_path} ...")
    ckpt = torch.load(gen_path, map_location="cpu", weights_only=False)

    adapter   = FeatureAdapter(args.feat_dim, ratio=args.adapter_ratio)
    modulator = CompositionModulator(args.feat_dim, args.d_c)
    generator = Generator(args.feat_dim, args.d_c, args.d_z)
    adapter.load_state_dict(ckpt["adapter"])
    modulator.load_state_dict(ckpt["modulator"])
    generator.load_state_dict(ckpt["generator"])
    adapter.to(device); modulator.to(device); generator.to(device)
    t_attrs = t_attrs.to(device)
    t_objs  = t_objs.to(device)

    # Get unseen pairs to synthesise
    unseen_list = get_unseen_pairs(meta, open_world=args.open_world)
    print(f"Unseen pairs to synthesise: {len(unseen_list)}, k={args.k_syn}")

    syn_feats, syn_labels = synthesise(
        generator, modulator, adapter,
        t_attrs, t_objs, unseen_list,
        args.k_syn, args.d_z, device
    )

    # Real seen features (already adapted during gen training; re-adapt here)
    adapter.eval()
    with torch.no_grad():
        seen_feats_list = []
        for i in range(0, feats_tr.size(0), 2048):
            fa = F.normalize(adapter(feats_tr[i:i+2048].to(device)), dim=-1)
            seen_feats_list.append(fa.cpu())
        seen_feats = torch.cat(seen_feats_list)

    seen_labels = build_seen_pair_idx(seen_feats, labels_tr, meta)

    # Combine
    all_feats  = torch.cat([seen_feats, syn_feats],   dim=0)
    all_labels = torch.cat([seen_labels, syn_labels],  dim=0)
    print(f"Training set: {seen_feats.size(0)} real seen + "
          f"{syn_feats.size(0)} synthetic unseen = {all_feats.size(0)} total")

    dset   = TensorDataset(all_feats, all_labels)
    loader = DataLoader(dset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    classifier = CompositionClassifier(args.feat_dim, n_pairs).to(device)
    opt = torch.optim.Adam(classifier.parameters(), lr=args.lr_clf)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs_clf)

    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "output", args.dataset, "clf")
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(args.epochs_clf):
        classifier.train()
        total_loss = 0.0
        for x, y in tqdm(loader, desc=f"Clf Epoch {epoch+1}/{args.epochs_clf}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = classifier(x)
            loss   = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"  epoch {epoch+1} loss={total_loss/len(loader):.4f}")

    clf_path = os.path.join(out_dir, "classifier_final.pt")
    torch.save({
        "classifier": classifier.state_dict(),
        "adapter":    adapter.state_dict(),
        "args":       vars(args),
    }, clf_path)
    print(f"Classifier saved to {clf_path}")


if __name__ == "__main__":
    args = get_args()
    if args.config:
        load_args(args.config, args)
    train_classifier(args)
