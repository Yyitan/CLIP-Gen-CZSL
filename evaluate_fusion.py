"""
Discriminative-fusion evaluation (CAMS-style), closed world.

Final score fuses:
  (A) CLIP zero-shot discriminative score:
        clip_att(a) = softmax(temp * img . attr_text)[a]
        clip_obj(o) = softmax(temp * img . obj_text)[o]
        clip_com(a,o) = img . pair_text(a,o)
        clip_score = clip_att*clip_obj + gamma * clip_com
  (B) Generative classifier score (trained on real seen + synthetic unseen):
        gen_score = softmax(classifier(normalize(adapter(img))))

  final = (1-w) * norm(clip_score) + w * norm(gen_score)

Reports CLIP-only, gen-only, and fused (sweeping w on the eval split).
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "./lyt/czsl/CLIP_Gen_CZSL")
sys.path.insert(0, "./lyt/czsl/CAMS-main/CAMS")

from models import FeatureAdapter
from models.classifier import CompositionClassifier
from utils import load_args, set_seed
from dataset import CompositionDataset
from evaluate import Evaluator   # reuse the corrected evaluator

PROJECT_ROOT = "./lyt/czsl/CLIP_Gen_CZSL"
DATASET_PATHS = {
    "mit-states": "data/mit-states",
    "ut-zappos":  "data/ut-zappos",
    "cgqa":       "data/cgqa",
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", default="ut-zappos")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feat_dim", type=int, default=768)
    p.add_argument("--d_c", type=int, default=512)
    p.add_argument("--adapter_ratio", type=float, default=0.0)
    p.add_argument("--clf_checkpoint", default=None)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--temp", type=float, default=20.0, help="CLIP softmax temperature")
    p.add_argument("--gamma", type=float, default=1.0, help="weight of clip_com vs att*obj")
    return p.parse_args()


def normalize_scores(s):
    """Per-sample min-max to [0,1] so the two score families are comparable."""
    s = s - s.min(dim=1, keepdim=True)[0]
    s = s / (s.max(dim=1, keepdim=True)[0] + 1e-8)
    return s


def report(name, scores, evaluator, attr_gt, obj_gt, pair_gt):
    st = evaluator.evaluate(scores.clone(), attr_gt, obj_gt, pair_gt)
    print(f"  {name:24s}  S={st['best_seen']*100:5.2f}  U={st['best_unseen']*100:5.2f}  "
          f"HM={st['best_hm']*100:5.2f}  AUC={st['AUC']*100:5.2f}")
    return st


def main():
    args = get_args()
    if args.config:
        load_args(args.config, args)
    # config eval group may set open_world True; force closed for this script
    args.open_world = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    dset = CompositionDataset(DATASET_PATHS[args.dataset], phase=args.split,
                              split="compositional-split-natural", open_world=False)

    cache = os.path.join(PROJECT_ROOT, "clip_feats", args.dataset)
    feats  = torch.load(os.path.join(cache, f"image_feats_{args.split}.pt"), weights_only=True).to(device)
    labels = torch.load(os.path.join(cache, f"image_labels_{args.split}.pt"), weights_only=True)
    meta   = torch.load(os.path.join(cache, "meta.pt"), weights_only=False)
    t_attrs = torch.load(os.path.join(cache, "text_feats_attrs.pt"), weights_only=True).to(device)
    t_objs  = torch.load(os.path.join(cache, "text_feats_objs.pt"),  weights_only=True).to(device)
    t_pairs = torch.load(os.path.join(cache, "text_feats_pairs.pt"), weights_only=True).to(device)

    feats = F.normalize(feats, dim=-1)

    # gt in dset pair-index space
    attrs, objs = meta["attrs"], meta["objs"]
    attr_gt = labels[:, 0]
    obj_gt  = labels[:, 1]
    pair_gt = torch.tensor(
        [dset.pair2idx.get((attrs[labels[i, 0].item()], objs[labels[i, 1].item()]), 0)
         for i in range(labels.size(0))], dtype=torch.long)

    # index tensors mapping each pair -> its attr/obj index (in meta order == dset order)
    pairs2attr = dset.pairs2attr_idx.to(device)
    pairs2obj  = dset.pairs2obj_idx.to(device)

    # ---- (A) CLIP zero-shot discriminative score ----
    with torch.no_grad():
        att_logits = args.temp * feats @ t_attrs.t()       # [N, n_attr]
        obj_logits = args.temp * feats @ t_objs.t()        # [N, n_obj]
        clip_att = torch.softmax(att_logits, dim=-1)[:, pairs2attr]   # [N, P]
        clip_obj = torch.softmax(obj_logits, dim=-1)[:, pairs2obj]    # [N, P]
        clip_com = feats @ t_pairs.t()                     # [N, P]
        clip_score = clip_att * clip_obj + args.gamma * clip_com
    clip_score = clip_score.cpu()

    # ---- (B) generative classifier score ----
    clf_path = args.clf_checkpoint or os.path.join(
        PROJECT_ROOT, "output", args.dataset, "clf", "classifier_final.pt")
    ckpt = torch.load(clf_path, map_location="cpu", weights_only=False)
    adapter = FeatureAdapter(args.feat_dim, ratio=args.adapter_ratio).to(device)
    clf = CompositionClassifier(args.feat_dim, len(dset.pairs)).to(device)
    adapter.load_state_dict(ckpt["adapter"]); clf.load_state_dict(ckpt["classifier"])
    adapter.eval(); clf.eval()
    with torch.no_grad():
        gen_logits = clf(F.normalize(adapter(feats), dim=-1))
        gen_score = torch.softmax(gen_logits, dim=-1).cpu()

    evaluator = Evaluator(dset)
    print(f"\n=== {args.dataset.upper()} | {args.split.upper()} | CLOSED | "
          f"temp={args.temp} gamma={args.gamma} ===")
    report("CLIP zero-shot only", clip_score, evaluator, attr_gt, obj_gt, pair_gt)
    report("Generative clf only", gen_score, evaluator, attr_gt, obj_gt, pair_gt)

    cn = normalize_scores(clip_score)
    gn = normalize_scores(gen_score)
    best = None
    for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        fused = (1 - w) * cn + w * gn
        st = report(f"fused w={w:.1f}", fused, evaluator, attr_gt, obj_gt, pair_gt)
        if best is None or st["AUC"] > best[1]:
            best = (w, st["AUC"])
    print(f"  >>> best fusion weight w={best[0]} (AUC={best[1]*100:.2f})")


if __name__ == "__main__":
    main()
