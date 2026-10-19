"""
Evaluation script — computes Seen/Unseen/HM/AUC on val or test split.

Works in both closed-world and open-world settings. For open-world,
applies feasibility masking from CAMS pre-computed feasibility files.
"""
import os
import sys
import copy
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.stats import hmean

sys.path.insert(0, "./lyt/czsl/CLIP_Gen_CZSL")
sys.path.insert(0, "./lyt/czsl/CAMS-main/CAMS")

from models import FeatureAdapter
from models.classifier import CompositionClassifier
from utils import load_args, set_seed
from dataset import CompositionDataset

PROJECT_ROOT = "./lyt/czsl/CLIP_Gen_CZSL"

DATASET_PATHS = {
    "mit-states": "data/mit-states",
    "ut-zappos":  "data/ut-zappos",
    "cgqa":       "data/cgqa",
}

FEASIBILITY_PATHS = {
    "mit-states": "./lyt/czsl/CAMS-main/CAMS/feasibilities/feasibility_mit.pt",
    "ut-zappos":  "./lyt/czsl/CAMS-main/CAMS/feasibilities/feasibility_zp.pt",
    "cgqa":       "./lyt/czsl/CAMS-main/CAMS/feasibilities/feasibility_cgqa.pt",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",              required=True)
    parser.add_argument("--dataset",             default="ut-zappos")
    parser.add_argument("--seed",                type=int,   default=0)
    parser.add_argument("--feat_dim",            type=int,   default=768)
    parser.add_argument("--d_c",                 type=int,   default=512)
    parser.add_argument("--adapter_ratio",       type=float, default=0.2)
    parser.add_argument("--clf_checkpoint",      default=None)
    parser.add_argument("--split",               default="test",
                        choices=["val", "test"])
    parser.add_argument("--open_world",          action="store_true", default=False)
    parser.add_argument("--threshold",           type=float, default=None)
    parser.add_argument("--eval_batch_size",     type=int,   default=512)
    return parser.parse_args()


# ── re-use CAMS evaluator logic inline (no import of CAMS test.py) ───────────

class Evaluator:
    def __init__(self, dset):
        self.dset = dset
        pairs = [(dset.attr2idx[a], dset.obj2idx[o]) for a, o in dset.pairs]
        self.train_pairs = [(dset.attr2idx[a], dset.obj2idx[o])
                            for a, o in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        if dset.phase == "val":
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt  = set(dset.val_pairs)
        else:
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt  = set(dset.test_pairs)

        self.test_pair_dict = {
            (dset.attr2idx[a], dset.obj2idx[o]): [dset.pair2idx[(a, o)], 0, 0]
            for a, o in test_pair_gt
        }

        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if p in test_pair_set else 0 for p in dset.pairs]
        self.closed_mask = torch.BoolTensor(masks)

        seen_set = set(dset.train_pairs)
        self.seen_mask = torch.BoolTensor(
            [1 if p in seen_set else 0 for p in dset.pairs])

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias
        mask2 = self.closed_mask.repeat(scores.shape[0], 1)
        cs = scores.clone()
        cs[~mask2] = -1e10
        _, pair_pred = cs.topk(topk, dim=1)
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred = self.pairs[pair_pred][:, 0].view(-1, topk)
        obj_pred  = self.pairs[pair_pred][:, 1].view(-1, topk)
        return {"closed": (attr_pred, obj_pred)}

    def evaluate(self, all_scores, all_attr_gt, all_obj_gt, all_pair_gt, topk=1):
        """Compute Seen/Unseen/HM/AUC from raw logit tensor [N, n_pairs]."""
        pairs_tensor = torch.LongTensor(
            [(self.dset.attr2idx[a], self.dset.obj2idx[o])
             for a, o in self.dset.pairs])

        seen_pairs_set = set(self.train_pairs)
        gt_pairs = list(zip(all_attr_gt.tolist(), all_obj_gt.tolist()))
        seen_idx   = torch.LongTensor(
            [i for i, p in enumerate(gt_pairs) if tuple(p) in seen_pairs_set])
        unseen_idx = torch.LongTensor(
            [i for i, p in enumerate(gt_pairs) if tuple(p) not in seen_pairs_set])

        def match(ap, op):
            am = (all_attr_gt.unsqueeze(1) == ap[:, :topk]).any(1).float()
            om = (all_obj_gt.unsqueeze(1)  == op[:, :topk]).any(1).float()
            return am * om

        # Biased (bias=1e3, unseen-favoured) closed predictions give the
        # best-case unseen matches and the sweep endpoints (CAMS protocol).
        res_b = self.score_fast_model(all_scores.clone(), None, bias=1e3, topk=topk)
        m_biased = match(res_b["closed"][0], res_b["closed"][1])
        seen_match   = float(m_biased[seen_idx].mean())   if len(seen_idx)   else 0.0
        unseen_match = float(m_biased[unseen_idx].mean()) if len(unseen_idx) else 0.0

        # AUC sweep: derive bias list from correctly-matched (biased) unseen
        correct_scores = all_scores[torch.arange(all_scores.size(0)), all_pair_gt][unseen_idx]
        max_seen = all_scores[unseen_idx][:, self.seen_mask].topk(topk, dim=1)[0][:, topk - 1]
        diff = (max_seen - correct_scores)
        unseen_matches_bool = m_biased[unseen_idx].bool()
        if unseen_matches_bool.sum() > 0:
            diffs_corr = diff[unseen_matches_bool] - 1e-4
            diffs_corr = torch.sort(diffs_corr)[0]
            skip = max(len(diffs_corr) // 20, 1)
            biaslist = diffs_corr[::skip].tolist()
        else:
            biaslist = []

        seen_acc_list, unseen_acc_list = [], []
        for bias in biaslist:
            sc = all_scores.clone()
            res = self.score_fast_model(sc, None, bias=bias, topk=topk)
            ap2 = res["closed"][0]; op2 = res["closed"][1]
            m2  = match(ap2, op2)
            seen_acc_list.append(float(m2[seen_idx].mean())   if len(seen_idx)   else 0.0)
            unseen_acc_list.append(float(m2[unseen_idx].mean()) if len(unseen_idx) else 0.0)

        seen_acc_list.append(seen_match)
        unseen_acc_list.append(unseen_match)
        seen_arr   = np.array(seen_acc_list)
        unseen_arr = np.array(unseen_acc_list)
        auc = float(np.trapz(seen_arr, unseen_arr))

        hm_arr  = hmean([seen_arr, unseen_arr], axis=0)
        best_hm_idx = int(np.argmax(hm_arr))
        best_hm  = float(hm_arr[best_hm_idx])

        return {
            "best_seen":   float(seen_arr.max()),
            "best_unseen": float(unseen_arr.max()),
            "best_hm":     best_hm,
            "hm_seen":     float(seen_arr[best_hm_idx]),
            "hm_unseen":   float(unseen_arr[best_hm_idx]),
            "AUC":         auc,
        }


def threshold_with_feasibility(logits, seen_mask, threshold, feasibility):
    score  = logits.clone()
    mask   = (feasibility >= threshold).float()
    score  = score * (mask + seen_mask.float())
    return score


def run_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    dset_root = DATASET_PATHS[args.dataset]
    dset = CompositionDataset(dset_root, phase=args.split,
                              split="compositional-split-natural",
                              open_world=args.open_world)

    # Load checkpoint
    clf_path = args.clf_checkpoint or os.path.join(
        PROJECT_ROOT, "output", args.dataset, "clf", "classifier_final.pt")
    print(f"Loading classifier from {clf_path} ...")
    ckpt = torch.load(clf_path, map_location="cpu", weights_only=False)

    adapter    = FeatureAdapter(args.feat_dim, ratio=args.adapter_ratio).to(device)
    classifier = CompositionClassifier(args.feat_dim, len(dset.pairs)).to(device)
    adapter.load_state_dict(ckpt["adapter"])
    classifier.load_state_dict(ckpt["classifier"])
    adapter.eval(); classifier.eval()

    # Load pre-extracted test features
    cache_dir  = os.path.join(PROJECT_ROOT, "clip_feats", args.dataset)
    feats_path = os.path.join(cache_dir, f"image_feats_{args.split}.pt")
    label_path = os.path.join(cache_dir, f"image_labels_{args.split}.pt")
    assert os.path.exists(feats_path), \
        f"Test features not found. Run extract_features.py first."
    feats  = torch.load(feats_path,  weights_only=True)
    labels = torch.load(label_path, weights_only=True)

    # Build pair_gt
    meta     = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=False)
    pair2idx = meta["pair2idx"]
    attrs    = meta["attrs"]
    objs     = meta["objs"]
    pair_gt  = []
    for i in range(labels.size(0)):
        a = attrs[labels[i, 0].item()]
        o = objs[labels[i, 1].item()]
        pair_gt.append(dset.pair2idx.get((a, o), 0))
    pair_gt  = torch.tensor(pair_gt, dtype=torch.long)
    attr_gt  = labels[:, 0]
    obj_gt   = labels[:, 1]

    # Forward pass
    all_scores = []
    with torch.no_grad():
        for i in tqdm(range(0, feats.size(0), args.eval_batch_size),
                      desc="Scoring"):
            x_batch  = feats[i:i+args.eval_batch_size].to(device)
            x_a      = F.normalize(adapter(x_batch), dim=-1)
            logits   = classifier(x_a)
            all_scores.append(logits.cpu())
    all_scores = torch.cat(all_scores)   # [N, n_pairs]

    # Open-world feasibility masking
    if args.open_world and args.threshold is not None:
        feas_data = torch.load(
            FEASIBILITY_PATHS[args.dataset], map_location="cpu", weights_only=True)
        feas_vec = feas_data["feasibility"]
        evaluator = Evaluator(dset)
        all_scores = threshold_with_feasibility(
            all_scores, evaluator.seen_mask, args.threshold, feas_vec)

    evaluator = Evaluator(dset)
    stats = evaluator.evaluate(all_scores, attr_gt, obj_gt, pair_gt)

    mode = "OPEN" if args.open_world else "CLOSED"
    print(f"\n=== {args.dataset.upper()} | {args.split.upper()} | {mode} WORLD ===")
    print(f"  Seen:    {stats['best_seen']*100:.2f}")
    print(f"  Unseen:  {stats['best_unseen']*100:.2f}")
    print(f"  HM:      {stats['best_hm']*100:.2f}")
    print(f"  AUC:     {stats['AUC']*100:.2f}")
    return stats


if __name__ == "__main__":
    args = get_args()
    if args.config:
        load_args(args.config, args)
    run_eval(args)
