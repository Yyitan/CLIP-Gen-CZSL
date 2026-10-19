"""
Offline CLIP feature extraction — run once per dataset, results cached under
  <project_root>/clip_feats/<dataset_name>/
Files saved:
  image_feats_{train,val,test}.pt   [N, 768] float32, L2-normalised
  image_labels_{train,val,test}.pt  [N, 2]   int64  (attr_idx, obj_idx)
  text_feats_attrs.pt               [n_attrs, 768]
  text_feats_objs.pt                [n_objs,  768]
  text_feats_pairs.pt               [n_pairs, 768]  (seen+unseen, ordered by dset.pairs)
  meta.pt                           dict with attrs/objs/pairs/train_pairs lists

Re-running re-encodes the text banks (cheap) and skips already-cached image feats.
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, './lyt/czsl/CLIP-main')
import clip

sys.path.insert(0, './lyt/czsl/CAMS-main/CAMS')
from dataset import CompositionDataset, transform_image

CLIP_MODEL_PATH = './lyt/czsl/CLIP-main/ViT-L-14.pt'
PROJECT_ROOT = './lyt/czsl/CLIP_Gen_CZSL'

DATASET_PATHS = {
    'mit-states': 'data/mit-states',
    'ut-zappos':  'data/ut-zappos',
    'cgqa':       'data/cgqa',
}


def clean(name):
    """Normalise raw class names for prompts: dots/underscores -> spaces.

    e.g. 'Boots.Ankle' -> 'boots ankle', 'Faux.Fur' -> 'faux fur'.
    """
    return name.replace('.', ' ').replace('_', ' ').strip().lower()


def ATTR_PROMPT(a):    return f"a photo of {clean(a)} object"
def OBJ_PROMPT(o):     return f"a photo of {clean(o)}"
def PAIR_PROMPT(a, o): return f"a photo of {clean(a)} {clean(o)}"


@torch.no_grad()
def extract_image_feats(clip_model, dataset, batch_size=256, device='cuda'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)
    feats, labels = [], []
    for batch in tqdm(loader, desc='  images'):
        imgs = batch[0].to(device)
        f = clip_model.encode_image(imgs).float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
        labels.append(torch.stack([batch[1], batch[2]], dim=1))
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def encode_texts(clip_model, prompts, device='cuda', batch_size=512):
    all_feats = []
    for i in range(0, len(prompts), batch_size):
        tokens = clip.tokenize(prompts[i:i+batch_size]).to(device)
        f = clip_model.encode_text(tokens).float()
        f = f / f.norm(dim=-1, keepdim=True)
        all_feats.append(f.cpu())
    return torch.cat(all_feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASET_PATHS.keys()))
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_dir = os.path.join(PROJECT_ROOT, 'clip_feats', args.dataset)
    os.makedirs(cache_dir, exist_ok=True)

    print('Loading CLIP ViT-L/14 ...')
    clip_model, preprocess = clip.load(CLIP_MODEL_PATH, device=device)
    clip_model.eval()

    dset_root = DATASET_PATHS[args.dataset]

    for phase in ['train', 'val', 'test']:
        feat_path  = os.path.join(cache_dir, f'image_feats_{phase}.pt')
        label_path = os.path.join(cache_dir, f'image_labels_{phase}.pt')
        if os.path.exists(feat_path):
            print(f'  {phase}: image feats already cached, skip')
            continue
        print(f'  Extracting {phase} image features ...')
        dset = CompositionDataset(dset_root, phase=phase,
                                  split='compositional-split-natural')
        dset.transform = preprocess
        feats, labels = extract_image_feats(clip_model, dset, args.batch_size, device)
        torch.save(feats,  feat_path)
        torch.save(labels, label_path)
        print(f'    saved {feats.shape} -> {feat_path}')

    dset = CompositionDataset(dset_root, phase='train',
                              split='compositional-split-natural')

    print('  Encoding attribute text ...')
    torch.save(encode_texts(clip_model, [ATTR_PROMPT(a) for a in dset.attrs], device),
               os.path.join(cache_dir, 'text_feats_attrs.pt'))
    print('  Encoding object text ...')
    torch.save(encode_texts(clip_model, [OBJ_PROMPT(o) for o in dset.objs], device),
               os.path.join(cache_dir, 'text_feats_objs.pt'))
    print('  Encoding pair text ...')
    torch.save(encode_texts(clip_model, [PAIR_PROMPT(a, o) for a, o in dset.pairs], device),
               os.path.join(cache_dir, 'text_feats_pairs.pt'))

    meta = {
        'attrs': dset.attrs, 'objs': dset.objs, 'pairs': dset.pairs,
        'train_pairs': dset.train_pairs, 'val_pairs': dset.val_pairs,
        'test_pairs': dset.test_pairs, 'attr2idx': dset.attr2idx,
        'obj2idx': dset.obj2idx, 'pair2idx': dset.pair2idx,
    }
    torch.save(meta, os.path.join(cache_dir, 'meta.pt'))
    print(f'Done. Cache at {cache_dir}')
    print(f'  attrs={len(dset.attrs)}, objs={len(dset.objs)}, '
          f'pairs={len(dset.pairs)}, train_pairs={len(dset.train_pairs)}')


if __name__ == '__main__':
    main()
