import os
import yaml
import torch
import random
import numpy as np


def load_args(filename, args):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    for group in data.values():
        for k, v in group.items():
            setattr(args, k, v)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def format_result(stats):
    keys = ['best_seen', 'best_unseen', 'best_hm', 'AUC']
    return ' | '.join(f'{k}: {stats[k]*100:.2f}' for k in keys if k in stats)
