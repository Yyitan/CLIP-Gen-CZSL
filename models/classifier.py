import torch.nn as nn


class DisentangleHeads(nn.Module):
    """Auxiliary heads that predict attr/obj from a feature.

    Used during generator training: the synthetic feature should decode back
    to the correct primitive labels (disentangle consistency loss).
    """
    def __init__(self, dim=768, n_attr=115, n_obj=245):
        super().__init__()
        self.attr_head = nn.Linear(dim, n_attr)
        self.obj_head  = nn.Linear(dim, n_obj)

    def forward(self, x):
        return self.attr_head(x), self.obj_head(x)


class CompositionClassifier(nn.Module):
    """Linear classifier over all composition pairs for Stage B."""
    def __init__(self, dim=768, n_pairs=1962):
        super().__init__()
        self.head = nn.Linear(dim, n_pairs)

    def forward(self, x):
        return self.head(x)   # [B, n_pairs]
