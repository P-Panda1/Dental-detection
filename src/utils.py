import torch
import numpy as np
from torch_geometric.transforms import BaseTransform
from sklearn.decomposition import PCA

class CanonicalAlignment(BaseTransform):
    def __call__(self, data):
        # 1. Centering (Translate to origin)
        pos = data.pos - data.pos.mean(dim=0)
        
        # 2. PCA Alignment
        pca = PCA(n_components=3)
        pca.fit(pos.numpy())
        # Transform points to principal component space
        pos_aligned = torch.from_numpy(pca.transform(pos.numpy())).float()
        
        # 3. Unit Normalization (Scale to fit in a sphere of radius 1)
        # This maintains the relative proportions of teeth vs gum
        max_dist = torch.max(torch.norm(pos_aligned, dim=1))
        data.pos = pos_aligned / max_dist
        
        return data