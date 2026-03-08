import torch
import numpy as np
from torch_geometric.transforms import BaseTransform
from sklearn.decomposition import PCA


class CanonicalAlignment(BaseTransform):
    # Rename __call__ to forward
    def forward(self, data):
        # 1. Centering
        pos = data.pos - data.pos.mean(dim=0)

        # 2. PCA Alignment
        pca = PCA(n_components=3)
        pca.fit(pos.numpy())
        pos_aligned = torch.from_numpy(pca.transform(pos.numpy())).float()

        # 3. Unit Normalization
        max_dist = torch.max(torch.norm(pos_aligned, dim=1))
        data.pos = pos_aligned / max_dist

        return data


class RobustCanonicalAlignment(BaseTransform):
    def forward(self, data):
        # 1. Center at origin
        pos = data.pos - data.pos.mean(dim=0)

        # 2. PCA Alignment
        pca = PCA(n_components=3)
        pca.fit(pos.numpy())
        pos_aligned = torch.from_numpy(pca.transform(pos.numpy())).float()

        # 3. Flip Correction (The "Incisor Check")
        # In a standard jaw, there are more points in the 'back' (molars)
        # than the 'front' (incisors) along the depth axis.
        # We force the 'thicker' part of the arch to be at negative Y.
        if pos_aligned[:, 1].mean() > 0:
            pos_aligned[:, 1] *= -1

        # 4. Unit Normalization
        max_dist = torch.max(torch.norm(pos_aligned, dim=1))
        data.pos = pos_aligned / max_dist
        return data


class LocalDentalStretch(BaseTransform):
    # Rename __call__ to forward
    def forward(self, data):
        # 1. Random anchor point
        idx = torch.randint(0, data.pos.size(0), (1,))
        anchor = data.pos[idx]

        # 2. Radius of Influence
        sigma = torch.empty(1).uniform_(0.2, 0.6)

        # 3. Distance calculation
        dist = torch.norm(data.pos - anchor, dim=1)

        # 4. Gaussian weight
        weights = torch.exp(-dist**2 / (2 * sigma**2)).unsqueeze(1)

        # 5. Local stretch (1.2x on X axis)
        stretch = torch.FloatTensor([1.2, 1.0, 1.0])

        # 6. Apply deformation
        stretched_pos = data.pos * stretch
        data.pos = data.pos + weights * (stretched_pos - data.pos)

        # Adding this for your visualization script to find the anchor
        data.anchor = anchor

        return data


class MultiPointDentalStretch(BaseTransform):
    def __init__(self, num_anchors=10):
        super().__init__()
        self.num_anchors = num_anchors

    def forward(self, data):
        # 1. Select 10 random anchor indices
        indices = torch.randint(0, data.pos.size(0), (self.num_anchors,))
        anchors = data.pos[indices]  # Shape: [10, 3]

        # 2. Randomize Sigma and Stretch for each anchor to get maximum variety
        # Keeping sigma smaller (0.1 - 0.2) helps keep the 10 points from merging into one big blob
        sigma = torch.empty(self.num_anchors).uniform_(0, 0.05)

        # 3. Calculate distance from ALL points to ALL anchors
        # data.pos is [N, 3], anchors is [10, 3]
        # dists shape will be [N, 10]
        dists = torch.cdist(data.pos, anchors)

        # 4. Calculate Gaussian weights for each anchor and combine
        # We use the max weight at each point so they don't "stack" infinitely
        weights = torch.exp(-dists**2 / (2 * sigma**2))
        combined_weights, _ = torch.max(weights, dim=1, keepdim=True)

        # 5. Define the stretch factor
        # (Using a slightly stronger stretch since the influence area is smaller)
        stretch = torch.FloatTensor([1.3, 1.0, 1.0])

        # 6. Apply deformation
        stretched_pos = data.pos * stretch
        data.pos = data.pos + combined_weights * (stretched_pos - data.pos)

        # Store anchors for visualization
        data.anchors = anchors

        return data


class AnatomicalDentalStretch(BaseTransform):
    def __init__(self, num_anchors=4):  # 4 points: 2 molars, 2 incisors
        super().__init__()
        self.num_anchors = num_anchors

    def forward(self, data):
        # 1. Pick anchors (You can even hardcode zones if you use PCA first)
        indices = torch.randint(0, data.pos.size(0), (self.num_anchors,))
        anchors = data.pos[indices]

        # 2. Use a smaller sigma (0.05 - 0.1) so only the specific tooth is affected
        sigma = 0.08
        dists = torch.cdist(data.pos, anchors)
        weights = torch.exp(-dists**2 / (2 * sigma**2))
        combined_weights, _ = torch.max(weights, dim=1, keepdim=True)

        # 3. RELATIVE STRETCH Logic
        # Instead of pos * 1.2, we do: anchor + (pos - anchor) * 1.2
        # We pick one primary anchor to scale relative to
        # (For multi-point, we use the nearest anchor for each point)
        nearest_anchor_idx = torch.argmin(dists, dim=1)
        target_anchors = anchors[nearest_anchor_idx]

        # 4. Apply stretch to the 'offset' from the tooth center
        # This makes the tooth wider without shifting the whole jaw
        stretch_vec = torch.tensor([1.5, 1.0, 1.0])  # 15% wider molars
        offset = data.pos - target_anchors
        stretched_offset = offset * stretch_vec

        # 5. Blend: Only the points near the anchor get the stretched offset
        data.pos = target_anchors + offset + \
            (combined_weights * (stretched_offset - offset))

        return data


class RandomizedDentalBandStretch(BaseTransform):
    def __init__(self, molar_y_base=-0.3, incisor_y_base=0.5):
        super().__init__()
        self.molar_y_base = molar_y_base
        self.incisor_y_base = incisor_y_base

    def forward(self, data):
        pos = data.pos.clone()
        y_coords = pos[:, 1]

        # 1. Randomize the Band Limits (shift lines by +/- 0.1)
        molar_limit = self.molar_y_base + \
            torch.empty(1).uniform_(-0.1, 0.1).item()
        incisor_limit = self.incisor_y_base + \
            torch.empty(1).uniform_(-0.1, 0.1).item()

        # 2. Randomize Stretch Magnitudes (from 0.95 to 1.20)
        # This allows the jaw to sometimes get NARROWER (contraction) or WIDER
        m_scale = torch.empty(1).uniform_(0.95, 1.05).item()
        i_scale = torch.empty(1).uniform_(0.95, 1.15).item()

        # 3. Apply Molar Transformation
        molar_mask = y_coords < molar_limit
        pos[molar_mask, 0] *= m_scale  # Stretch X

        # 4. Apply Incisor Transformation
        incisor_mask = y_coords > incisor_limit
        pos[incisor_mask, 1] *= i_scale  # Stretch Y (Depth)

        # 5. Optional: Global Width Jitter
        # Slightly scales the entire arch width to prevent 'memorizing' the PCA scale
        global_w = torch.empty(1).uniform_(0.98, 1.02).item()
        pos[:, 0] *= global_w

        data.pos = pos
        # Store for visualization if needed
        data.molar_line = molar_limit
        data.incisor_line = incisor_limit

        return data
