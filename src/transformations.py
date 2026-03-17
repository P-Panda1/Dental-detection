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
        stretch_vec = torch.tensor([2, 2, 1.0])  # 15% wider molars
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


class RandomBlobRemoval:
    def __init__(self, num_blobs=1, radius=0.1, p=0.5):
        self.num_blobs = num_blobs
        self.radius = radius
        self.p = p  # Probability of the transform executing

    def __call__(self, data):
        # Roll the dice: if random value > p, skip the transform
        if torch.rand(1).item() > self.p:
            return data

        pos = data.pos
        final_mask = torch.ones(
            pos.size(0), dtype=torch.bool, device=pos.device)

        for _ in range(self.num_blobs):
            # Pick a random center
            idx = torch.randint(0, pos.size(0), (1,))
            center = pos[idx]

            # dist > radius keeps points OUTSIDE, creating the hole
            dist = torch.norm(pos - center, dim=1)
            blob_mask = dist > self.radius
            final_mask &= blob_mask

        # Convert to indices. .subgraph() is key for training because it
        # correctly prunes the 'face' (triangles) array so no triangles
        # try to point to deleted vertices.
        keep_indices = final_mask.nonzero(as_tuple=False).view(-1)

        # Safety: if the blob somehow deletes the whole tooth, return original
        if keep_indices.numel() < 10:
            return data

        return data.subgraph(keep_indices)


class MolarBandExpansion(BaseTransform):
    """
    Expands or contracts the molar regions outward along the X axis.
    After PCA alignment, molars sit at negative Y. We find the molar band
    and push those points outward from the jaw centerline.
    """

    def __init__(self, scale_range=(0.9, 1.1), band_y_max=-0.15, p=0.8):
        super().__init__()
        self.scale_range = scale_range
        self.band_y_max = band_y_max   # points below this Y are molars
        self.p = p

    def forward(self, data):
        if torch.rand(1).item() > self.p:
            return data

        pos = data.pos.clone()
        scale = torch.empty(1).uniform_(*self.scale_range).item()

        # Randomize the band cutoff slightly so the model doesn't memorize
        # a hard boundary
        band_limit = self.band_y_max + \
            torch.empty(1).uniform_(-0.08, 0.08).item()
        molar_mask = pos[:, 1] < band_limit

        if molar_mask.sum() < 10:
            return data

        # Expand outward from X=0 (jaw centerline after PCA)
        # Points on the left go further left, right go further right
        pos[molar_mask, 0] *= scale

        # Soft blend at the boundary to avoid a hard seam
        # Points near the band_limit get partial influence
        near_mask = (pos[:, 1] >= band_limit) & (pos[:, 1] < band_limit + 0.1)
        if near_mask.sum() > 0:
            blend = (band_limit + 0.1 - pos[near_mask, 1]) / 0.1  # 0→1
            pos[near_mask, 0] *= (1.0 + blend * (scale - 1.0))

        data.pos = pos
        return data


class IncisalArchExpansion(BaseTransform):
    """
    Expands or contracts the incisor arch — the front teeth region.
    After PCA alignment, incisors sit at positive Y.
    We push them forward/backward along Y and optionally fan them
    outward along X to simulate different arch widths.
    """

    def __init__(self, y_scale_range=(0.9, 1.1),
                 x_fan_range=(0.9, 1.1),
                 band_y_min=0.2, p=0.8):
        super().__init__()
        self.y_scale_range = y_scale_range
        self.x_fan_range = x_fan_range
        self.band_y_min = band_y_min   # points above this Y are incisors
        self.p = p

    def forward(self, data):
        if torch.rand(1).item() > self.p:
            return data

        pos = data.pos.clone()
        y_scale = torch.empty(1).uniform_(*self.y_scale_range).item()
        x_fan = torch.empty(1).uniform_(*self.x_fan_range).item()

        band_limit = self.band_y_min + \
            torch.empty(1).uniform_(-0.08, 0.08).item()
        incisor_mask = pos[:, 1] > band_limit

        if incisor_mask.sum() < 10:
            return data

        # Push incisors forward/backward (protrusion/retrusion)
        pos[incisor_mask, 1] *= y_scale

        # Fan outward from centerline — simulates wide vs narrow arch
        pos[incisor_mask, 0] *= x_fan

        # Soft blend at boundary
        near_mask = (pos[:, 1] <= band_limit) & (pos[:, 1] > band_limit - 0.1)
        if near_mask.sum() > 0:
            blend = (pos[near_mask, 1] - (band_limit - 0.1)) / 0.1
            pos[near_mask, 1] *= (1.0 + blend * (y_scale - 1.0))
            pos[near_mask, 0] *= (1.0 + blend * (x_fan - 1.0))

        data.pos = pos
        return data


class JawShear(BaseTransform):
    """
    Applies a shear transform to simulate different jaw opening angles
    and asymmetric jaw shapes. Two modes:

    - 'lateral': tilts the jaw sideways (one side higher than other)
      simulates asymmetric bite / crossbite
    - 'sagittal': tilts front-to-back (overbite / underbite geometry)
      simulates different vertical jaw relationships
    """

    def __init__(self, shear_range=(-0.1, 0.1), mode='both', p=0.8):
        super().__init__()
        self.shear_range = shear_range
        self.mode = mode   # 'lateral', 'sagittal', or 'both'
        self.p = p

    def forward(self, data):
        if torch.rand(1).item() > self.p:
            return data

        pos = data.pos.clone()

        if self.mode in ('lateral', 'both'):
            # Lateral shear: Z shifts proportionally to X
            # Simulates one side of jaw being higher
            shear_xz = torch.empty(1).uniform_(*self.shear_range).item()
            pos[:, 2] += shear_xz * pos[:, 0]

        if self.mode in ('sagittal', 'both'):
            # Sagittal shear: Z shifts proportionally to Y
            # Simulates overbite / underbite angle
            shear_yz = torch.empty(1).uniform_(*self.shear_range).item()
            pos[:, 2] += shear_yz * pos[:, 1]

        # Re-center Z after shear so NormalizeScale isn't thrown off
        pos[:, 2] -= pos[:, 2].mean()

        data.pos = pos
        return data


class InterMolarGapStretch(BaseTransform):
    """
    Stretches or compresses the gap between the left and right molar
    regions independently. Simulates different jaw widths where one
    side may be wider than the other (asymmetric arch).
    """

    def __init__(self, left_scale_range=(0.9, 1.1),
                 right_scale_range=(0.9, 1.1),
                 p=0.8):
        super().__init__()
        self.left_scale_range = left_scale_range
        self.right_scale_range = right_scale_range
        self.p = p

    def forward(self, data):
        if torch.rand(1).item() > self.p:
            return data

        pos = data.pos.clone()
        left_scale = torch.empty(1).uniform_(*self.left_scale_range).item()
        right_scale = torch.empty(1).uniform_(*self.right_scale_range).item()

        # After PCA, X<0 = left side, X>0 = right side
        left_mask = pos[:, 0] < 0
        right_mask = pos[:, 0] > 0

        # Scale X distance from centerline independently per side
        pos[left_mask,  0] *= left_scale
        pos[right_mask, 0] *= right_scale

        # Soft blend near X=0 to avoid seam
        near_mask = (pos[:, 0].abs() < 0.05)
        if near_mask.sum() > 0:
            blend = pos[near_mask, 0].abs() / 0.05   # 0 at center, 1 at edge
            scale_interp = torch.where(
                pos[near_mask, 0] < 0,
                torch.full_like(blend, left_scale),
                torch.full_like(blend, right_scale)
            )
            pos[near_mask, 0] *= (1.0 + blend * (scale_interp - 1.0))

        data.pos = pos
        return data
