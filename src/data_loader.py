import os
import torch
import pyvista as pv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, NormalizeScale, FixedPoints
from torch_cluster import knn
from torch.utils.data import random_split
from torch.utils.data import RandomSampler

from transformations import (
    RobustCanonicalAlignment,
    MolarBandExpansion,
    IncisalArchExpansion,
    JawShear,
    InterMolarGapStretch
)


class SpatialPatchSampler:
    """
    Extracts a spatially coherent patch instead of random global sample.
    At training time this matches exactly what patch inference does.

    Strategy: pick a random center point, take the N nearest neighbors.
    This gives a spatially connected region at full local density.
    """

    def __init__(self, num_points=8192):
        self.num_points = num_points

    def __call__(self, data):
        pos = data.pos
        N = pos.shape[0]

        if N <= self.num_points:
            # Mesh is smaller than patch — use all points, pad if needed
            return data

        # Pick a random center — bias toward boundary-rich regions
        # by occasionally picking a point with high local variance
        if torch.rand(1).item() < 0.5:
            # Random center anywhere
            center_idx = torch.randint(0, N, (1,)).item()
        else:
            # Pick from the middle X band (where boundaries tend to be)
            mid_mask = (pos[:, 0].abs() < pos[:, 0].abs().median())
            mid_idx = mid_mask.nonzero(as_tuple=False).view(-1)
            center_idx = mid_idx[torch.randint(0, len(mid_idx), (1,))].item()

        center = pos[center_idx]

        # Take N nearest neighbors to center — spatially coherent patch
        dists = (pos - center).norm(dim=1)
        _, idx = dists.topk(self.num_points, largest=False)

        # Apply to all attributes
        data.pos = pos[idx]
        if data.x is not None:
            data.x = data.x[idx]
        if data.y is not None:
            data.y = data.y[idx]

        return data


class ComputeNormalsFromPos:
    def __init__(self, k=10):
        self.k = k

    def __call__(self, data):
        pos = data.pos
        N = pos.shape[0]
        batch = torch.zeros(N, dtype=torch.long)

        edge_index = knn(pos, pos, k=self.k, batch_x=batch, batch_y=batch)

        src, dst = edge_index[0], edge_index[1]
        neighbors = pos[src] - pos[dst]

        cov = torch.zeros(N, 3, 3)
        cov.scatter_add_(
            0,
            dst.view(-1, 1, 1).expand(-1, 3, 3),
            neighbors.unsqueeze(2) * neighbors.unsqueeze(1)
        )

        _, eigvecs = torch.linalg.eigh(cov)
        normals = eigvecs[:, :, 0]
        norms = normals.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normals = normals / norms

        # pos is already normalized by NormalizeScale — safe to include
        data.x = torch.cat([pos, normals], dim=-1)   # [N, 6]
        return data


def clean_labels(pos, y, k=20):
    """
    Two passes:
    1. Majority vote — each point takes the most common label
       among its K nearest neighbors. Removes isolated streaks.
    2. Border enforcement — any point where red and white are
       direct neighbors gets relabeled to black (border).
       Enforces the anatomical rule that gum never touches tooth.
    """
    N = pos.shape[0]
    batch = torch.zeros(N, dtype=torch.long)

    # ── Pass 1: Majority vote to remove streaks ──────────────────────
    edge_index = knn(pos, pos, k=k, batch_x=batch, batch_y=batch)
    src, dst = edge_index[0], edge_index[1]

    # Count votes per class for each point
    clean_y = y.clone()
    for c in range(3):
        votes = (y[src] == (c + 1)).float()   # y is 1-indexed
        vote_count = torch.zeros(N).scatter_add_(0, dst, votes)
        # Only relabel if this class has a strong majority (>60%)
        majority = vote_count / k
        strong_majority = majority > 0.6
        clean_y[strong_majority] = c + 1

    # ── Pass 2: Border enforcement ───────────────────────────────────
    # Find any point where a red neighbor and white neighbor coexist
    # That point and its ambiguous neighbors become border (black=2)
    has_red_neighbor = torch.zeros(N, dtype=torch.bool)
    has_white_neighbor = torch.zeros(N, dtype=torch.bool)

    is_red = (clean_y == 1)   # gum
    is_white = (clean_y == 3)   # tooth

    # Scatter neighbor labels to each point
    red_signal = is_red[src].float()
    white_signal = is_white[src].float()

    red_count = torch.zeros(N).scatter_add_(0, dst, red_signal)
    white_count = torch.zeros(N).scatter_add_(0, dst, white_signal)

    has_red_neighbor = red_count > 0
    has_white_neighbor = white_count > 0

    # Points that have BOTH red and white neighbors are on the border
    # Force them to black regardless of their current label
    conflict_mask = has_red_neighbor & has_white_neighbor
    clean_y[conflict_mask] = 2   # border (black)

    # Also force any red point that directly touches white to border
    is_red_touching_white = is_red & has_white_neighbor
    is_white_touching_red = is_white & has_red_neighbor
    clean_y[is_red_touching_white] = 2
    clean_y[is_white_touching_red] = 2

    return clean_y


class DentalDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.root = os.path.abspath(root)
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Directory not found: {self.root}")

        self.files = [f for f in os.listdir(self.root) if f.endswith('.ply')]
        print(
            f"--- Dataset initialized: Found {len(self.files)} .ply files in {self.root} ---")

    def len(self):
        return len(self.files)

    def get(self, idx):
        path = os.path.join(self.root, self.files[idx])
        mesh = pv.read(path)

        pos = torch.from_numpy(mesh.points).float()
        colors = (mesh.point_data.get('colors') or
                  mesh.point_data.get('RGB') or
                  mesh.point_data.get('RGBA'))

        # Labels from color
        is_red = (colors[:, 0] > 180) & (
            colors[:, 1] < 100) & (colors[:, 2] < 100)
        is_black = (colors[:, 0] < 80) & (
            colors[:, 1] < 80) & (colors[:, 2] < 80)
        is_white = (colors[:, 0] > 180) & (
            colors[:, 1] > 180) & (colors[:, 2] > 180)

        y = torch.zeros(len(pos), dtype=torch.long)
        y[is_red] = 1
        y[is_black] = 2
        y[is_white] = 3

        unknown_mask = (y == 0)
        known_mask = ~unknown_mask
        if unknown_mask.any() and known_mask.any():
            assign_idx = knn(pos[known_mask], pos[unknown_mask], k=1)
            y[unknown_mask] = y[known_mask][assign_idx[1]]
        elif unknown_mask.all():
            y = torch.ones(len(pos), dtype=torch.long)

        # After KNN label propagation for unknown points...
        y = clean_labels(pos, y, k=20)

        return Data(pos=pos, y=y)


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data = self.subset[idx]
        if self.transform:
            data = self.transform(data)
        return data


def get_dental_loaders(data_path, batch_size=2, num_points=8192):

    train_transform = Compose([
        RobustCanonicalAlignment(),
        MolarBandExpansion(),
        IncisalArchExpansion(),
        JawShear(),
        InterMolarGapStretch(),
        NormalizeScale(),
        SpatialPatchSampler(num_points),
        ComputeNormalsFromPos(k=10),   # normals in final aligned+sampled space
    ])

    val_transform = Compose([
        RobustCanonicalAlignment(),
        NormalizeScale(),
        SpatialPatchSampler(num_points),
        ComputeNormalsFromPos(k=10),   # same for val
    ])

    dataset = DentalDataset(root=data_path)
    total_count = len(dataset)
    train_size = int(0.9 * total_count)
    val_size = total_count - train_size

    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(
        f"--- Split Complete: {len(train_subset)} Train | {len(val_subset)} Val ---")

    train_set = TransformSubset(train_subset, transform=train_transform)
    val_set = TransformSubset(val_subset,   transform=val_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=RandomSampler(train_set, num_samples=len(
            train_set) * 2, replacement=True),
        drop_last=True
    )
    val_loader = DataLoader(val_set,   batch_size=1,
                            shuffle=False, drop_last=False)

    return train_loader, val_loader, None
