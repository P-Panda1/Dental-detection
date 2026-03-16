import os
import torch
import pyvista as pv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, NormalizeScale, FixedPoints
from torch_cluster import knn
from torch.utils.data import random_split

from transformations import (
    RobustCanonicalAlignment,
    RandomizedDentalBandStretch,
    AnatomicalDentalStretch,
    RandomBlobRemoval
)


class ComputeNormalsFromPos:
    """
    Recomputes normals from current (post-alignment) pos via PCA on
    local neighborhood, then sets x = normals only (no absolute XYZ).
    Stripping absolute position forces the model to learn local geometry
    rather than memorizing spatial location.
    """

    def __init__(self, k=10):
        self.k = k

    def __call__(self, data):
        pos = data.pos                                        # [N, 3]
        batch = torch.zeros(pos.shape[0], dtype=torch.long)

        # knn returns [2, N*k]: row0=query idx, row1=source idx
        edge_index = knn(pos, pos, k=self.k,
                         batch_x=batch, batch_y=batch)

        normals = torch.zeros_like(pos)
        for i in range(pos.shape[0]):
            neighbor_idx = edge_index[1][edge_index[0] == i]
            neighbors = pos[neighbor_idx] - pos[i]
            if neighbors.shape[0] < 3:
                continue
            cov = neighbors.T @ neighbors
            _, eigvecs = torch.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]   # smallest eigenvalue → normal

        # Unit normalize
        norms = normals.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normals = normals / norms

        # x = normals only — no absolute XYZ position leak
        data.x = normals                                        # [N, 3]
        return data


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

        # No x here — normals computed AFTER alignment in the transform pipeline
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
        AnatomicalDentalStretch(),
        RandomBlobRemoval(num_blobs=10, radius=0.05, p=0.8),
        RandomBlobRemoval(num_blobs=3,  radius=0.1,  p=0.8),
        RandomBlobRemoval(num_blobs=2,  radius=0.3,  p=0.8),
        RandomBlobRemoval(num_blobs=1,  radius=0.4,  p=0.8),
        NormalizeScale(),
        FixedPoints(num_points),
        ComputeNormalsFromPos(k=10),   # normals in final aligned+sampled space
    ])

    val_transform = Compose([
        RobustCanonicalAlignment(),
        NormalizeScale(),
        FixedPoints(num_points),
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

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader = DataLoader(val_set,   batch_size=1,
                            shuffle=False, drop_last=False)

    return train_loader, val_loader, None
