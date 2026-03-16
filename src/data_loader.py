import os
import torch
import pyvista as pv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, NormalizeScale, FixedPoints
from torch.utils.data import random_split

# Import from our other file
from transformations import (
    RobustCanonicalAlignment,
    RandomizedDentalBandStretch,
    AnatomicalDentalStretch,
    RandomBlobRemoval
)


def custom_batched_knn_graph(pos, k, batch=None):
    """
    A pure PyTorch implementation of batched KNN graph generation.
    Bypasses the need for torch_cluster and avoids OOM errors by processing per-batch.
    """
    # 1. Handle the case where batch is None (single graph)
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    edge_indices = []

    # Process each graph in the batch separately
    for b in batch.unique():
        mask = batch == b
        pos_b = pos[mask]

        # 2. Skip if a batch index is somehow empty
        if pos_b.size(0) == 0:
            continue

        # Calculate pairwise distances for just this point cloud
        dist = torch.cdist(pos_b, pos_b)

        # Get top k nearest neighbors
        actual_k = min(k + 1, pos_b.size(0))
        _, col_b = dist.topk(actual_k, dim=1, largest=False, sorted=False)

        # Create row indices
        row_b = torch.arange(pos_b.size(
            0), device=pos.device).view(-1, 1).expand(-1, actual_k)

        # Map the local batch indices back to the global indices
        orig_indices = torch.where(mask)[0]
        col = orig_indices[col_b.reshape(-1)]
        row = orig_indices[row_b.reshape(-1)]

        edge_indices.append(torch.stack([col, row], dim=0))

    # 3. Final safeguard in case no edges were generated at all
    if len(edge_indices) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)

    return torch.cat(edge_indices, dim=1)


class DentalDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.root = os.path.abspath(root)  # Ensure absolute
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
        colors = mesh.point_data.get('colors') or mesh.point_data.get(
            'RGB') or mesh.point_data.get('RGBA')

        # 1. Create Masks for "Certain" Points
        is_red = (colors[:, 0] > 180) & (
            colors[:, 1] < 100) & (colors[:, 2] < 100)
        is_black = (colors[:, 0] < 80) & (
            colors[:, 1] < 80) & (colors[:, 2] < 80)
        is_white = (colors[:, 0] > 180) & (
            colors[:, 1] > 180) & (colors[:, 2] > 180)

        # 2. Assign initial labels (0 = unknown)
        y = torch.zeros(len(pos), dtype=torch.long)
        y[is_red] = 1
        y[is_black] = 2
        y[is_white] = 3

        # 3. Handle Neighbors for Unknown Points
        unknown_mask = (y == 0)
        known_mask = ~unknown_mask

        if unknown_mask.any() and known_mask.any():
            # Find the nearest 'known' point for every 'unknown' point
            # k=1 because we want the single best neighbor match
            assign_idx = custom_batched_knn_graph(
                pos[known_mask], pos[unknown_mask], k=1)

            # Map the neighbor's label to the unknown point
            # assign_idx[0] is index of unknown point, [1] is index of known neighbor
            neighbor_labels = y[known_mask][assign_idx[1]]
            y[unknown_mask] = neighbor_labels
        elif unknown_mask.all():
            # Fallback if the file has NO valid colors
            y = torch.ones(len(pos), dtype=torch.long)

        return Data(pos=pos, y=y)

# Wrapper to apply different transforms to subsets


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # subset[index] already gives us the Data object from DentalDataset
        data = self.subset[index]
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.subset)


def get_dental_loaders(data_path, batch_size=2, num_points=8192):
    # Transforms remain the same
    train_transform = Compose([
        RobustCanonicalAlignment(),
        AnatomicalDentalStretch(),
        RandomBlobRemoval(num_blobs=1, radius=0.05, p=0.9),
        RandomBlobRemoval(num_blobs=5, radius=0.05, p=0.4),
        RandomBlobRemoval(num_blobs=3, radius=0.05, p=0.8),
        RandomBlobRemoval(num_blobs=1, radius=0.1, p=0.8),
        RandomBlobRemoval(num_blobs=1, radius=0.3, p=0.8),
        RandomBlobRemoval(num_blobs=1, radius=0.5, p=0.3)
    ])

    val_transform = Compose([
        RobustCanonicalAlignment(),
        FixedPoints(num_points),
        NormalizeScale()
    ])

    dataset = DentalDataset(root=data_path)
    total_count = len(dataset)

    # Calculate 90/10 split sizes
    train_size = int(0.9 * total_count)
    val_size = total_count - train_size

    # Split the base dataset randomly
    # Generator with seed ensures reproducibility if you need to debug
    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(
        f"--- Split Complete: {len(train_subset)} Train | {len(val_subset)} Val ---")

    # Wrap with respective transforms
    train_set = TransformSubset(train_subset, transform=train_transform)
    val_set = TransformSubset(val_subset, transform=val_transform)

    # Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, None
