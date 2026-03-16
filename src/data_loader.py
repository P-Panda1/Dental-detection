import os
import torch
import pyvista as pv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, NormalizeScale, FixedPoints
from torch_cluster import knn
from torch.utils.data import random_split

# Import from our other file
from transformations import (
    RobustCanonicalAlignment,
    RandomizedDentalBandStretch,
    AnatomicalDentalStretch,
    RandomBlobRemoval
)


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

        # Calculate normals (crucial for finding sharp dental edges)
        # point_normals=True calculates orientation for every point
        mesh = mesh.compute_normals(
            cell_normals=False, point_normals=True, flip_normals=True)
        normals = torch.from_numpy(mesh['Normals']).float()

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
            assign_idx = knn(pos[known_mask], pos[unknown_mask], k=1)

            # Map the neighbor's label to the unknown point
            # assign_idx[0] is index of unknown point, [1] is index of known neighbor
            neighbor_labels = y[known_mask][assign_idx[1]]
            y[unknown_mask] = neighbor_labels
        elif unknown_mask.all():
            # Fallback if the file has NO valid colors
            y = torch.ones(len(pos), dtype=torch.long)

        features = torch.cat([pos, normals], dim=-1)

        return Data(pos=pos, x=features, y=y)

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
        RandomBlobRemoval(num_blobs=10, radius=0.05, p=0.8),
        RandomBlobRemoval(num_blobs=3, radius=0.1, p=0.8),
        RandomBlobRemoval(num_blobs=2, radius=0.3, p=0.8),
        RandomBlobRemoval(num_blobs=1, radius=0.4, p=0.8),
        NormalizeScale(),
        FixedPoints(num_points)
    ])

    val_transform = Compose([
        RobustCanonicalAlignment(),
        NormalizeScale(),
        FixedPoints(num_points)
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
