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
    AnatomicalDentalStretch
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

        # Handle colors (RGBA or RGB)
        colors = mesh.point_data.get('colors') or mesh.point_data.get(
            'RGB') or mesh.point_data.get('RGBA')

        y = torch.zeros(len(pos), dtype=torch.long)
        if colors is not None:
            # Labels: 1=Gum, 2=Border, 3=Tooth
            y[(colors[:, 0] > 180) & (colors[:, 1] < 100)
              & (colors[:, 2] < 100)] = 1
            y[(colors[:, 0] < 80) & (colors[:, 1] < 80) & (colors[:, 2] < 80)] = 2
            y[(colors[:, 0] > 180) & (colors[:, 1] > 180)
              & (colors[:, 2] > 180)] = 3

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
    train_transform = Compose([
        RobustCanonicalAlignment(),
        RandomizedDentalBandStretch(),
        AnatomicalDentalStretch(),
        FixedPoints(num_points),
        NormalizeScale()
    ])

    val_transform = Compose([
        RobustCanonicalAlignment(),
        FixedPoints(num_points),
        NormalizeScale()
    ])

    dataset = DentalDataset(root=data_path)

    # Generate indices for the split
    total_count = len(dataset)
    indices = list(range(total_count))

    # Example: 4 for train, 1 for val (if you have 5 files)
    train_indices = indices[:4]
    val_indices = indices[4:5]

    # Create the subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    # Wrap them with their respective transforms
    train_set = TransformSubset(train_subset, transform=train_transform)
    val_set = TransformSubset(val_subset, transform=val_transform)

    # Use the PyG DataLoader (important for batching graphs correctly)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, None
