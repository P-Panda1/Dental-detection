import os
import torch
import pyvista as pv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, NormalizeScale, FixedPoints
from torch.utils.data import random_split

# Import from our other file
from transforms import (
    RobustCanonicalAlignment,
    RandomizedDentalBandStretch,
    AnatomicalDentalStretch
)


class DentalDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.root = root
        self.files = [f for f in os.listdir(root) if f.endswith('.ply')]

    def len(self):
        return len(self.files)

    def get(self, idx):
        path = os.path.join(self.root, self.files[idx])
        mesh = pv.read(path)

        pos = torch.from_numpy(mesh.points).float()

        # Color Map: Red=1 (Gum), Black=2 (Border), White=3 (Teeth)
        # Using .get() to avoid KeyError if array is named differently
        colors = mesh.point_data.get('colors') or mesh.point_data.get('RGBA')
        y = torch.zeros(len(pos), dtype=torch.long)

        if colors is not None:
            # We use a threshold of 180 to handle anti-aliasing in paint
            y[(colors[:, 0] > 180) & (colors[:, 1] < 100)
              & (colors[:, 2] < 100)] = 1
            y[(colors[:, 0] < 80) & (colors[:, 1] < 80) & (colors[:, 2] < 80)] = 2
            y[(colors[:, 0] > 180) & (colors[:, 1] > 180)
              & (colors[:, 2] > 180)] = 3

        return Data(pos=pos, y=y)


def get_dental_loaders(data_path, batch_size=2, num_points=8192):
    # 1. Pipeline for Training (With Heavy Augmentation)
    train_transform = Compose([
        RobustCanonicalAlignment(),
        RandomizedDentalBandStretch(),
        AnatomicalDentalStretch(),
        # Ensures all meshes in a batch have same size
        FixedPoints(num_points),
        NormalizeScale()
    ])

    # 2. Pipeline for Validation (No stretching, just alignment)
    val_transform = Compose([
        RobustCanonicalAlignment(),
        FixedPoints(num_points),
        NormalizeScale()
    ])

    dataset = DentalDataset(root=data_path)

    # 3. Split: 3 train, 1 val, 1 test
    train_set, val_set, test_set = random_split(dataset, [4, 1, 0])

    # Assign transforms to the underlying dataset objects
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform
    test_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


# Example Usage
# if __name__ == "__main__":
#     t_loader, v_loader, _ = get_dental_loaders("../data/annotated/")
#     for batch in t_loader:
#         print(
#             f"Batch loaded with {batch.num_graphs} jaws and {batch.pos.shape} points.")
#         break
