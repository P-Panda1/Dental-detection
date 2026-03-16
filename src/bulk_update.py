import os
import torch
import pyvista as pv
import numpy as np
import meshio
from tqdm import tqdm
from torch_cluster import knn
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, FixedPoints, NormalizeScale

from model import DentalMetricDGCNN
from config import config
from transformations import RobustCanonicalAlignment


def bulk_label_data(input_dir="../data/unlabeled", output_dir="../data/labeled"):
    device = config.DEVICE
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load model
    model = DentalMetricDGCNN(
        k=config.K_NEIGHBORS,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.EMBEDDING_DIM
    ).to(device)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: No best_model.pth found in {config.CHECKPOINT_DIR}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"--- Model loaded from epoch {checkpoint['epoch']} ---")

    # 2. Same pipeline as DentalDataset
    align = RobustCanonicalAlignment()
    sampler = Compose(
        [FixedPoints(config.NUM_POINTS_GLOBAL), NormalizeScale()])

    files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]
    print(f"Found {len(files)} unlabeled files. Starting inference...")

    for filename in tqdm(files, desc="Labeling Jaws"):
        path = os.path.join(input_dir, filename)

        mesh = pv.read(path)
        mesh = mesh.compute_normals(
            cell_normals=False, point_normals=True, flip_normals=True
        )

        pos = torch.from_numpy(mesh.points).float()
        normals = torch.from_numpy(mesh['Normals']).float()
        x = torch.cat([pos, normals], dim=-1)

        raw_data = Data(pos=pos, x=x)

        aligned_data = align(raw_data.clone())
        low_res_data = sampler(aligned_data.clone())

        low_res_data = low_res_data.to(device)
        low_res_data.y = torch.zeros(
            low_res_data.pos.shape[0], dtype=torch.long
        ).to(device)

        with torch.no_grad():
            logits = model(low_res_data)
            low_res_preds = torch.argmax(logits, dim=1).cpu()

        assign_idx = knn(low_res_data.pos.cpu(), aligned_data.pos, k=1)
        high_res_preds = low_res_preds[assign_idx[1]].numpy()

        points = mesh.points
        faces = pv.read(path).faces.reshape(-1, 4)[:, 1:]

        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[high_res_preds == 0] = [255, 0,   0]
        colors[high_res_preds == 1] = [0,   0,   0]
        colors[high_res_preds == 2] = [255, 255, 255]

        out_mesh = meshio.Mesh(
            points,
            [("triangle", faces)],
            point_data={
                "red":   colors[:, 0],
                "green": colors[:, 1],
                "blue":  colors[:, 2],
                "label": high_res_preds.astype(np.float32)
            }
        )

        new_filename = filename.replace(".ply", "_labeled.ply")
        output_path = os.path.join(output_dir, new_filename)
        out_mesh.write(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    bulk_label_data()
