import os
import torch
import pyvista as pv
import numpy as np
import meshio
from tqdm import tqdm
from torch_cluster import knn
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, FixedPoints, NormalizeScale

# Import your custom modules
from model import DentalMetricDGCNN
from config import config
from transformations import RobustCanonicalAlignment


def bulk_label_data(input_dir="../data/unlabeled", output_dir="../data/labeled"):
    device = config.DEVICE

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the Best Model
    model = DentalMetricDGCNN(
        k=config.K_NEIGHBORS,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.GLOBAL_EMBED_DIM // 8
    )
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: No best_model.pth found in {config.CHECKPOINT_DIR}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"--- Model loaded from epoch {checkpoint['epoch']} ---")

    # 2. Setup Transformations
    # Note: We use RobustCanonicalAlignment to ensure the mesh matches
    # the orientation the model was trained on.
    align = RobustCanonicalAlignment()
    sampler = Compose([FixedPoints(8192), NormalizeScale()])

    # 3. Process Files
    files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]
    print(f"Found {len(files)} unlabeled files. Starting inference...")

    for filename in tqdm(files, desc="Labeling Jaws"):
        try:
            # Load original mesh
            path = os.path.join(input_dir, filename)
            original_pv_mesh = pv.read(path)

            # Prepare data for model
            raw_pos = torch.from_numpy(original_pv_mesh.points).float()
            raw_data = Data(pos=raw_pos)

            # Apply same pipeline as training
            aligned_data = align(raw_data.clone())
            low_res_data = sampler(aligned_data.clone())

            # Inference
            with torch.no_grad():
                logits = model(low_res_data)
                low_res_preds = torch.argmax(logits, dim=1)

            # KNN Upsampling (Map back to original high-res points)
            # Use aligned_data.pos for KNN to ensure coordinate matching
            assign_idx = knn(low_res_data.pos, aligned_data.pos, k=1)
            high_res_preds = low_res_preds[assign_idx[1]].numpy()

            # 4. Color Mapping
            points = original_pv_mesh.points
            # Reshape faces for meshio [N, 3]
            faces = original_pv_mesh.faces.reshape(-1, 4)[:, 1:]

            colors = np.zeros((len(points), 3), dtype=np.uint8)
            colors[high_res_preds == 0] = [255, 0, 0]     # Gum
            colors[high_res_preds == 1] = [0, 0, 0]       # Border
            colors[high_res_preds == 2] = [255, 255, 255]  # Tooth

            # 5. Save with Meshio (The "MeshLab-Safe" way)
            mesh = meshio.Mesh(
                points,
                [("triangle", faces)],
                point_data={
                    "red": colors[:, 0],
                    "green": colors[:, 1],
                    "blue": colors[:, 2],
                    "label": high_res_preds.astype(np.float32)
                }
            )

            new_filename = filename.replace(".ply", "_c.ply")
            output_path = os.path.join(output_dir, new_filename)
            mesh.write(output_path)

        except Exception as e:
            print(f"\nFailed to process {filename}: {e}")

    print(f"\n--- Done! Labeled files saved to {output_dir} ---")


if __name__ == "__main__":
    bulk_label_data()
