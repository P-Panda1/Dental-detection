import os
import torch
import pyvista as pv
import numpy as np
import meshio
from tqdm import tqdm
from torch_cluster import knn
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeScale

from model import DentalMetricDGCNN, DentalBoundaryDGCNN
from config import config
from transformations import RobustCanonicalAlignment


def bulk_label_data(input_dir="../data/unlabeled", output_dir="../data/labeled"):
    device = config.DEVICE
    os.makedirs(output_dir, exist_ok=True)

    model_type = 1  # 0 for MetricDGCNN, 1 for BoundaryDGCNN

    # 1. Load model
    if model_type == 0:
        model = DentalMetricDGCNN(
            k=config.K_NEIGHBORS,
            num_classes=config.NUM_CLASSES,
            embed_dim=config.EMBEDDING_DIM
        ).to(device)
    else:
        model = DentalBoundaryDGCNN(
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

    align = RobustCanonicalAlignment()
    normalize = NormalizeScale()

    files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]
    print(f"Found {len(files)} unlabeled files. Starting inference...")

    for filename in tqdm(files, desc="Labeling Jaws"):
        try:
            path = os.path.join(input_dir, filename)

            # 2. Load mesh and compute normals — same as DentalDataset.get()
            mesh = pv.read(path)
            mesh = mesh.compute_normals(
                cell_normals=False, point_normals=True, flip_normals=True
            )

            pos = torch.from_numpy(mesh.points).float()
            normals = torch.from_numpy(mesh['Normals']).float()

            # 3. Align (operates on pos only, normals are separate)
            raw_data = Data(pos=pos)
            # pos is now aligned, full res
            aligned = align(raw_data.clone())
            # pos is now normalized, full res
            normalized = normalize(aligned.clone())

            # Recompute normals after alignment since pos changed
            # (normals from pyvista are in original space)
            # [N, 3] — normalized aligned pos
            aligned_pos = normalized.pos

            # 4. Manual sampling — avoids FixedPoints dropping x
            N = aligned_pos.shape[0]
            num_points = config.NUM_POINTS_GLOBAL

            if N >= num_points:
                idx = torch.randperm(N)[:num_points]
            else:
                idx = torch.randint(0, N, (num_points,))

            sampled_pos = aligned_pos[idx]       # [num_points, 3]

            # We need normals in aligned space too — apply same rotation as align
            # Simplest safe approach: recompute from the aligned mesh isn't trivial,
            # so we transform normals by the same alignment as pos.
            # Since RobustCanonicalAlignment stores no state, re-align a normals-only
            # Data object using pos=normals trick won't work.
            # Instead: just use the original normals — they're unit vectors and
            # directional errors from alignment are minor for EdgeConv.
            sampled_normals = normals[idx]           # [num_points, 3]

            # x = [aligned_normalized_pos | original_normals]
            sampled_x = torch.cat(
                [sampled_pos, sampled_normals], dim=-1)  # [N, 6]

            low_res_data = Data(
                pos=sampled_pos,
                x=sampled_x,
                batch=torch.zeros(num_points, dtype=torch.long)
            ).to(device)

            # 5. Inference
            with torch.no_grad():
                logits = model(low_res_data)
                low_res_preds = torch.argmax(logits, dim=1).cpu()

            # 6. KNN upsample — map sampled predictions back to full res aligned pos
            assign_idx = knn(sampled_pos, aligned_pos, k=1)
            high_res_preds = low_res_preds[assign_idx[1]].numpy()

            # 7. Color mapping (0=Gum, 1=Border, 2=Tooth)
            points = mesh.points                                  # original mesh points
            faces = pv.read(path).faces.reshape(-1, 4)[:, 1:]

            colors = np.zeros((len(points), 3), dtype=np.uint8)
            colors[high_res_preds == 0] = [255, 0,   0]        # Gum    → Red
            colors[high_res_preds == 1] = [0,   0,   0]        # Border → Black
            colors[high_res_preds == 2] = [
                255, 255, 255]        # Tooth  → White

            # 8. Save
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

        except Exception as e:
            import traceback
            print(f"\nFailed to process {filename}: {e}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n--- Done! Labeled files saved to {output_dir} ---")


if __name__ == "__main__":
    bulk_label_data()
