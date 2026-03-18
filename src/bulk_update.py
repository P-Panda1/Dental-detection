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
from data_loader import ComputeNormalsFromPos


def smooth_upsample(sampled_pos, sampled_logits, full_pos, k=5):
    """
    Interpolates logits from K nearest sampled points for every
    full-res point, weighted by distance. Produces smooth boundaries
    instead of hard Voronoi edges from k=1 KNN copying.
    """
    assign = knn(sampled_pos, full_pos, k=k)
    src, dst = assign[0], assign[1]

    dists = (full_pos[src] - sampled_pos[dst]).norm(dim=1)
    weights = 1.0 / (dists + 1e-8)

    weight_sum = torch.zeros(full_pos.shape[0]).scatter_add_(0, src, weights)
    weights = weights / weight_sum[src]

    full_logits = torch.zeros(full_pos.shape[0], sampled_logits.shape[1])
    for c in range(sampled_logits.shape[1]):
        contrib = weights * sampled_logits[dst, c]
        full_logits[:, c].scatter_add_(0, src, contrib)

    return full_logits


def patch_inference(model, aligned_pos, device,
                    patch_size=8192, num_patches=12, overlap_factor=0.4):
    """
    Runs inference on multiple overlapping spatial patches.
    Patch centers are spread across the jaw so every point
    appears in multiple patches and gets averaged.

    num_patches controls boundary smoothness:
    - More patches = smoother boundaries, slower inference
    - 12 patches is a good balance for a jaw mesh
    """
    compute_normals = ComputeNormalsFromPos(k=10)
    N = aligned_pos.shape[0]
    all_logits = torch.zeros(N, config.NUM_CLASSES)
    all_counts = torch.zeros(N)

    # Spread patch centers evenly along X axis (left → right jaw)
    x_coords = aligned_pos[:, 0]
    x_min = x_coords.min().item()
    x_max = x_coords.max().item()
    centers_x = torch.linspace(x_min, x_max, num_patches)

    # Also add some patches centered on Y axis (front/back jaw)
    y_coords = aligned_pos[:, 1]
    y_min = y_coords.min().item()
    y_max = y_coords.max().item()
    centers_y = torch.linspace(y_min, y_max, num_patches // 2)

    # Build center points: X-spread + Y-spread + random centers
    center_points = []
    for cx in centers_x:
        center_points.append(torch.tensor([cx, 0.0, 0.0]))
    for cy in centers_y:
        center_points.append(torch.tensor([0.0, cy, 0.0]))
    # Add a few random centers for coverage
    rand_idx = torch.randperm(N)[:num_patches // 2]
    for ri in rand_idx:
        center_points.append(aligned_pos[ri])

    for center in center_points:
        center = center.to(aligned_pos.device)

        # Find nearest patch_size points to this center
        dists = (aligned_pos - center).norm(dim=1)
        _, idx = dists.topk(patch_size, largest=False)

        patch_pos = aligned_pos[idx]   # [patch_size, 3]

        # Compute normals in same space as training
        patch_data = Data(pos=patch_pos)
        patch_data = compute_normals(patch_data)
        patch_data.batch = torch.zeros(
            patch_size, dtype=torch.long
        )
        patch_data = patch_data.to(device)

        with torch.no_grad():
            logits = model(patch_data).cpu()   # [patch_size, 3]

        # Points near the patch center get higher weight
        # — they had more context around them
        center_dists = dists[idx]
        max_dist = center_dists.max().clamp(min=1e-8)
        weights = (1.0 - center_dists /
                   max_dist).clamp(min=0.1)  # [patch_size]

        all_logits[idx] += logits * weights.unsqueeze(1)
        all_counts[idx] += weights

    # Handle uncovered points
    uncovered = all_counts == 0
    if uncovered.any():
        covered_pos = aligned_pos[~uncovered]
        covered_logits = all_logits[~uncovered] / \
            all_counts[~uncovered].unsqueeze(1)
        interp = smooth_upsample(covered_pos, covered_logits,
                                 aligned_pos[uncovered], k=5)
        all_logits[uncovered] = interp
        all_counts[uncovered] = 1

    final_logits = all_logits / all_counts.unsqueeze(1).clamp(min=1)
    return final_logits


# How patch count affects quality

# num_patches = 6   → fast, coarser boundaries, some points only seen once
# num_patches = 12  → good balance, most boundary points seen 3-4 times  ← recommended
# num_patches = 20  → slow, very smooth boundaries, diminishing returns after this


def bulk_label_data(input_dir="../data/unlabeled", output_dir="../data/labeled"):
    device = config.DEVICE
    os.makedirs(output_dir, exist_ok=True)

    model_type = 1  # 0 for MetricDGCNN, 1 for BoundaryDGCNN

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

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
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
            mesh = pv.read(path)
            pos = torch.from_numpy(mesh.points).float()

            # Align + normalize — same pipeline as val_transform
            raw_data = Data(pos=pos)
            aligned = align(raw_data.clone())
            normalized = normalize(aligned.clone())
            aligned_pos = normalized.pos               # [N, 3] full res

            # Patch inference — no aggressive downsampling, smooth boundaries
            final_logits = patch_inference(
                model, aligned_pos, device,
                patch_size=config.NUM_POINTS_GLOBAL,
                overlap=0.4,
                num_patches=20
            )
            high_res_preds = torch.argmax(final_logits, dim=1).numpy()

            # Color mapping (0=Gum, 1=Border, 2=Tooth)
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

        except Exception as e:
            import traceback
            print(f"\nFailed to process {filename}: {e}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n--- Done! Labeled files saved to {output_dir} ---")


if __name__ == "__main__":
    bulk_label_data()
