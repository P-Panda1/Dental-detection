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
                    patch_size=8192, num_patches=12):
    compute_normals = ComputeNormalsFromPos(k=10)
    N = aligned_pos.shape[0]
    all_logits = torch.zeros(N, config.NUM_CLASSES)
    all_counts = torch.zeros(N)

    # ── Step 1: Deterministic coverage pass ─────────────────────────
    # Divide ALL points into non-overlapping chunks first.
    # This guarantees every single point is seen at least once
    # before we add any overlap patches.
    shuffled_idx = torch.randperm(N)
    chunks = shuffled_idx.split(patch_size)

    for chunk_idx in chunks:
        # Pad last chunk if smaller than patch_size
        if len(chunk_idx) < patch_size:
            pad = torch.randint(0, N, (patch_size - len(chunk_idx),))
            chunk_idx = torch.cat([chunk_idx, pad])

        patch_pos = aligned_pos[chunk_idx]
        patch_data = Data(pos=patch_pos)
        patch_data = compute_normals(patch_data)
        patch_data.batch = torch.zeros(patch_size, dtype=torch.long)
        patch_data = patch_data.to(device)

        with torch.no_grad():
            logits = model(patch_data).cpu()

        # All points in coverage pass get weight 1.0
        all_logits[chunk_idx] += logits
        all_counts[chunk_idx] += 1.0

    # ── Step 2: Spatial overlap pass ────────────────────────────────
    # Now add spatially coherent patches centered across the jaw.
    # These improve boundary smoothness by letting boundary points
    # appear in multiple spatially-aware contexts.
    # Since Step 1 already covered everything, these are pure bonus.
    x_coords = aligned_pos[:, 0]
    y_coords = aligned_pos[:, 1]

    centers_x = torch.linspace(
        x_coords.min(), x_coords.max(), num_patches
    )
    centers_y = torch.linspace(
        y_coords.min(), y_coords.max(), num_patches // 2
    )

    center_points = []
    for cx in centers_x:
        # Use actual jaw points closest to each X position
        # so centers are never floating in empty space
        closest = (x_coords - cx).abs().argmin()
        center_points.append(aligned_pos[closest])

    for cy in centers_y:
        closest = (y_coords - cy).abs().argmin()
        center_points.append(aligned_pos[closest])

    for center in center_points:
        dists = (aligned_pos - center).norm(dim=1)
        _, idx = dists.topk(patch_size, largest=False)

        patch_pos = aligned_pos[idx]
        patch_data = Data(pos=patch_pos)
        patch_data = compute_normals(patch_data)
        patch_data.batch = torch.zeros(patch_size, dtype=torch.long)
        patch_data = patch_data.to(device)

        with torch.no_grad():
            logits = model(patch_data).cpu()

        # Weight by proximity to patch center —
        # points near center had full spatial context
        center_dists = dists[idx]
        max_dist = center_dists.max().clamp(min=1e-8)
        weights = (1.0 - center_dists / max_dist).clamp(min=0.1)

        all_logits[idx] += logits * weights.unsqueeze(1)
        all_counts[idx] += weights

    # ── Step 3: Final prediction ─────────────────────────────────────
    # all_counts is guaranteed >= 1 for every point after Step 1
    # so no uncovered point handling needed
    final_logits = all_logits / all_counts.unsqueeze(1)
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
                num_patches=25
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
