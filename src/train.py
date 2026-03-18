import matplotlib.pyplot as plt
import multiprocessing
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from IPython.display import Image, display
import pyvista as pv
import torch
import torch.nn.functional as F
import pyvista as pv
import numpy as np
import os
from tqdm import tqdm
from torch_cluster import knn

# Import custom modules
from config import config
from model import DentalMetricDGCNN, DentalBoundaryDGCNN
from data_loader import get_dental_loaders

import torch
import gc
import warnings

# Filter out the specific scatter acceleration warning from torch_geometric
warnings.filterwarnings(
    "ignore", message=".*accelerated via the 'torch-scatter' package.*")


def clear_gpu():
    # Clear Python's garbage collector
    gc.collect()
    # Clear the PyTorch CUDA cache
    torch.cuda.empty_cache()
    # Optionally reset peak memory stats
    torch.cuda.reset_peak_memory_stats()


def save_checkpoint(model, optimizer, epoch, loss, is_best=False):
    """Saves model weights and optimizer state."""
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR, "latest_checkpoint.pth")
    torch.save(state, checkpoint_path)

    # Save best model separately
    if is_best:
        best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
        torch.save(state, best_path)
        print(f"--- Best model saved at Epoch {epoch} ---")


def isolated_plotter(points, gt_y, preds, epoch):
    """
    Uses Matplotlib for 2D projections. 
    Mathematically stable and won't crash headless servers.
    """
    try:
        os.makedirs("val_plots", exist_ok=True)

        # 0: Gum (Red), 1: Border (Black), 2: Tooth (Blue for contrast)
        color_map = {1: 'red', 2: 'black', 3: 'blue'}
        pred_map = {0: 'red', 1: 'black', 2: 'blue'}

        fig = plt.figure(figsize=(12, 6))

        # --- Top View Projection (X and Y coordinates) ---
        # Ground Truth
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title(f"Epoch {epoch} - Ground Truth")
        colors_gt = [color_map.get(l, 'gray') for l in gt_y]
        ax1.scatter(points[:, 0], points[:, 1], c=colors_gt, s=0.5, alpha=0.7)
        ax1.axis('equal')

        # Prediction
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title(f"Epoch {epoch} - Prediction")
        colors_pred = [pred_map.get(l, 'gray') for l in preds]
        ax2.scatter(points[:, 0], points[:, 1],
                    c=colors_pred, s=0.5, alpha=0.7)
        ax2.axis('equal')

        plt.tight_layout()
        save_path = f"val_plots/epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    except Exception as e:
        print(f"Matplotlib Plotter failed: {e}")


def balanced_mean_loss(logits, labels, num_classes=3):
    # Compute per-point CE loss without reduction
    per_point_loss = F.cross_entropy(logits, labels, reduction='none')  # [N]

    # Compute mean loss per class using scatter, fully on GPU
    class_counts = torch.bincount(labels, minlength=num_classes).float()  # [C]
    class_sums = torch.zeros(num_classes, device=logits.device)
    class_sums.scatter_add_(0, labels, per_point_loss)

    # Only average over classes that actually appear in this batch
    mask = class_counts > 0
    class_means = class_sums[mask] / class_counts[mask]
    return class_means.mean()


def dice_loss(logits, labels, num_classes=3, smooth=1e-6):
    """
    Differentiable approximation of per-class IoU.
    Works on soft probabilities so gradients flow.
    """
    probs = torch.softmax(logits, dim=1)           # [N, num_classes]
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)   # [N, num_classes]

    dims = (0,)  # sum over points
    intersection = (probs * one_hot).sum(dims)
    cardinality = (probs + one_hot).sum(dims)

    dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice_per_class.mean()


def combined_loss(logits, labels, num_classes=3, dice_weight=0.5):
    ce = balanced_mean_loss(logits, labels, num_classes)
    dice = dice_loss(logits, labels, num_classes)
    return ce + dice_weight * dice


def topology_loss(logits, pos, batch, k=10):
    """
    Penalizes any prediction where gum (class 0) and tooth (class 2)
    are direct geometric neighbors without border (class 1) between them.
    Uses soft probabilities so it's differentiable.
    """
    probs = torch.softmax(logits, dim=1)   # [N, 3]
    p_gum = probs[:, 0]                    # prob of gum
    p_tooth = probs[:, 2]                    # prob of tooth

    edge_index = knn(pos, pos, k=k,
                     batch_x=batch, batch_y=batch)
    src, dst = edge_index[0], edge_index[1]

    # For each point, get max gum and tooth probability among neighbors
    neighbor_gum_prob = p_gum[src]
    neighbor_tooth_prob = p_tooth[src]

    max_neighbor_gum = torch.zeros_like(p_gum).scatter_reduce_(
        0, dst, neighbor_gum_prob, reduce='amax', include_self=False
    )
    max_neighbor_tooth = torch.zeros_like(p_tooth).scatter_reduce_(
        0, dst, neighbor_tooth_prob, reduce='amax', include_self=False
    )

    # Conflict score: high when a point's neighborhood has both
    # high gum and high tooth probability
    conflict = max_neighbor_gum * max_neighbor_tooth   # [N]

    # Penalize conflicts — this pushes the model to insert border
    # between gum and tooth regions
    penalty = (conflict * p_gum).mean() + (conflict * p_tooth).mean()
    return penalty


def boundary_loss(logits, boundary_logits, boundary_score,
                  labels, pos, batch, num_classes=3):
    ce = balanced_mean_loss(logits, labels, num_classes)
    dice = dice_loss(logits, labels, num_classes)

    border_mask = (labels == 1).float()
    border_weight = 1.0 + 4.0 * border_mask
    aux_loss = F.cross_entropy(
        boundary_logits, labels, reduction='none'
    )
    aux_loss = (aux_loss * border_weight).mean()

    with torch.no_grad():
        target_score = border_mask.unsqueeze(1)
    consistency = F.binary_cross_entropy(boundary_score, target_score)

    # Topology term — enforces gum never touches tooth
    topo = topology_loss(logits, pos, batch, k=10)

    return ce + 0.3 * dice + 0.4 * aux_loss + 0.1 * consistency + 0.2 * topo


def train():
    clear_gpu()
    train_loader, val_loader, _ = get_dental_loaders(
        "../data", batch_size=config.BATCH_SIZE, num_points=config.NUM_POINTS_GLOBAL)

    model_type = 1

    if model_type == 0:
        model = DentalMetricDGCNN(
            k=config.K_NEIGHBORS,
            num_classes=config.NUM_CLASSES,
            embed_dim=config.EMBEDDING_DIM
        ).to(config.DEVICE)
    else:
        model = DentalBoundaryDGCNN(
            k=config.K_NEIGHBORS,
            num_classes=config.NUM_CLASSES,
            embed_dim=config.EMBEDDING_DIM
        ).to(config.DEVICE)

    # model.compile()  # Optional: Use PyTorch 2.0 compilation for potential speedup

    # --- LOAD BEST MODEL IF IT EXISTS ---
    best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_path):
        print(f"--- Loading existing best model from {best_path} ---")
        checkpoint = torch.load(best_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    # T_max is the number of steps to reach the minimum LR (usually set to total epochs)
    # eta_min is the lowest the learning rate will go (e.g., 1e-6)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=1e-6
    )
    best_val_loss = float('inf')

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = batch.to(config.DEVICE)
            optimizer.zero_grad()
            if model_type == 0:
                logits = model(batch)
                # loss = F.cross_entropy(logits, batch.y - 1)
                loss = combined_loss(
                    logits, batch.y - 1, num_classes=config.NUM_CLASSES)
            else:
                logits, boundary_logits, boundary_score = model(batch)
                loss = boundary_loss(
                    logits, boundary_logits, boundary_score,
                    batch.y - 1, batch.pos, batch.batch,
                    num_classes=config.NUM_CLASSES
                )
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # 1. Calculate avg_train_loss here, immediately after the batch loop
        avg_train_loss = total_train_loss / len(train_loader)

        # --- STEP THE SCHEDULER HERE ---
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} complete. Current LR: {current_lr:.6f}")
        # --- VALIDATION & PLOTTING ---
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(config.DEVICE)
                    val_logits = model(val_batch)
                    v_loss = F.cross_entropy(
                        val_logits, val_batch.y - 1, weight=config.LOSS_WEIGHTS)
                    total_val_loss += v_loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                preds = torch.argmax(val_logits, dim=1).cpu().numpy()
                points_cpu = val_batch.pos.cpu().numpy()
                gt_y_cpu = val_batch.y.cpu().numpy()

            # 2. Now avg_train_loss is safely defined and can be printed
            print(
                f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Spawn the plotter for the latest batch
            p = multiprocessing.Process(target=isolated_plotter, args=(
                points_cpu, gt_y_cpu, preds, epoch))
            p.start()
            p.join()

            # Save Checkpoint based on AVERAGE validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, epoch,
                                avg_val_loss, is_best=True)
            else:
                save_checkpoint(model, optimizer, epoch,
                                avg_val_loss, is_best=False)


if __name__ == "__main__":
    # Multiprocessing fix for Windows/Jupyter
    multiprocessing.set_start_method('spawn', force=True)
    train()
