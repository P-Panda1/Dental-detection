import matplotlib.pyplot as plt
import multiprocessing
from IPython.display import Image, display
import pyvista as pv
import torch
import torch.nn.functional as F
import pyvista as pv
import numpy as np
import os
from tqdm import tqdm

# Import custom modules
from config import config
from model import DentalMetricDGCNN, DentalPointTransformer
from data_loader import get_dental_loaders

import torch
import gc


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


def train():
    clear_gpu()
    train_loader, val_loader, _ = get_dental_loaders(
        "../data", batch_size=config.BATCH_SIZE)

    # model = DentalMetricDGCNN(
    #     k=config.K_NEIGHBORS,
    #     num_classes=config.NUM_CLASSES,
    #     embed_dim=config.GLOBAL_EMBED_DIM // 8
    # ).to(config.DEVICE)

    model = DentalPointTransformer(
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
    best_val_loss = float('inf')

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = batch.to(config.DEVICE)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y - 1,
                                   weight=config.LOSS_WEIGHTS)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # 1. Calculate avg_train_loss here, immediately after the batch loop
        avg_train_loss = total_train_loss / len(train_loader)

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
