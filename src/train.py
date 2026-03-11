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
from model import DentalMetricDGCNN
from data_loader import get_dental_loaders


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
    Runs in a separate process to avoid OpenGL/CUDA collisions.
    """
    try:
        os.makedirs("val_plots", exist_ok=True)
        pv.OFF_SCREEN = True

        # Setup colors
        gt_colors = np.zeros((len(gt_y), 3), dtype=np.uint8)
        gt_colors[gt_y == 1] = [255, 0, 0]   # Gum
        gt_colors[gt_y == 2] = [0, 0, 0]     # Border
        gt_colors[gt_y == 3] = [255, 255, 255]  # Tooth

        pred_colors = np.zeros((len(preds), 3), dtype=np.uint8)
        pred_colors[preds == 0] = [255, 0, 0]
        pred_colors[preds == 1] = [0, 0, 0]
        pred_colors[preds == 2] = [255, 255, 255]

        plotter = pv.Plotter(shape=(1, 2), off_screen=True)

        plotter.subplot(0, 0)
        plotter.add_mesh(pv.PolyData(points),
                         scalars=gt_colors, rgb=True, point_size=4)
        plotter.add_text("Ground Truth", font_size=10)

        plotter.subplot(0, 1)
        plotter.add_mesh(pv.PolyData(points),
                         scalars=pred_colors, rgb=True, point_size=4)
        plotter.add_text(f"Pred Epoch {epoch}", font_size=10)

        save_path = f"val_plots/epoch_{epoch}.png"
        plotter.screenshot(save_path)
        plotter.close()
    except Exception as e:
        print(f"Plotting process failed: {e}")


def train():
    train_loader, val_loader, _ = get_dental_loaders(
        "../data", batch_size=config.BATCH_SIZE)

    model = DentalMetricDGCNN(
        k=config.K_NEIGHBORS,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.GLOBAL_EMBED_DIM // 8
    ).to(config.DEVICE)

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

        # --- VALIDATION & PLOTTING ---
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader)).to(config.DEVICE)
                val_logits = model(val_batch)
                val_loss = F.cross_entropy(
                    val_logits, val_batch.y - 1, weight=config.LOSS_WEIGHTS).item()
                preds = torch.argmax(val_logits, dim=1).cpu().numpy()

                # Move to CPU for plotting
                points_cpu = val_batch.pos.cpu().numpy()
                gt_y_cpu = val_batch.y.cpu().numpy()

            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

            # Spawn the plotter in a SEPARATE process
            p = multiprocessing.Process(target=isolated_plotter, args=(
                points_cpu, gt_y_cpu, preds, epoch))
            p.start()
            p.join()  # Wait for it to finish so we can display it

            # Show the image in Jupyter
            img_path = f"val_plots/epoch_{epoch}.png"
            if os.path.exists(img_path):
                display(Image(filename=img_path))

            # Save Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch,
                                val_loss, is_best=True)
            else:
                save_checkpoint(model, optimizer, epoch,
                                val_loss, is_best=False)


if __name__ == "__main__":
    # Multiprocessing fix for Windows/Jupyter
    multiprocessing.set_start_method('spawn', force=True)
    train()
