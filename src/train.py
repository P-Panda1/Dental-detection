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


def validate_and_plot(model, loader, epoch, device):
    model.eval()
    # Get the first sample from the validation set
    try:
        data = next(iter(loader)).to(device)
    except StopIteration:
        return 1e9  # Return high loss if loader is empty

    with torch.no_grad():
        logits = model(data)
        # Calculate val loss for checkpointing
        val_loss = F.cross_entropy(
            logits, data.y - 1, weight=config.LOSS_WEIGHTS)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Define color map for plotting
    pred_colors = np.zeros((len(preds), 3), dtype=np.uint8)
    pred_colors[preds == 0] = [255, 0, 0]   # Gum
    pred_colors[preds == 1] = [0, 0, 0]     # Border
    pred_colors[preds == 2] = [255, 255, 255]  # Tooth

    # Visualization
    plotter = pv.Plotter(shape=(1, 2), title=f"Epoch {epoch} Comparison")

    # Ground Truth
    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth")
    gt_colors = np.zeros((len(data.y), 3), dtype=np.uint8)
    y_cpu = data.y.cpu().numpy()
    gt_colors[y_cpu == 1] = [255, 0, 0]
    gt_colors[y_cpu == 2] = [0, 0, 0]
    gt_colors[y_cpu == 3] = [255, 255, 255]
    plotter.add_mesh(pv.PolyData(data.pos.cpu().numpy()),
                     scalars=gt_colors, rgb=True, point_size=5)

    # Prediction
    plotter.subplot(0, 1)
    plotter.add_text("ArcFace Prediction")
    plotter.add_mesh(pv.PolyData(data.pos.cpu().numpy()),
                     scalars=pred_colors, rgb=True, point_size=5)

    plotter.link_views()
    plotter.show()

    return val_loss.item()


def train():
    train_loader, val_loader, _ = get_dental_loaders(
        config.DATA_DIR, batch_size=config.BATCH_SIZE)

    model = DentalMetricDGCNN(
        k=config.K_NEIGHBORS,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.GLOBAL_EMBED_DIM // 8
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    best_val_loss = float('inf')
    print(f"Starting training on {config.DEVICE}...")

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

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")

        # Periodic Validation, Plotting, and Saving
        if epoch % 5 == 0 or epoch == 1:
            val_loss = validate_and_plot(
                model, val_loader, epoch, config.DEVICE)

            # Check if this is the best model so far
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            save_checkpoint(model, optimizer, epoch, val_loss, is_best=is_best)


if __name__ == "__main__":
    train()
