import torch
import torch.nn.functional as F
import pyvista as pv
import numpy as np
from tqdm import tqdm

# Import your custom modules
from config import config
from model import DentalMetricDGCNN
from data_loader import get_dental_loaders


def validate_and_plot(model, loader, epoch, device):
    """
    Takes a low-res prediction and interpolates it back to high-res for plotting.
    """
    model.eval()
    # Pull one sample from validation loader
    data = next(iter(loader)).to(device)

    with torch.no_grad():
        # Get predictions (8192 points)
        logits = model(data)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # 1. Create the Predicted Mesh (Low-Res)
    pred_colors = np.zeros((len(preds), 3), dtype=np.uint8)
    pred_colors[preds == 0] = [255, 0, 0]   # Gum -> Red
    pred_colors[preds == 1] = [0, 0, 0]     # Border -> Black
    pred_colors[preds == 2] = [255, 255, 255]  # Tooth -> White

    # 2. Map predictions back to the high-res visualization
    # In a real scenario, you'd use KNN to map 8k labels to 200k points.
    # For this plot, we visualize the points the model actually saw.
    plotter = pv.Plotter(shape=(1, 2), title=f"Epoch {epoch} Comparison")

    # Left: Ground Truth
    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth (Manual)")
    gt_colors = np.zeros((len(data.y), 3), dtype=np.uint8)
    y_cpu = data.y.cpu().numpy()
    gt_colors[y_cpu == 1] = [255, 0, 0]
    gt_colors[y_cpu == 2] = [0, 0, 0]
    gt_colors[y_cpu == 3] = [255, 255, 255]

    gt_mesh = pv.PolyData(data.pos.cpu().numpy())
    plotter.add_mesh(gt_mesh, scalars=gt_colors, rgb=True,
                     point_size=5, render_points_as_spheres=True)

    # Right: Prediction
    plotter.subplot(0, 1)
    plotter.add_text("Model Prediction (ArcFace DGCNN)")
    pred_mesh = pv.PolyData(data.pos.cpu().numpy())
    plotter.add_mesh(pred_mesh, scalars=pred_colors, rgb=True,
                     point_size=5, render_points_as_spheres=True)

    plotter.link_views()
    plotter.show()


def train():
    # 1. Setup Data and Model
    train_loader, val_loader, _ = get_dental_loaders(
        config.DATA_DIR, batch_size=config.BATCH_SIZE)

    model = DentalMetricDGCNN(
        k=config.K_NEIGHBORS,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.GLOBAL_EMBED_DIM // 8  # Matching your config dim
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    print(f"Starting training on {config.DEVICE}...")

    # 2. Training Loop
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = batch.to(config.DEVICE)
            optimizer.zero_grad()

            # Forward pass (labels passed for ArcFace margin)
            logits = model(batch)

            # Weighted Loss to handle the rare 'Border' class
            # y-1 to align 1,2,3 to 0,1,2
            loss = F.cross_entropy(logits, batch.y - 1,
                                   weight=config.LOSS_WEIGHTS)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")

        # 3. Periodically Validate and Plot
        if epoch % 5 == 0 or epoch == 1:
            validate_and_plot(model, val_loader, epoch, config.DEVICE)


if __name__ == "__main__":
    train()
