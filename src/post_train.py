import torch
import pyvista as pv
import numpy as np
import os
from model import DentalMetricDGCNN
from config import config
from data_loader import get_dental_loaders


def visualize_best_model():
    # 1. Setup Device (CPU is safer for plotting if GPU is busy)
    device = torch.device("cpu")

    # 2. Load the Best Model
    model = DentalMetricDGCNN(
        k=config.K_NEIGHBORS,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.GLOBAL_EMBED_DIM // 8
    )

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Did the training finish?")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")

    # 3. Get one sample from validation data
    _, val_loader, _ = get_dental_loaders(config.DATA_DIR, batch_size=1)
    data = next(iter(val_loader)).to(device)

    # 4. Run Prediction
    with torch.no_grad():
        logits = model(data)
        preds = torch.argmax(logits, dim=1).numpy()

    # 5. Prepare Colors
    points = data.pos.numpy()

    # Ground Truth Colors (1,2,3 -> RGB)
    gt_y = data.y.numpy()
    gt_colors = np.zeros((len(gt_y), 3), dtype=np.uint8)
    gt_colors[gt_y == 1] = [255, 0, 0]   # Gum
    gt_colors[gt_y == 2] = [0, 0, 0]     # Border
    gt_colors[gt_y == 3] = [255, 255, 255]  # Tooth

    # Prediction Colors (0,1,2 -> RGB)
    pred_colors = np.zeros((len(preds), 3), dtype=np.uint8)
    pred_colors[preds == 0] = [255, 0, 0]
    pred_colors[preds == 1] = [0, 0, 0]
    pred_colors[preds == 2] = [255, 255, 255]

    # 6. Render to Image (No interactive window to avoid crashes)
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(shape=(1, 2), off_screen=True)

    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth (Manual)", font_size=12)
    plotter.add_mesh(pv.PolyData(points), scalars=gt_colors,
                     rgb=True, point_size=5)

    plotter.subplot(0, 1)
    plotter.add_text("DGCNN + ArcFace Prediction", font_size=12)
    plotter.add_mesh(pv.PolyData(points), scalars=pred_colors,
                     rgb=True, point_size=5)

    plotter.link_views()

    # Save and Output
    output_img = "final_prediction_comparison.png"
    plotter.screenshot(output_img)
    plotter.close()

    print(f"Successfully saved prediction comparison to: {output_img}")

    # Display in Notebook
    from IPython.display import Image, display
    display(Image(filename=output_img))


if __name__ == "__main__":
    visualize_best_model()
