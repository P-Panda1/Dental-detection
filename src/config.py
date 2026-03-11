import torch


class DentalActiveConfig:
    """Refined Config for Closed-Mesh Dental Segmentation"""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = "../data/"

    # --- Classification (No Background) ---
    # 0: Gum (Red), 1: Border (Black), 2: Tooth (White)
    NUM_CLASSES = 3

    # We prioritize the Border because it has the fewest points
    # Weights: [Gum, Border, Tooth]
    LOSS_WEIGHTS = torch.tensor([1.0, 8.0, 1.0]).to(DEVICE)

    # --- Model Selection (DGCNN) ---
    NUM_POINTS_GLOBAL = 8192
    K_NEIGHBORS = 20
    GLOBAL_EMBED_DIM = 1024

    # --- Training ---
    BATCH_SIZE = 4
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100

    # --- Active Learning ---
    CONFIDENCE_THRESHOLD = 0.90
    UNLABELED_DIR = "../data/unlabeled/"
    PREDICT_DIR = "../data/predictions/"

    # --- Metric Learning ---
    USE_ARCFACE = True
    ARC_S = 30.0        # Radius of the feature sphere
    ARC_M = 1.2         # Margin (1.2 radians is ~69 degrees of separation)
    EMBEDDING_DIM = 128  # The size of the vector before the ArcFace head


config = DentalActiveConfig()
