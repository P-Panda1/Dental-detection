import torch


class DentalActiveConfig:
    """Refined Config for Closed-Mesh Dental Segmentation"""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = "../data/"

    CHECKPOINT_DIR = "./model"

    # --- Classification (No Background) ---
    # 0: Gum (Red), 1: Border (Black), 2: Tooth (White)
    NUM_CLASSES = 3

    # We prioritize the Border because it has the fewest points
    # Weights: [Gum, Border, Tooth]
    LOSS_WEIGHTS = torch.tensor([1.0, 12.0, 2.0]).to(DEVICE)

    # --- Model Selection (DGCNN) ---
    NUM_POINTS_GLOBAL = 16384
    K_NEIGHBORS = 20
    EMBEDDING_DIM = 1024

    # --- Training ---
    BATCH_SIZE = 8
    LR = 5e-5
    WEIGHT_DECAY = 1e-4
    EPOCHS = 10000

    # --- Active Learning ---
    CONFIDENCE_THRESHOLD = 0.90
    UNLABELED_DIR = "../data/unlabeled/"
    PREDICT_DIR = "../data/predictions/"

    # --- Metric Learning ---
    USE_ARCFACE = True
    ARC_S = 30.0        # Radius of the feature sphere
    ARC_M = 0.4       # Margin (in radians) to separate classes on the sphere
    EMBEDDING_DIM = 128  # The size of the vector before the ArcFace head


config = DentalActiveConfig()
