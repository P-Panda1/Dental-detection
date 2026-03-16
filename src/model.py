from torch_geometric.nn import global_max_pool
from torch_geometric.nn import SAModule, FPModule, MLP
from torch_geometric.nn import DynamicEdgeConv, MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PointArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(PointArcFace, self).__init__()
        self.s = s
        self.m = m
        # Weights represent the 'center' of each class on the sphere
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label=None):
        # 1. Normalize features and weights to project onto the unit sphere
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        # During inference, we just return the scaled cosine similarity
        if not self.training or label is None:
            return cosine * self.s

        # Ensure labels are 0, 1, 2.
        # If your data_loader gives 1, 2, 3, we subtract 1 here.
        if label.min() >= 1:
            label = label - 1

        # 2. Apply the Angular Margin (ArcFace logic)
        # We clamp for numerical stability before acos
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logit = torch.cos(theta + self.m)

        # 3. Create the output logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * target_logit) + ((1.0 - one_hot) * cosine)

        return output * self.s


class PointNet2Backbone(nn.Module):
    def __init__(self, num_classes=3, embed_dim=128):
        super().__init__()

        # 1. ENCODER (Set Abstraction Layers)
        # SA1: Groups points into local clusters to find edges/textures
        self.sa1_module = SAModule(0.2, 0.05, MLP([3 + 3, 64, 64, 128]))
        # SA2: Groups those clusters to understand larger structures (tooth shape)
        self.sa2_module = SAModule(0.25, 0.1, MLP([128 + 3, 128, 128, 256]))
        # SA3: Condenses everything into a global context vector
        self.sa3_module = SAModule(None, None, MLP([256 + 3, 256, 512, 1024]))

        # 2. DECODER (Feature Propagation Layers)
        # FP3: Spreads global info back to SA2 clusters
        self.fp3_module = FPModule(1024, MLP([1024 + 256, 256, 256]))
        # FP2: Spreads info back to SA1 clusters
        self.fp2_module = FPModule(256, MLP([256 + 128, 256, 128]))
        # FP1: Spreads info back to the original 8192 points
        self.fp1_module = FPModule(128, MLP([128 + 3, 128, 128, embed_dim]))

        # 3. METRIC HEAD
        self.arcface = PointArcFace(
            in_features=embed_dim, out_features=num_classes)

    def forward(self, data):
        # We start with original points
        sa0_out = (data.x, data.pos, data.batch)

        # --- Downsampling Path ---
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        # --- Upsampling Path (Skip Connections) ---
        x, pos, batch = sa3_out
        x = self.fp3_module(x, pos, batch, *sa2_out)
        x = self.fp2_module(x, pos, batch, *sa1_out)
        x = self.fp1_module(x, pos, batch, *sa0_out)

        # x is now a [8192, embed_dim] feature matrix
        logits = self.arcface(x, data.y)

        return logits


class DentalMetricDGCNN(nn.Module):
    def __init__(self, k=20, num_classes=3, embed_dim=128):
        super().__init__()
        self.k = k

        # 1. EdgeConv Layers
        # We manually define the MLPs inside EdgeConv to include Batch Norm
        self.conv1 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ), k, aggr='max')

        self.conv2 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * 64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ), k, aggr='max')

        self.conv3 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        ), k, aggr='max')

        # 2. Global Feature Extractor with Dropout
        self.global_mlp = nn.Sequential(
            nn.Linear(64 + 64 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Prevent memorizing specific jaw patterns
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # 3. Embedding Head with heavier Dropout
        # This is where the model maps features to the sphere
        self.embedding_head = nn.Sequential(
            nn.Linear(1024 + 64 + 64 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim)  # Final norm before ArcFace
        )

        self.arcface = PointArcFace(
            in_features=embed_dim, out_features=num_classes)

    def forward(self, data):
        pos, batch, label = data.pos, data.batch, data.y

        # Multi-scale Local Feature Extraction
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)

        combined = torch.cat([x1, x2, x3], dim=1)

        # Global Context
        global_feat = self.global_mlp(combined)

        # Point-wise Embeddings
        x = torch.cat([combined, global_feat], dim=1)
        embeddings = self.embedding_head(x)

        # ArcFace Logits
        logits = self.arcface(embeddings, label)

        return logits
