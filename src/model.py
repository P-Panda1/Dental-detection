from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn import PointConv, MLP, fps, radius
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


class ResPointBlock(nn.Module):
    """A Residual Block for Point Clouds"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = PointConv(local_nn=MLP(
            [in_channels + 3, out_channels, out_channels]))
        self.shortcut = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x, pos, edge_index):
        identity = self.shortcut(x)
        out = self.conv(x, pos, edge_index)
        return self.relu(out + identity)


class DentalResPointNet(nn.Module):
    def __init__(self, num_classes=3, embed_dim=128):
        super().__init__()

        # Stage 1: High Res (8192 -> 2048)
        self.enc1 = ResPointBlock(3, 64)
        self.enc1_b = ResPointBlock(64, 64)  # Residual Layer 2

        # Stage 2: Mid Res (2048 -> 512)
        self.enc2 = ResPointBlock(64, 128)
        self.enc2_b = ResPointBlock(128, 128)  # Residual Layer 2

        # Stage 3: Low Res (512 -> 128)
        self.enc3 = ResPointBlock(128, 256)

        # Decoder / Feature Propagation
        self.fp3 = MLP([256 + 128, 256, 128])
        self.fp2 = MLP([128 + 64, 128, 128])
        self.fp1 = MLP([128 + 3, 128, embed_dim])

        self.arcface = PointArcFace(
            in_features=embed_dim, out_features=num_classes)

    def forward(self, data):
        # We start with raw coordinates as features
        x0, pos0, batch0 = data.pos, data.pos, data.batch

        # --- ENCODER 1 ---
        idx1 = fps(pos0, batch0, ratio=0.25)
        row, col = radius(pos0, pos0[idx1], r=0.1, batch_x=batch0,
                          batch_y=batch0[idx1], max_num_neighbors=32)
        edge_index1 = torch.stack([col, row], dim=0)
        x1 = self.enc1(x0, pos0, edge_index1)[idx1]
        # Simplified neighbor reuse
        x1 = self.enc1_b(x1, pos0[idx1], edge_index1[:, ::4])
        pos1, batch1 = pos0[idx1], batch0[idx1]

        # --- ENCODER 2 ---
        idx2 = fps(pos1, batch1, ratio=0.25)
        row, col = radius(pos1, pos1[idx2], r=0.2, batch_x=batch1,
                          batch_y=batch1[idx2], max_num_neighbors=32)
        edge_index2 = torch.stack([col, row], dim=0)
        x2 = self.enc2(x1, pos1, edge_index2)[idx2]
        x2 = self.enc2_b(x2, pos1[idx2], edge_index2[:, ::4])
        pos2, batch2 = pos1[idx2], batch1[idx2]

        # --- ENCODER 3 ---
        idx3 = fps(pos2, batch2, ratio=0.25)
        row, col = radius(pos2, pos2[idx3], r=0.4, batch_x=batch2,
                          batch_y=batch2[idx3], max_num_neighbors=32)
        edge_index3 = torch.stack([col, row], dim=0)
        x3 = self.enc3(x2, pos2, edge_index3)[idx3]
        pos3, batch3 = pos2[idx3], batch2[idx3]

        # --- DECODER with Skip Connections ---
        # Up from x3 to pos2
        upsampled3 = knn_interpolate(
            x3, pos3, pos2, batch_x=batch3, batch_y=batch2, k=3)
        x_up2 = self.fp3(torch.cat([upsampled3, x2], dim=-1))

        # Up from x_up2 to pos1
        upsampled2 = knn_interpolate(
            x_up2, pos2, pos1, batch_x=batch2, batch_y=batch1, k=3)
        x_up1 = self.fp2(torch.cat([upsampled2, x1], dim=-1))

        # Up from x_up1 to pos0
        upsampled1 = knn_interpolate(
            x_up1, pos1, pos0, batch_x=batch1, batch_y=batch0, k=3)
        embeddings = self.fp1(torch.cat([upsampled1, pos0], dim=-1))

        return self.arcface(embeddings, data.y)


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
