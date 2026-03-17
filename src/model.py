from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATv2Conv, MLP, global_max_pool
from torch_geometric.nn import DynamicEdgeConv
import torch
from torch_cluster import knn_graph
import torch.nn as nn
import torch.nn.functional as F
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, global_mean_pool
from torch_cluster import knn


class PointArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=16.0, m=0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        cos_theta = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)

        if not self.training or labels is None:
            return cos_theta * self.s

        theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * torch.cos(theta + self.m)) + \
            ((1.0 - one_hot) * cos_theta)
        return output * self.s


class EdgeConvBlock(nn.Module):
    """
    EdgeConv that builds its KNN graph in POSITION space (not feature space).
    This is critical for boundaries — the graph topology stays stable and
    reflects true geometric proximity rather than drifting with features.
    """

    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        ), k=k, aggr='max')

        # Residual projection if channels change
        self.residual = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, batch):
        return self.conv(x, batch) + self.residual(x)


class BoundaryAttentionHead(nn.Module):
    """
    Detects boundary points by comparing each point's embedding to its
    neighbors. Points whose neighborhood has HIGH feature variance are
    likely on a boundary. We use this as an attention weight to sharpen
    boundary predictions.
    """

    def __init__(self, in_channels, k=10):
        super().__init__()
        self.k = k
        self.score = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, pos, batch):
        # Find geometric neighbors
        edge_index = knn(pos, pos, k=self.k,
                         batch_x=batch, batch_y=batch)
        src, dst = edge_index[0], edge_index[1]

        # Compute neighbor feature variance per point
        neighbor_feats = x[src]                            # [N*k, C]
        center_feats = x[dst]                            # [N*k, C]
        diff = (neighbor_feats - center_feats).pow(2)  # [N*k, C]

        # Mean variance per point — high = boundary region
        variance = torch.zeros_like(x)
        variance.scatter_add_(0, dst.unsqueeze(1).expand_as(diff), diff)
        variance = variance / self.k                       # [N, C]

        # Boundary score: 1 = boundary, 0 = interior
        boundary_score = self.score(variance)              # [N, 1]
        return boundary_score


class DentalBoundaryDGCNN(nn.Module):
    def __init__(self, k=20, num_classes=3, embed_dim=128):
        super().__init__()
        self.k = k

        # ── Encoder: position-anchored EdgeConv blocks ──────────────────
        # Each block uses residual connections to preserve fine detail
        self.conv1 = EdgeConvBlock(6,   64,  k=k)
        self.conv2 = EdgeConvBlock(64,  64,  k=k)
        self.conv3 = EdgeConvBlock(64,  128, k=k)
        self.conv4 = EdgeConvBlock(128, 128, k=k)

        # ── Global context — both max AND mean pooling ──────────────────
        # Max captures dominant structures (tooth body, gum body)
        # Mean captures overall shape distribution
        # Concatenating both gives richer global signal
        self.global_mlp = nn.Sequential(
            nn.Linear(384 * 2, 512),   # 384 = 64+64+128+128, ×2 for max+mean
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )

        # ── Boundary attention ──────────────────────────────────────────
        self.boundary_attn = BoundaryAttentionHead(in_channels=384, k=10)

        # ── Point-wise embedding head ───────────────────────────────────
        # Input: local(384) + global(512) + boundary_score(1)
        self.embedding_head = nn.Sequential(
            nn.Linear(384 + 512 + 1, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # ── Boundary refinement head ────────────────────────────────────
        # Separate lightweight head that focuses only on boundary points
        # Its loss is weighted more heavily during training
        self.boundary_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.arcface = PointArcFace(
            in_features=embed_dim,
            out_features=num_classes,
            s=16.0,
            m=0.3
        )

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        # ── Encode ──────────────────────────────────────────────────────
        x1 = self.conv1(x,  batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)

        # Multi-scale local features
        local_feat = torch.cat([x1, x2, x3, x4], dim=1)   # [N, 384]

        # ── Global context (max + mean) ─────────────────────────────────
        g_max = global_max_pool(local_feat,  batch)       # [B, 384]
        g_mean = global_mean_pool(local_feat, batch)       # [B, 384]
        global_vec = self.global_mlp(
            torch.cat([g_max, g_mean], dim=1)
        )                                                   # [B, 512]

        pts_per_graph = torch.bincount(batch)
        global_feat = torch.repeat_interleave(
            global_vec, pts_per_graph, dim=0
        )                                                   # [N, 512]

        # ── Boundary attention ──────────────────────────────────────────
        boundary_score = self.boundary_attn(local_feat, pos, batch)  # [N, 1]

        # ── Embed ───────────────────────────────────────────────────────
        combined = torch.cat([local_feat, global_feat, boundary_score], dim=1)
        embeddings = self.embedding_head(combined)          # [N, embed_dim]

        # ── Logits ──────────────────────────────────────────────────────
        labels_0idx = (data.y - 1) if (data.y is not None) else None
        logits = self.arcface(embeddings, labels_0idx)  # [N, 3]

        # Boundary head logits returned separately for aux loss
        boundary_logits = self.boundary_head(embeddings)    # [N, 3]

        if self.training:
            return logits, boundary_logits, boundary_score
        return logits


class DentalMetricDGCNN(nn.Module):
    def __init__(self, k=20, num_classes=3, embed_dim=128):
        super().__init__()
        self.k = k

        self.conv1 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * 6, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        ), k, aggr='max')

        self.conv2 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * 64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        ), k, aggr='max')

        self.conv3 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * 64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        ), k, aggr='max')

        self.global_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU()
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(256 + 1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.arcface = PointArcFace(
            in_features=embed_dim,
            out_features=num_classes,
            s=30.0,
            m=0.4
        )

    def forward(self, data):
        x, batch = data.x, data.batch

        # ── Local features ──────────────────────────────────────────────
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        local_feat = torch.cat([x1, x2, x3], dim=1)        # [N, 256]

        # ── True global context ─────────────────────────────────────────
        pooled = global_max_pool(local_feat, batch)         # [B, 256]
        global_vec = self.global_mlp(pooled)                # [B, 1024]
        pts_per_graph = torch.bincount(batch)
        global_feat = torch.repeat_interleave(
            global_vec, pts_per_graph, dim=0
        )                                                   # [N, 1024]

        # ── Embeddings ──────────────────────────────────────────────────
        combined = torch.cat([local_feat, global_feat], dim=1)
        embeddings = self.embedding_head(combined)

        # ── ArcFace logits ──────────────────────────────────────────────
        # At inference data.y is None — ArcFace handles this via self.training check
        labels_0idx = (data.y - 1) if (data.y is not None) else None
        logits = self.arcface(embeddings, labels_0idx)
        return logits
