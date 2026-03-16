from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATv2Conv, MLP, global_max_pool
from torch_geometric.nn import DynamicEdgeConv
import torch
from torch_cluster import knn_graph
import torch.nn as nn
import torch.nn.functional as F
import math


class PointArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize both embeddings and weights onto the unit sphere
        cos_theta = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)

        # Add angular margin only to the correct class
        theta = torch.acos(cos_theta)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = torch.cos(theta + one_hot * self.m)
        output = output * self.s
        return output
# class PointArcFace(nn.Module):
#     def __init__(self, in_features, out_features, s=30.0, m=0.50):
#         super(PointArcFace, self).__init__()
#         self.s = s
#         self.m = m
#         # Weights represent the 'center' of each class on the sphere
#         self.weight = nn.Parameter(
#             torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, x, label=None):
#         # 1. Normalize features and weights to project onto the unit sphere
#         cosine = F.linear(F.normalize(x), F.normalize(self.weight))

#         # During inference, we just return the scaled cosine similarity
#         if not self.training or label is None:
#             return cosine * self.s

#         # Ensure labels are 0, 1, 2.
#         # If your data_loader gives 1, 2, 3, we subtract 1 here.
#         if label.min() >= 1:
#             label = label - 1

#         # 2. Apply the Angular Margin (ArcFace logic)
#         # We clamp for numerical stability before acos
#         theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
#         target_logit = torch.cos(theta + self.m)

#         # 3. Create the output logits
#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
#         output = (one_hot * target_logit) + ((1.0 - one_hot) * cosine)

#         return output * self.s


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        # GATv2 is a robust dynamic attention mechanism
        self.attn = GATv2Conv(in_channels, out_channels //
                              heads, heads=heads, add_self_loops=True)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        # Attention Layer
        x_attn = self.attn(x, edge_index)
        x = self.norm1(x_attn + x if x.size(-1) == x_attn.size(-1) else x_attn)

        # Feed Forward Layer
        x_ffn = self.ffn(x)
        x = self.norm2(x_ffn + x)
        return x


class DentalPointTransformer(nn.Module):
    def __init__(self, num_classes=3, embed_dim=128, k=16):
        super().__init__()
        self.k = k

        # Initial projection from XYZ to Hidden
        self.lin0 = nn.Linear(6, 64)

        # Deep Transformer Layers (Residual)
        self.layer1 = TransformerBlock(64, 64)
        self.layer2 = TransformerBlock(64, 128)
        self.layer3 = TransformerBlock(128, 128)
        self.layer4 = TransformerBlock(128, 256)

        # Final Embedding Head
        self.head = nn.Sequential(
            nn.Linear(256 + 128 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim)
        )

        self.arcface = PointArcFace(
            in_features=embed_dim, out_features=num_classes)

    def forward(self, data):
        pos, batch, label = data.pos, data.batch, data.y

        # 1. Build a stable KNN Graph (one time only)
        # Using a fixed K is much safer than Radius search
        edge_index = knn_graph(pos, k=self.k, batch=batch)

        # 2. Extract Features
        x = self.lin0(data.x)  # From 6 (pos+normals) to 64

        x1 = self.layer1(x, edge_index)
        x2 = self.layer2(x1, edge_index)
        x3 = self.layer3(x2, edge_index)
        x4 = self.layer4(x3, edge_index)

        # 3. Multi-scale Concatenation (Skip Connections)
        # We combine early fine details with late global features
        combined = torch.cat([x1, x3, x4], dim=1)

        embeddings = self.head(combined)
        logits = self.arcface(embeddings, label)

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
        combined = torch.cat([local_feat, global_feat], dim=1)  # [N, 1280]
        embeddings = self.embedding_head(
            combined)              # [N, embed_dim]

        # ── ArcFace logits ──────────────────────────────────────────────
        # Labels must be 0-indexed here AND in the loss call.
        # data.y is 1-indexed (1=Gum, 2=Border, 3=Tooth), so subtract 1.
        # [N]  values: 0,1,2
        labels_0idx = data.y - 1
        logits = self.arcface(embeddings, labels_0idx)          # [N, 3]
        return logits
