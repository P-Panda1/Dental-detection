from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_cluster import knn_graph
from torch_geometric.nn.models import GraphUNet
from torch_geometric.nn import global_max_pool
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


class LiteGraphUNet(GraphUNet):
    def __init__(self, *args, **kwargs):
        # 1. Store the actual input dimension (e.g., 3 for XYZ)
        actual_in_channels = kwargs.get('in_channels', 3)

        # 2. Force the parent class to see 'in_channels' as the 'hidden_channels'
        # This ensures all internal GraphUNet layers (convs/pools) are size 64
        kwargs['in_channels'] = kwargs.get('hidden_channels', 64)

        super().__init__(*args, **kwargs)

        # 3. Create our custom projection layer to go from 3 -> 64
        self.lin0 = nn.Linear(actual_in_channels, kwargs['in_channels'])

    def forward(self, x, edge_index, batch, edge_weight=None):
        # Project raw features to the hidden dimension
        x = self.lin0(x)

        # Call the parent forward method, which now has all attributes (unpools, etc.)
        return super().forward(x, edge_index, batch, edge_weight)


class DentalGraphUNet(nn.Module):
    def __init__(self, num_classes=3, embed_dim=128, k=20):
        super().__init__()
        self.k = k

        self.backbone = LiteGraphUNet(
            # Pass hidden_channels to in_channels so internal layers match
            in_channels=3,
            hidden_channels=64,
            out_channels=embed_dim,
            depth=4,
            pool_ratios=0.5,
            sum_res=False
        )

        self.arcface = PointArcFace(
            in_features=embed_dim,
            out_features=num_classes
        )

    def forward(self, data):
        pos, batch, label = data.pos, data.batch, data.y

        # Use self.k instead of config.K_NEIGHBORS
        edge_index = knn_graph(pos, k=self.k, batch=batch)

        embeddings = self.backbone(pos, edge_index, batch)
        logits = self.arcface(embeddings, label)

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
