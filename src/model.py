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

        # 2. Apply the Angular Margin (ArcFace logic)
        # We clamp for numerical stability before acos
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logit = torch.cos(theta + self.m)

        # 3. Create the output logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * target_logit) + ((1.0 - one_hot) * cosine)

        return output * self.s


class DentalMetricDGCNN(nn.Module):
    def __init__(self, k=20, num_classes=3, embed_dim=128):
        super().__init__()
        # K-Nearest Neighbors for local graph construction
        self.k = k

        # EdgeConv Layers: Learn local 'trench' and 'dome' features
        # MLP([input_dim * 2, hidden, output])
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64]), k, aggr='max')
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr='max')
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 128, 128]), k, aggr='max')

        # Global Feature Extractor (The context of the whole jaw)
        self.global_mlp = nn.Sequential(
            nn.Linear(64 + 64 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )

        # Feature Bottleneck (The vector we send to ArcFace)
        self.embedding_head = MLP([1024 + 64 + 64 + 128, 512, 256, embed_dim])

        # The Metric Learning Head
        self.arcface = PointArcFace(
            in_features=embed_dim, out_features=num_classes)

    def forward(self, data):
        pos, batch, label = data.pos, data.batch, data.y

        # 1. Multi-scale Local Feature Extraction
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)

        # 2. Global Context Aggregation
        combined = torch.cat([x1, x2, x3], dim=1)
        # Use global_max_pool-like behavior per batch
        # For simplicity in this block, we assume a single graph or use global_feat
        global_feat = self.global_mlp(combined)

        # 3. Create Point-wise Embeddings
        # Concatenate local features with global context
        x = torch.cat([combined, global_feat], dim=1)
        embeddings = self.embedding_head(x)

        # 4. Final ArcFace Logits
        # During training, we pass labels to apply the margin
        logits = self.arcface(embeddings, label)

        return logits
