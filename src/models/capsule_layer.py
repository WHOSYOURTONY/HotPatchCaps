"""
capsule_layer.py

Core capsule network components:
  - PrimaryCapsLayer  : maps an MLP embedding to N primary capsule vectors
  - CapsuleLayer      : standard dynamic-routing capsule layer (Sabour et al., 2017)
  - CapsuleNetwork    : full TF-IDF -> CapsNet classifier

These building blocks are used by HotPatchCapsModel (see models/hotpatch_caps.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCapsLayer(nn.Module):
    """
    Maps a flat embedding vector to N primary capsule vectors.

    Each capsule is an independent Linear + LayerNorm projection.
    Output shape: [batch_size, num_capsules, capsule_dim]
    """

    def __init__(self, in_channels: int, num_capsules: int, capsule_dim: int):
        super().__init__()
        self.capsules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, capsule_dim),
                nn.LayerNorm(capsule_dim),
            )
            for _ in range(num_capsules)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels]
        Returns:
            [batch_size, num_capsules, capsule_dim]
        """
        return torch.stack([cap(x) for cap in self.capsules], dim=1)


class CapsuleLayer(nn.Module):
    """
    Dynamic-routing capsule layer (Sabour et al., NeurIPS 2017).

    Routing algorithm (3 iterations by default):
        b_ij  <- 0
        for r in range(routing_iters):
            c_ij  <- softmax(b_ij)        # coupling coefficients
            s_j   <- sum_i c_ij * W_ij x_i
            v_j   <- squash(s_j)
            b_ij  <- b_ij + <u_j|i, v_j>  (agreement update)

    Input shape : [batch_size, num_route_nodes, in_channels]
    Output shape: [batch_size, num_capsules,   out_channels]
    """

    def __init__(
        self,
        num_capsules: int,
        num_route_nodes: int,
        in_channels: int,
        out_channels: int,
        routing_iters: int = 3,
    ):
        super().__init__()
        self.num_capsules   = num_capsules
        self.num_route_nodes = num_route_nodes
        self.routing_iters  = routing_iters

        # Weight tensor W: [num_route_nodes, num_capsules, out_channels, in_channels]
        self.weights = nn.Parameter(
            torch.randn(num_route_nodes, num_capsules, out_channels, in_channels) * 0.1
        )

    @staticmethod
    def squash(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Squash non-linearity: maps vector magnitude to (0, 1)."""
        sq_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale   = sq_norm / (1.0 + sq_norm)
        return scale * tensor / torch.sqrt(sq_norm + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_route_nodes, in_channels]
        Returns:
            [batch_size, num_capsules, out_channels]
        """
        B = x.size(0)

        # Prediction vectors: [B, num_route_nodes, num_capsules, out_channels]
        predictions = torch.einsum('bni,njoi->bnjo', x, self.weights)

        # Routing logits: [B, num_route_nodes, num_capsules, 1]
        b = torch.zeros(B, self.num_route_nodes, self.num_capsules, 1, device=x.device)

        v = None
        for i in range(self.routing_iters):
            c = F.softmax(b, dim=2)                           # [B, Nr, K, 1]
            s = (predictions * c).sum(dim=1, keepdim=True)    # [B, 1, K, d]
            v = self.squash(s, dim=-1)                         # [B, 1, K, d]

            if i < self.routing_iters - 1:
                agreement = (predictions * v).sum(dim=-1, keepdim=True)
                b = b + agreement

        return v.squeeze(1)    # [B, num_capsules, out_channels]


class CapsuleNetwork(nn.Module):
    """
    End-to-end TF-IDF -> CapsNet classifier.

    Architecture:
        TF-IDF features [B, input_dim]
        -> MLP (512 -> 256, ReLU)
        -> PrimaryCapsLayer (32 caps, 8-dim each)
        -> CapsuleLayer     (K class caps, 16-dim each)
        -> flatten + Linear -> logits [B, K]

    Args:
        input_dim : TF-IDF feature dimension
        num_labels: number of output classes
    """

    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        num_primary_caps = 32
        primary_cap_dim  = 8
        class_cap_dim    = 16

        self.primary_capsules = PrimaryCapsLayer(
            in_channels=256,
            num_capsules=num_primary_caps,
            capsule_dim=primary_cap_dim,
        )
        self.class_capsules = CapsuleLayer(
            num_capsules=num_labels,
            num_route_nodes=num_primary_caps,
            in_channels=primary_cap_dim,
            out_channels=class_cap_dim,
        )
        self.classifier = nn.Linear(class_cap_dim * num_labels, num_labels)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            features: [batch_size, input_dim]  float TF-IDF vector
            labels  : [batch_size]             long, optional

        Returns:
            dict with keys 'logits' (and 'loss' when labels are given)
        """
        x = self.embedding(features)                         # [B, 256]
        primary = self.primary_capsules(x)                   # [B, 32, 8]
        class_caps = self.class_capsules(primary)            # [B, K, 16]

        logits = self.classifier(class_caps.view(class_caps.size(0), -1))

        result = {'logits': logits}
        if labels is not None:
            result['loss'] = nn.CrossEntropyLoss()(logits, labels)
        return result
