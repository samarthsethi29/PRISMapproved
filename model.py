"""
Multi-Task GNN for PRISM target activity prediction.

Architecture:
  • 4× GINEConv layers (GIN + edge features) with batch norm
  • Global mean + max pooling → 2×hidden_dim vector
  • Multi-task head: one sigmoid output per target
    (Phase 3 = ABL1, c-KIT, PDGFRB)

Designed so adding more targets is a one-line change.

Label convention (IMPORTANT):
  Per-molecule labels are stored as y of shape [1, K] so that
  torch_geometric Batch.from_data_list batches them to [B, K],
  matching the model's logit output. A flat [K] tensor gets
  *concatenated* by PyG into [B*K], silently breaking the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch

NUM_ATOM_FEATURES = 7
NUM_EDGE_FEATURES = 8

# Phase 3 Targets (ABL1 / c-KIT / PDGFRB)
TARGET_NAMES = ["ABL1", "c-KIT", "PDGFRB"]


class MultiTargetGNN(nn.Module):
    """
    Args:
        hidden_dim:   width of GIN hidden layers (default 256)
        num_layers:   number of GINEConv message-passing layers (default 4)
        dropout:      dropout probability (default 0.2)
        target_names: list of target names (one output head each)
    """

    def __init__(
        self,
        hidden_dim:   int       = 256,
        num_layers:   int       = 4,
        dropout:      float     = 0.2,
        target_names: list[str] = None,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.dropout      = dropout
        self.target_names = target_names or TARGET_NAMES
        self.num_tasks    = len(self.target_names)

        # ── Input projection ─────────────────────────────────────────────────
        self.atom_proj = nn.Linear(NUM_ATOM_FEATURES, hidden_dim)
        self.edge_proj = nn.Linear(NUM_EDGE_FEATURES, hidden_dim)

        # ── GINEConv layers ──────────────────────────────────────────────────
        self.convs  = nn.ModuleList()
        self.bns    = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ── Global pooling → graph embedding ─────────────────────────────────
        pool_dim = hidden_dim * 2

        # ── Shared MLP before task heads ─────────────────────────────────────
        self.shared_mlp = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Per-target output heads ───────────────────────────────────────────
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(self.num_tasks)
        ])

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Returns logits of shape [B, K] where K is the number of targets
        and B the number of molecules in the batch.
        Use logits with MaskedMultiTaskLoss; apply sigmoid for inference.
        Expects data.y (if present) of shape [B, K] — see module docstring.
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Project to hidden_dim
        x         = F.relu(self.atom_proj(x))
        edge_attr = F.relu(self.edge_proj(edge_attr))

        # Message passing
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling
        x_mean = global_mean_pool(x, batch)   # [B, hidden_dim]
        x_max  = global_max_pool(x, batch)    # [B, hidden_dim]
        x_pool = torch.cat([x_mean, x_max], dim=1)  # [B, 2*hidden_dim]

        # Shared MLP
        h = self.shared_mlp(x_pool)   # [B, hidden_dim]

        # Per-target logits -> stack to [B, K]
        logits_list = [head(h) for head in self.heads]
        return torch.cat(logits_list, dim=1)

    def get_embedding(self, data: Batch) -> torch.Tensor:
        """
        Return the hidden_dim-dim molecular embedding (after shared MLP).
        Used for UMAP visualisation.
        """
        self.eval()
        with torch.no_grad():
            x, edge_index, edge_attr, batch = (
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            x         = F.relu(self.atom_proj(x))
            edge_attr = F.relu(self.edge_proj(edge_attr))
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index, edge_attr)
                x = bn(x)
                x = F.relu(x)
            x_mean = global_mean_pool(x, batch)
            x_max  = global_max_pool(x, batch)
            x_pool = torch.cat([x_mean, x_max], dim=1)
            return self.shared_mlp(x_pool)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Custom Masked Loss (PRISM multi-task formulation) ────────────────────────

class MaskedMultiTaskLoss(nn.Module):
    """
    Loss = (1/K) × Σᵢ [ (1/|Mᵢ|) × Σⱼ∈Mᵢ BCE(ŷᵢⱼ, yᵢⱼ) × pos_weightⱼ ]

    Where:
      - K = total number of targets
      - i = molecule index in batch
      - Mᵢ = set of targets with VALID labels for molecule i (ignores -1)
      - j = target index
      - pos_weight = task-specific weight to handle class imbalance
    """

    def __init__(self, pos_weights: torch.Tensor):
        """
        Args:
            pos_weights: Tensor of shape [K] containing pos_weight for each target
        """
        super().__init__()
        self.pos_weights = pos_weights
        self.K = len(pos_weights)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds:   [B, K] raw logits
            targets: [B, K] ground truth (0.0, 1.0, or -1.0 for missing)
        """
        # 1. Create valid mask (True where label is not -1)
        valid_mask = (targets != -1).float()  # [B, K]

        # 2. Calculate raw BCE with pos_weights.
        # We clamp targets to [0, 1] for the BCE math, but the mask handles ignoring them.
        bce = F.binary_cross_entropy_with_logits(
            preds,
            targets.clamp(min=0.0),
            pos_weight=self.pos_weights,
            reduction='none'
        )  # [B, K]

        # 3. Apply mask (zeroes out loss for missing targets)
        masked_bce = bce * valid_mask  # [B, K]

        # 4. Calculate |Mᵢ| (number of valid targets per molecule)
        Mi = valid_mask.sum(dim=1).clamp(min=1)  # [B] (clamp to avoid div by 0)

        # 5. (1/|Mᵢ|) × Σⱼ∈Mᵢ BCE
        per_molecule_loss = masked_bce.sum(dim=1) / Mi  # [B]

        # 6. (1/K) × Σᵢ
        loss = (1.0 / self.K) * per_molecule_loss.sum()

        return loss


# ── Quick architecture test ──────────────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    def dummy_molecule(n_atoms=12, n_bonds=13):
        x          = torch.randn(n_atoms, NUM_ATOM_FEATURES)
        src        = torch.randint(0, n_atoms, (n_bonds * 2,))
        dst        = torch.randint(0, n_atoms, (n_bonds * 2,))
        edge_index = torch.stack([src, dst])
        edge_attr  = torch.randn(n_bonds * 2, NUM_EDGE_FEATURES)
        # Labels stored as [1, K] so PyG batches to [B, K]:
        # a flat [K] y would be concatenated to [B*K] and break the loss.
        # -1 means missing label.
        y = torch.tensor([[1.0, -1.0, 0.0]])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    batch = Batch.from_data_list([dummy_molecule() for _ in range(4)])

    model = MultiTargetGNN(hidden_dim=256, num_layers=4, dropout=0.2)
    print(f"Parameters: {model.count_parameters():,}")
    print(f"  batch.y shape: {tuple(batch.y.shape)} (must be [4, 3])")

    model.eval()
    with torch.no_grad():
        logits = model(batch)  # Shape: [4, 3]
    print(f"  Output logits shape: {tuple(logits.shape)}")

    # Test the loss math with -1 (missing) labels present
    dummy_pos_weights = torch.tensor([0.16, 0.20, 0.43])  # realistic Phase 3 weights
    loss_fn = MaskedMultiTaskLoss(dummy_pos_weights)

    dummy_targets = batch.y  # [4, 3], contains -1s
    loss = loss_fn(logits, dummy_targets)
    print(f"  Masked Multi-Task Loss: {loss.item():.4f}")

    emb = model.get_embedding(batch)
    print(f"  Embedding shape: {tuple(emb.shape)}")  # [4, 256]
    print("\n✓ Architecture & Loss OK")
