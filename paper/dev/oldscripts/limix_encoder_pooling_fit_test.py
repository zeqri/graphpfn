"""Train LimiX's encoder_x (feat-dim -> model-dim) + mean-pool + a small
regression head end-to-end against the virtual-node graph-level label, to see
whether the "small signal on a large shared baseline" pattern found in
dev/limix_encoder_pooling_probe.py (untrained) actually grows once trained --
same question the plain-GNN fit tests answered for a from-scratch GNN, now
for this LimiX-native pooler.

Supervised regression (encoder_x -> mean-pool -> MLP head -> predict
y_per_molecule via MSE), same methodology as graph_level_prior_fit_test.py
and node_level_prior_fit_test.py -- NOT Mole-BERT's actual (label-free,
masking + contrastive) training recipe.

Important architectural note: unlike graph_level_prior_fit_test.py's GNN
(SAGEConv message passing over real bonds), this pooler is graph-structure
BLIND -- encoder_x embeds each atom independently (no neighbor information
at all), and pooling is a plain mean over a molecule's atoms (DeepSets-style,
like pooling.py's SumPooler, not GeometricAttentionStack). So this result is
also informative about how much of the label's fittable signal comes from
atom composition alone vs. requires topology -- if this pooler's test R2
lands well below the SAGEConv version's ~0.65-0.69, that's evidence the label
depends on bond structure, not just which/how-many atoms are present.

Usage: python dev/limix_encoder_pooling_fit_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lib.graphpfn.prior.graph_level import sample_graph_level_dataset  # noqa: E402
from dev.limix_encoder_pooling_probe import (  # noqa: E402
    BASE_PRIOR_CONFIG,
    EMBED_DIM,
    GroupFusion,
    build_grouped_x_encoder,
    encode_atoms_grouped,
    n_groups_for,
    report_degeneracy,
)

SEED = 0
TRAIN_FRACTION = 0.8
N_EPOCHS = 300
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
LR = 1e-3
LOG_EVERY = 20


class LimixPoolerRegressor(nn.Module):
    """encoder_x (per-atom, feat-dim -> model-dim, proper features_per_group
    chunking so this same module works for ANY dataset's n_features) ->
    mean-pool atoms->molecules per group -> GroupFusion (concat+Linear)
    groups->one vector -> small MLP regression head. No message passing --
    pooling atoms is a plain per-molecule mean over atom embeddings.
    """

    def __init__(self, n_features: int, embed_dim: int, dropout: float):
        super().__init__()
        self.x_encoder = build_grouped_x_encoder(embed_dim)
        self.group_fusion = GroupFusion(n_groups=n_groups_for(n_features), embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, 1)
        )

    def encode_and_pool(
        self, atom_features: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int
    ) -> torch.Tensor:
        n_atoms = atom_features.shape[0]
        atom_embeddings_grouped = encode_atoms_grouped(self.x_encoder, atom_features)  # (n_atoms, n_groups, embed_dim)
        n_groups, embed_dim = atom_embeddings_grouped.shape[1:]

        pooled_grouped = torch.zeros(n_molecules, n_groups, embed_dim, device=atom_features.device)
        counts = torch.zeros(n_molecules, 1, 1, device=atom_features.device)
        pooled_grouped = pooled_grouped.index_add(0, molecule_id, atom_embeddings_grouped)
        counts = counts.index_add(0, molecule_id, torch.ones(n_atoms, 1, 1, device=atom_features.device))
        pooled_grouped = pooled_grouped / counts.clamp(min=1)  # (n_molecules, n_groups, embed_dim)

        return self.group_fusion(pooled_grouped)  # (n_molecules, embed_dim)

    def forward(self, atom_features: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int) -> torch.Tensor:
        pooled = self.encode_and_pool(atom_features, molecule_id, n_molecules)
        return self.head(pooled).squeeze(-1)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Sampling molecules from the synthetic prior...")
    atom_features, graph, y_per_molecule, _ = sample_graph_level_dataset(BASE_PRIOR_CONFIG)
    counts = graph.batch_num_nodes()
    n_molecules = counts.shape[0]
    molecule_id = torch.repeat_interleave(torch.arange(n_molecules), counts)
    print(f"n_atoms={graph.num_nodes()}, n_molecules={n_molecules}, n_features={atom_features.shape[-1]}")

    perm = np.random.permutation(n_molecules)
    n_train = int(n_molecules * TRAIN_FRACTION)
    train_mol_idx = torch.from_numpy(perm[:n_train])
    test_mol_idx = torch.from_numpy(perm[n_train:])

    y_mean = y_per_molecule[train_mol_idx].mean()
    y_std = y_per_molecule[train_mol_idx].std().clamp(min=1e-6)
    y_norm = (y_per_molecule - y_mean) / y_std

    model = LimixPoolerRegressor(n_features=atom_features.shape[-1], embed_dim=EMBED_DIM, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_test_r2 = -float("inf")
    best_epoch = -1

    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(atom_features, molecule_id, n_molecules)
        loss = F.mse_loss(pred[train_mol_idx], y_norm[train_mol_idx])
        loss.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                pred = model(atom_features, molecule_id, n_molecules)
                train_r2 = r2_score(pred[train_mol_idx], y_norm[train_mol_idx])
                test_r2 = r2_score(pred[test_mol_idx], y_norm[test_mol_idx])
            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                best_epoch = epoch
            print(
                f"epoch {epoch:4d} | loss {loss.item():.4f} | "
                f"train R2 {train_r2:.4f} | test R2 {test_r2:.4f}"
            )

    print(f"\nbest test R2 = {best_test_r2:.4f} at epoch {best_epoch} (early-stopping proxy)")

    print("\n--- degeneracy check on TRAINED pooled embeddings (all molecules) ---")
    model.eval()
    with torch.no_grad():
        pooled = model.encode_and_pool(atom_features, molecule_id, n_molecules)
    report_degeneracy(pooled)


if __name__ == "__main__":
    main()
