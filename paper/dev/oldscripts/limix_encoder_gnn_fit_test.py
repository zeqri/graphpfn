"""Same SAGEConv message-passing GNN as graph_level_prior_fit_test.py
(real bonds used, mean-pool readout), but with LimiX's own encoder_x
(feat-dim -> model-dim, vendor/limix/model/encoders.py's MaskEmbEncoder)
swapped in as the input embedding stage, instead of a plain trainable
nn.Linear projection. Checks whether using LimiX's own feature encoder
changes performance relative to the plain-Linear-input GNN result
(peaked ~0.65-0.69 test R2) and relative to the edge-blind pooling-only
LimiX-encoder tests (mean-pool ~0.46 plateau, attention-pool TBD) -- this
is the "encoder + real message passing" combination, the piece those two
were missing.

Uses the same sampled-dataset convention (BASE_PRIOR_CONFIG,
sample_graph_level_dataset) as limix_encoder_pooling_fit_test.py and
limix_encoder_attention_pooling_fit_test.py for a consistent 3-way (now
4-way) comparison: mean-pool, attention-pool, this (encoder+GNN), and the
original plain-Linear-input GNN.

Usage: python dev/limix_encoder_gnn_fit_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import dgl  # noqa: E402
import dgl.nn.pytorch as dglnn  # noqa: E402
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
)

SEED = 0
TRAIN_FRACTION = 0.8
N_EPOCHS = 300
N_GNN_LAYERS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
LOG_EVERY = 20


class LimixEncoderGNN(nn.Module):
    """encoder_x (per-atom, feat-dim -> model-dim, proper features_per_group
    chunking so this module works for ANY dataset's n_features) -> SAGEConv
    message passing over real bonds -> mean-pool per molecule -> MLP head.
    Same conv/readout shape as graph_level_prior_fit_test.py's SimpleGNN,
    with encoder_x replacing that script's plain nn.Linear input_proj.
    """

    def __init__(self, n_features: int, embed_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.x_encoder = build_grouped_x_encoder(embed_dim)
        self.group_fusion = GroupFusion(n_groups=n_groups_for(n_features), embed_dim=embed_dim)
        self.convs = nn.ModuleList(
            dglnn.SAGEConv(embed_dim, embed_dim, aggregator_type="mean") for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.readout_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, 1)
        )

    def encode(self, atom_features: torch.Tensor) -> torch.Tensor:
        atom_embeddings_grouped = encode_atoms_grouped(self.x_encoder, atom_features)  # (n_atoms, n_groups, embed_dim)
        # The group split is an internal encoding-capacity device for a
        # single atom's own feature vector, not something message passing
        # should treat as separate nodes/channels -- fuse it to one embed_dim
        # vector per atom before the GNN layers (see GroupFusion's docstring
        # for why concat+Linear, not mean).
        return self.group_fusion(atom_embeddings_grouped)  # (n_atoms, embed_dim)

    def forward(self, g: dgl.DGLGraph, atom_features: torch.Tensor) -> torch.Tensor:
        h = self.encode(atom_features)
        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(F.relu(norm(conv(g, h))))
        g.ndata["h"] = h
        pooled = dgl.mean_nodes(g, "h")
        return self.readout_mlp(pooled).squeeze(-1)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Sampling molecules from the synthetic prior...")
    atom_features, graph, y_per_molecule, _ = sample_graph_level_dataset(BASE_PRIOR_CONFIG)
    print(
        f"n_atoms={graph.num_nodes()}, n_molecules={graph.batch_num_nodes().shape[0]}, "
        f"n_features={atom_features.shape[-1]}"
    )

    graph.ndata["feat"] = atom_features
    mol_graphs = dgl.unbatch(graph)

    n_molecules = len(mol_graphs)
    perm = np.random.permutation(n_molecules)
    n_train = int(n_molecules * TRAIN_FRACTION)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    y_mean = y_per_molecule[train_idx].mean()
    y_std = y_per_molecule[train_idx].std().clamp(min=1e-6)
    y_norm = (y_per_molecule - y_mean) / y_std

    train_graph = dgl.batch([mol_graphs[i] for i in train_idx])
    test_graph = dgl.batch([mol_graphs[i] for i in test_idx])
    y_train = y_norm[torch.from_numpy(train_idx)]
    y_test = y_norm[torch.from_numpy(test_idx)]

    model = LimixEncoderGNN(
        n_features=atom_features.shape[-1], embed_dim=EMBED_DIM, n_layers=N_GNN_LAYERS, dropout=DROPOUT
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_test_r2 = -float("inf")
    best_epoch = -1

    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(train_graph, train_graph.ndata["feat"])
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                train_pred = model(train_graph, train_graph.ndata["feat"])
                test_pred = model(test_graph, test_graph.ndata["feat"])
                train_r2 = r2_score(train_pred, y_train)
                test_r2 = r2_score(test_pred, y_test)
            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                best_epoch = epoch
            print(
                f"epoch {epoch:4d} | loss {loss.item():.4f} | "
                f"train R2 {train_r2:.4f} | test R2 {test_r2:.4f}"
            )

    print(f"\nbest test R2 = {best_test_r2:.4f} at epoch {best_epoch} (early-stopping proxy)")


if __name__ == "__main__":
    main()
