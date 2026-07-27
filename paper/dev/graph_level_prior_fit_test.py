"""Sanity-check the graph-level virtual-node label prior.

Fixes ONE SCM instantiation, generates many independent molecules under that
*same* fixed function, and checks whether a plain supervised GNN can fit it
on held-out molecules (train/test split by molecule).

This is deliberately different from calling sample_graph_level_dataset (or
looping over sample_graph_level_labels_via_virtual_node) repeatedly:
sample_attributes_gnn builds and randomly re-initializes a FRESH SCM on
every call, so pooling molecules sampled across many calls would mix many
different label-generating functions into one training set -- not learnable
by any fixed-weight model, by construction, regardless of how "good" the
prior is (that's exactly why in-context learning exists: no single trained
mapping can cover a family of tasks). Here, sample_multi_graph draws many
independent molecule *topologies*, but sample_graph_level_labels_via_virtual_node
is called exactly ONCE over the whole batch, so every molecule's label comes
from the identical SCM weights -- the meaningful question this answers is
"is a single realized instance of this prior's label function learnable."

Usage: python dev/graph_level_prior_fit_test.py
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

from lib.graphpfn.prior.attributes import sample_graph_level_labels_via_virtual_node  # noqa: E402
from lib.graphpfn.prior.graphs.multi_graph import sample_multi_graph  # noqa: E402

SEED = 0
N_MOLECULES = 3000
BASE_N_NODES = 25
TRAIN_FRACTION = 0.8
N_EPOCHS = 300
HIDDEN = 64
N_GNN_LAYERS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
LOG_EVERY = 20

# A fixed (non-distribution, concrete) sub-graph topology config -- only the
# molecule *structure* varies molecule-to-molecule; the SCM below is drawn
# once and applied identically to all of them.
SUB_GRAPH_CONFIG = {
    "avg_degree": 2.2,
    "sampler": {
        "_type_": "molecule-skeleton",
        "heavy_atom_fraction": 0.42,
        "min_ring_size": 3,
        "max_ring_size": 7,
    },
}

# A fixed SCM config. conv_type="sage-mean" + graph_conv_ratio=1.0: every
# node's hidden state at every layer is pure neighbor aggregation (no
# self-fallback), so the virtual node's final output is a genuine function
# of its molecule's atoms/topology, propagated through n_layers of
# message-passing -- not just a shallow linear readout.
SCM_CONFIG = {
    "base": {
        "n_features": 16,
        "n_layers": 3,
        "hidden_dim": 32,
        "activation_type": "tanh",
        "causes": {"strategy": "normal", "pre_sample_stats": False},
        "noise": {"std": 0.02, "pre_sample_std": False},
        "init": {
            "std": 1.0,
            "block_wise_dropout": False,
            "p_dropout": 0.0,
            "scale_std_by_dropout": True,
        },
        "causal": {
            "enabled": False,
            "y_is_effect": True,
            "in_clique": True,
            "sort_features": True,
        },
    },
    "conv_type": "sage-mean",
    "graph_conv_ratio": 1.0,
    "structural": {"use_degree": True, "use_pagerank": False, "lappe_k": 0},
}


class SimpleGNN(nn.Module):
    """Plain supervised GIN-ish GNN: fixed input dim, mean-pool readout."""

    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            dglnn.SAGEConv(hidden, hidden, aggregator_type="mean") for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(feat)
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

    print(f"Sampling {N_MOLECULES} molecules under ONE fixed SCM instantiation...")
    graph = sample_multi_graph(
        n_graphs=N_MOLECULES,
        sub_graph=SUB_GRAPH_CONFIG,
        base_n_nodes=BASE_N_NODES,
        size_jitter=0.15,
    )
    atom_features, y_per_molecule = sample_graph_level_labels_via_virtual_node(graph, SCM_CONFIG)
    print(
        f"n_atoms={graph.num_nodes()}, n_molecules={y_per_molecule.shape[0]}, "
        f"y mean={y_per_molecule.mean().item():.4f}, y std={y_per_molecule.std().item():.4f}"
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

    model = SimpleGNN(in_dim=atom_features.shape[-1], hidden=HIDDEN, n_layers=N_GNN_LAYERS, dropout=DROPOUT)
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
