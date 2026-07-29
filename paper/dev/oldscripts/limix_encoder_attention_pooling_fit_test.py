"""Same as limix_encoder_pooling_fit_test.py, but replaces the plain mean-pool
with a LEARNABLE attention pooler: one shared learnable query per molecule
cross-attends over that molecule's own atom embeddings (post encoder_x) and
combines them via content-dependent softmax weights + a multi-head output
projection, instead of a fixed uniform average. Mean-pool has no learnable
parameters at all in the pooling step itself -- it can't learn to weight
atoms differently based on content, only the encoder and the head can adapt.
This pooler adds real learnable capacity specifically to the pooling step.

Still edge-feature-blind by design (per instruction): the pooling graph
connects each atom only to its own molecule's query node (a bipartite
atom->query graph), not to other atoms via real bonds -- no message passing
between atoms, no edge/bond information used at all. This isolates whether a
more expressive (but still structure-blind) aggregation over the SET of atom
embeddings already helps, before adding real edge-based message passing as a
separate, later improvement.

Architecture: AttentionPooler is adapted from SparseAttentionPooler (found
in the nested repo's lib/graphpfn/pooling.py during earlier exploration this
session) -- same sparse bipartite dgl.ops attention pattern
(u_dot_v/edge_softmax/u_mul_e_sum), O(n_atoms) not O(n_molecules * n_atoms),
just simplified for our (n_atoms, embed_dim) 2D convention (no n_groups dim).

Usage: python dev/limix_encoder_attention_pooling_fit_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import dgl  # noqa: E402
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
POOLER_HEADS = 4


class AttentionPooler(nn.Module):
    """One shared learnable query cross-attends over each molecule's own
    atom embeddings, via a sparse bipartite atom->molecule-query dgl graph
    (edges: atom -> its own molecule's query node only). No real bonds used.
    """

    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads
        self.attn_scores_coef = 1.0 / self.d_head**0.5

        self.query = nn.Parameter(torch.empty(1, embed_dim))
        nn.init.xavier_uniform_(self.query)
        self.kv_linear = nn.Linear(embed_dim, embed_dim * 2)
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, atom_embeddings: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int
    ) -> torch.Tensor:
        n_atoms, embed_dim = atom_embeddings.shape
        device = atom_embeddings.device

        kv = self.kv_linear(atom_embeddings).reshape(n_atoms, self.n_heads, 2 * self.d_head)
        k, v = kv.split(self.d_head, dim=-1)  # each (n_atoms, n_heads, d_head)

        q_query = self.query.reshape(1, self.n_heads, self.d_head).expand(n_molecules, -1, -1)
        # Atom nodes are never a `dst` (only query nodes receive edges), so
        # their own q slot / query nodes' own k,v slots are never read --
        # zero-pad for a uniform node-indexed tensor as dgl.ops requires.
        q_atoms_pad = torch.zeros(n_atoms, self.n_heads, self.d_head, device=device)
        kv_query_pad = torch.zeros(n_molecules, self.n_heads, self.d_head, device=device)
        q_full = torch.cat([q_atoms_pad, q_query], dim=0)
        k_full = torch.cat([k, kv_query_pad], dim=0)
        v_full = torch.cat([v, kv_query_pad], dim=0)

        src = torch.arange(n_atoms, device=device)
        dst = n_atoms + molecule_id
        pool_graph = dgl.graph((src, dst), num_nodes=n_atoms + n_molecules)

        attn_scores = dgl.ops.u_dot_v(pool_graph, k_full, q_full) * self.attn_scores_coef
        attn_probs = dgl.ops.edge_softmax(pool_graph, attn_scores)
        pooled_full = dgl.ops.u_mul_e_sum(pool_graph, v_full, attn_probs)

        pooled = pooled_full[n_atoms:].reshape(n_molecules, embed_dim)
        return self.output_linear(pooled)


class LimixAttentionPoolerRegressor(nn.Module):
    """encoder_x (per-atom, feat-dim -> model-dim, proper features_per_group
    chunking so this module works for ANY dataset's n_features) -> GroupFusion
    (concat+Linear) groups into one per-atom vector -> learnable attention
    pool per molecule -> small MLP regression head.
    """

    def __init__(self, n_features: int, embed_dim: int, pooler_heads: int, dropout: float):
        super().__init__()
        self.x_encoder = build_grouped_x_encoder(embed_dim)
        self.group_fusion = GroupFusion(n_groups=n_groups_for(n_features), embed_dim=embed_dim)
        self.pooler = AttentionPooler(embed_dim=embed_dim, n_heads=pooler_heads)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, 1)
        )

    def encode_and_pool(
        self, atom_features: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int
    ) -> torch.Tensor:
        atom_embeddings_grouped = encode_atoms_grouped(self.x_encoder, atom_features)  # (n_atoms, n_groups, embed_dim)
        # The group split is an internal encoding-capacity device for a
        # single atom's own feature vector, not something the attention
        # query should treat as separate attendable items -- fuse it to one
        # embed_dim vector per atom before pooling atoms->molecule (see
        # GroupFusion's docstring for why concat+Linear, not mean).
        atom_embeddings = self.group_fusion(atom_embeddings_grouped)  # (n_atoms, embed_dim)
        return self.pooler(atom_embeddings, molecule_id, n_molecules)

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

    model = LimixAttentionPoolerRegressor(
        n_features=atom_features.shape[-1], embed_dim=EMBED_DIM, pooler_heads=POOLER_HEADS, dropout=DROPOUT
    )
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
