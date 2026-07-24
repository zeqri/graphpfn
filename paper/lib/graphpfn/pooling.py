"""Graph-level ("whole-molecule") prediction via learnable pooling on top of
the LimiX/GraphPFN backbone.

Architecture (vectorized -- no Python loop over molecules, unlike the
original one-off sanity check at `dev/pool_icl_real_limix.py`):
  1. `encoder_x`/`x_preprocess` run ONCE over every atom of every molecule in
     the batch (fixes the small-sample-normalization problem the dev script
     had, since stats are now computed across the whole context population).
  2. Geometric atom-attention: reuses `GraphPFNGraphAttentionModule`
     unmodified, operating directly on the raw batched atom graph -- since
     `sample_multi_graph`/`dgl.batch` never add cross-molecule edges, this is
     already exactly block-diagonal per molecule, no extra masking needed.
  3. `SparseAttentionPooler`: reduces every molecule's atoms down to one
     token via a sparse bipartite atom->molecule-query graph (same
     `dgl.ops.u_dot_v`/`edge_softmax`/`u_mul_e_sum` pattern as step 2) --
     O(total_atoms), not the O(n_graphs * total_atoms) a dense masked
     attention would cost at this scale.
  4. The pooled per-molecule tokens go through the real `FeaturesTransformer`
     machinery (`add_embeddings`, `mixed_y_embedding`, `transformer_encoder`,
     `y_decoder`) unchanged, with the backbone's graph-adapter repurposed to
     do full attention among the (now molecule-granularity) pooled tokens.
"""

from __future__ import annotations

import dgl
import torch
import torch.nn as nn

from .model import GraphPFN, GraphPFNGraphAttentionModule, GraphPFNResidualModule, SDPAInput


class GeometricAttentionStack(nn.Module):
    """`n_rounds` rounds of distance-biased self-attention among each
    molecule's own atoms, reusing `GraphPFNGraphAttentionModule` (and its
    residual wrapper) directly -- fresh, dedicated parameters, not shared
    with the backbone's own 12 wrapped layers.
    """

    def __init__(self, embed_dim: int, n_rounds: int = 1, n_heads: int = 4):
        super().__init__()
        self.rounds = nn.ModuleList(
            [
                GraphPFNResidualModule(
                    base=GraphPFNGraphAttentionModule(
                        d=embed_dim, n_heads=n_heads, zero_init=False
                    ),
                    d_hidden=embed_dim,
                )
                for _ in range(n_rounds)
            ]
        )

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        for round_ in self.rounds:
            x = round_(graph, x)
        return x


class SparseAttentionPooler(nn.Module):
    """Learnable, Set-Transformer-style pooling: a single shared learnable
    query cross-attends over each molecule's own atoms, via a sparse
    bipartite atom->molecule-query `dgl` graph (edges only from an atom to
    its own molecule's query node) instead of a dense
    (n_graphs, total_atoms) masked attention -- see module docstring.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4):
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
        self, atom_embeddings: torch.Tensor, mol_id: torch.Tensor, n_graphs: int
    ) -> torch.Tensor:
        # atom_embeddings: (1, total_atoms, n_groups, embed_dim)
        assert atom_embeddings.ndim == 4 and atom_embeddings.shape[0] == 1
        x = atom_embeddings.squeeze(0)  # (total_atoms, n_groups, embed_dim)
        total_atoms, n_groups, _ = x.shape
        device = x.device

        kv = self.kv_linear(x).reshape(total_atoms, n_groups, self.n_heads, 2 * self.d_head)
        k, v = kv.split(self.d_head, dim=-1)  # each (total_atoms, n_groups, n_heads, d_head)

        # One shared learnable query, broadcast to every molecule and group.
        q_query = (
            self.query.to(x.dtype)
            .reshape(1, 1, self.n_heads, self.d_head)
            .expand(n_graphs, n_groups, -1, -1)
        )
        # Atom nodes are never a `dst` (only query nodes receive edges), so
        # their own q/k/v "query-node" slot is never read -- zero-pad for a
        # uniform, single node-indexed tensor as dgl.ops requires.
        q_atoms_pad = torch.zeros(
            total_atoms, n_groups, self.n_heads, self.d_head, device=device, dtype=x.dtype
        )
        kv_query_pad = torch.zeros(
            n_graphs, n_groups, self.n_heads, self.d_head, device=device, dtype=x.dtype
        )
        q_full = torch.cat([q_atoms_pad, q_query], dim=0)
        k_full = torch.cat([k, kv_query_pad], dim=0)
        v_full = torch.cat([v, kv_query_pad], dim=0)

        src = torch.arange(total_atoms, device=device)
        dst = total_atoms + mol_id
        pool_graph = dgl.graph((src, dst), num_nodes=total_atoms + n_graphs)

        attn_scores = dgl.ops.u_dot_v(pool_graph, k_full, q_full) * self.attn_scores_coef
        attn_probs = dgl.ops.edge_softmax(pool_graph, attn_scores)
        pooled_full = dgl.ops.u_mul_e_sum(pool_graph, v_full, attn_probs)

        pooled = pooled_full[total_atoms:]  # (n_graphs, n_groups, n_heads, d_head)
        pooled = pooled.reshape(n_graphs, n_groups, self.embed_dim)
        pooled = self.output_linear(pooled)
        return pooled.unsqueeze(0)  # (1, n_graphs, n_groups, embed_dim)


class SumPooler(nn.Module):
    """EGNN-style graph-level readout (Satorras et al. 2021, Section 5.3,
    QM9 implementation details): a per-atom MLP, summed (not attention-
    averaged) over each molecule's atoms, followed by a post-pooling MLP.

    Unlike `SparseAttentionPooler`'s softmax-normalized attention -- a convex
    combination of atom vectors, structurally invariant to atom count except
    through whatever the embeddings themselves encode -- summing lets the
    pooled representation grow naturally with molecule size. That's the
    right inductive bias for *extensive* properties (total energy, zpve,
    atomization energy) that scale with atom count, which is exactly why
    EGNN's own QM9 readout uses sum-pooling rather than attention pooling.
    """

    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.pre_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embed_dim)
        )
        self.post_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embed_dim)
        )

    def forward(
        self, atom_embeddings: torch.Tensor, mol_id: torch.Tensor, n_graphs: int
    ) -> torch.Tensor:
        # atom_embeddings: (1, total_atoms, n_groups, embed_dim)
        assert atom_embeddings.ndim == 4 and atom_embeddings.shape[0] == 1
        x = atom_embeddings.squeeze(0)  # (total_atoms, n_groups, embed_dim)
        total_atoms, n_groups, embed_dim = x.shape

        x = self.pre_mlp(x)
        pooled = torch.zeros(n_graphs, n_groups, embed_dim, device=x.device, dtype=x.dtype)
        pooled.index_add_(0, mol_id, x)
        pooled = self.post_mlp(pooled)
        return pooled.unsqueeze(0)  # (1, n_graphs, n_groups, embed_dim)


class GraphLevelGraphPFN(nn.Module):
    """Wraps a `GraphPFN` instance to predict one label per molecule instead
    of one per node, via the pipeline in this module's docstring.
    """

    def __init__(
        self,
        graphpfn: GraphPFN,
        embed_dim: int,
        n_geometric_rounds: int = 1,
        geometric_heads: int = 4,
        pooler_heads: int = 4,
        pooler_type: str = "attention",
    ):
        super().__init__()
        self.graphpfn = graphpfn
        self.geometric_attention = GeometricAttentionStack(
            embed_dim=embed_dim, n_rounds=n_geometric_rounds, n_heads=geometric_heads
        )
        if pooler_type == "attention":
            self.pooler = SparseAttentionPooler(embed_dim=embed_dim, n_heads=pooler_heads)
        elif pooler_type == "sum":
            self.pooler = SumPooler(embed_dim=embed_dim)
        else:
            raise ValueError(f"unknown pooler_type={pooler_type!r}, expected 'attention' or 'sum'")

    def forward(
        self,
        molecule_graph: dgl.DGLGraph,
        atom_features: torch.Tensor,
        y_all: torch.Tensor,
        is_context_molecule: torch.Tensor,
        n_random_features: int = 0,
    ) -> torch.Tensor:
        """
        molecule_graph: block-diagonal dgl graph over every molecule's atoms
            (`molecule_graph.batch_num_nodes()` gives each molecule's atom
            count; edata["distance"] used by geometric attention if present).
        atom_features: (total_atoms, n_features), same node order as
            `molecule_graph`.
        y_all: (n_graphs,) float label per molecule, in `molecule_graph`'s
            `batch_num_nodes()` order -- context values are revealed to the
            model, query values are only used by the caller for the loss.
        is_context_molecule: (n_graphs,) bool, same order as `y_all`.

        Returns: (n_graphs,) predictions, in the ORIGINAL molecule order.
        """
        tfm = self.graphpfn.tfm.module
        device = atom_features.device

        if n_random_features > 0:
            random_features = torch.randn(
                [atom_features.shape[0], n_random_features], device=device
            )
            atom_features = torch.cat([atom_features, random_features], dim=-1)

        counts = molecule_graph.batch_num_nodes()
        n_graphs = counts.shape[0]
        n_atoms = molecule_graph.num_nodes()

        # >>> Vectorized context-first molecule/atom reorder (no Python loop).
        mol_perm = torch.argsort((~is_context_molecule).float(), stable=True)
        n_context = int(is_context_molecule.sum().item())

        orig_mol_id = torch.repeat_interleave(torch.arange(n_graphs, device=device), counts)
        rank = torch.empty(n_graphs, dtype=torch.long, device=device)
        rank[mol_perm] = torch.arange(n_graphs, device=device)
        atom_perm = torch.argsort(rank[orig_mol_id], stable=True)
        inv_atom_perm = torch.argsort(atom_perm, stable=True)

        src, dst = molecule_graph.edges()
        distance = molecule_graph.edata.get("distance")
        graph2 = dgl.graph(
            (inv_atom_perm[src], inv_atom_perm[dst]), num_nodes=n_atoms
        )
        if distance is not None:
            graph2.edata["distance"] = distance

        features2 = atom_features[atom_perm]
        counts_reordered = counts[mol_perm]
        mol_id_reordered = torch.repeat_interleave(
            torch.arange(n_graphs, device=device), counts_reordered
        )
        eval_pos = int(counts_reordered[:n_context].sum().item())
        # <<<

        num_feature = features2.shape[-1]
        x = {
            "data": features2.unsqueeze(0),
            "mask": torch.isnan(features2).to(torch.int32).unsqueeze(0),
        }
        feature_to_add = num_feature % tfm.features_per_group
        if feature_to_add > 0:
            for key in x:
                x[key] = torch.cat(
                    [
                        x[key],
                        torch.zeros(
                            1, n_atoms, feature_to_add, device=device, dtype=x[key].dtype
                        ),
                    ],
                    dim=-1,
                )
        for key in x:
            x[key] = x[key].reshape(
                1, n_atoms, x[key].shape[2] // tfm.features_per_group, tfm.features_per_group
            )
        x["eval_pos"] = eval_pos

        preprocessed = tfm.x_preprocess(x)
        preprocessed = tfm.process_4_x(preprocessed)
        x_encoder_result = tfm.encoder_x(preprocessed)
        x_emb = x_encoder_result["data"]  # (1, n_atoms, n_groups, embed_dim)

        x_emb = self.geometric_attention(graph2, x_emb)

        pooled = self.pooler(x_emb, mol_id_reordered, n_graphs)  # (1, n_graphs, n_groups, embed_dim)
        embedded_x = tfm.add_embeddings(pooled)

        y_full = torch.full((1, n_graphs), float("nan"), device=device, dtype=atom_features.dtype)
        y_full[0, :n_context] = y_all[mol_perm][:n_context].to(atom_features.dtype)
        y_dict = {"data": y_full.unsqueeze(-1)}
        y_type = torch.ones_like(y_dict["data"])
        embedded_y = tfm.mixed_y_embedding(y_dict, y_type=y_type, eval_pos=n_context)

        embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)

        attn_mask = torch.ones(n_graphs, n_graphs, dtype=torch.bool, device=device)
        zero_degree_mask = torch.zeros(n_graphs, dtype=torch.bool, device=device)
        self.graphpfn._graph_holder.graph = SDPAInput(
            attn_mask=attn_mask, zero_degree_mask=zero_degree_mask, edge_distance=None
        )
        try:
            with torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                ]
            ):
                encoder_out = tfm.transformer_encoder(
                    embedded_all, feature_atten_mask=None, eval_pos=n_context
                )[0]
        finally:
            self.graphpfn._graph_holder.graph = None

        encoder_out = tfm.encoder_out_norm(encoder_out)
        test_encoder_out = encoder_out[:, n_context:, -1]
        test_y_type = y_type[:, n_context:, 0]
        _, reg_pred = tfm.y_decoder(test_encoder_out, test_y_type)
        pred_reordered = reg_pred.float().squeeze(0).squeeze(-1)  # (n_query,)

        query_order_orig_idx = mol_perm[n_context:]
        pred_by_orig_idx = torch.zeros(n_graphs, device=device)
        pred_by_orig_idx[query_order_orig_idx] = pred_reordered
        return pred_by_orig_idx
