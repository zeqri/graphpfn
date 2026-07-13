"""GCN graph-pooling encoder + LimiX ICL, combined into one graph-level model.

Each graph (e.g. a molecule) is pooled by a node+edge-feature GCN into a
single embedding row; the set of embeddings across many graphs is then
treated as an ordinary tabular dataset for LimiX's in-context learning, with
context graphs supplying (embedding, label) pairs and query graphs supplying
only embeddings to predict. This sidesteps GraphPFN's own node-level-only
pretraining (it has no native graph-level task support) by using LimiX
purely as a tabular ICL classifier/regressor sitting on top of a trained
graph encoder, rather than the paper's node-level message-passing adapters.
"""

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpfn.model.limix import LimiXWrapper
from graphpfn.util import TaskType


class EdgeConditionedGCNEncoder(nn.Module):
    """Node+edge-feature GCN that pools each graph into one embedding vector.

    Uses edge-conditioned convolution (`dgl.nn.NNConv`, the message function
    from Gilmer et al. 2017's MPNN): each edge's features parameterize a
    per-edge weight matrix applied to the sender node's hidden state, so
    edge attributes (e.g. QM9's single/double/triple/aromatic bond-type
    one-hot) directly shape message passing rather than only node identity.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        n_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        def make_edge_func() -> nn.Module:
            return nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )

        self.convs = nn.ModuleList(
            [
                dglnn.NNConv(hidden_dim, hidden_dim, make_edge_func(), aggregator_type="mean")
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        # mean + sum pooling concatenated: mean captures average local
        # environment, sum additionally carries a size/extensiveness signal.
        self.readout_proj = nn.Linear(2 * hidden_dim, embed_dim)

    def forward(self, graph: dgl.DGLGraph, node_feats: Tensor, edge_feats: Tensor) -> Tensor:
        h = F.relu(self.node_encoder(node_feats))
        for conv in self.convs:
            h = self.dropout(F.relu(conv(graph, h, edge_feats)))

        with graph.local_scope():
            graph.ndata["h"] = h
            pooled = torch.cat(
                [dgl.mean_nodes(graph, "h"), dgl.sum_nodes(graph, "h")], dim=-1
            )  # [n_graphs, 2 * hidden_dim]

        return self.readout_proj(pooled)  # [n_graphs, embed_dim]


class LimixGCN(nn.Module):
    """Graph-level regression/classification: GCN pooling -> LimiX ICL.

    LimiX's weights are frozen by default -- only `encoder` is trained -- to
    preserve its pretrained ICL prior, mirroring GraphPFN's own pretraining
    (which freezes the LimiX backbone and only trains new graph-aware
    components; GraphPFN paper, Section 3.2).
    """

    def __init__(
        self,
        encoder: EdgeConditionedGCNEncoder,
        *,
        freeze_limix: bool = True,
        limix_load_weights: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.limix = LimiXWrapper(load_weights=limix_load_weights)
        self.freeze_limix = freeze_limix
        if freeze_limix:
            for p in self.limix.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True) -> "LimixGCN":
        super().train(mode)
        if self.freeze_limix:
            # Frozen LimiX must stay in eval mode (no dropout) regardless of
            # the outer module's train()/eval() calls.
            self.limix.eval()
        return self

    def embed_graphs(
        self, batched_graph: dgl.DGLGraph, node_feats: Tensor, edge_feats: Tensor
    ) -> Tensor:
        return self.encoder(batched_graph, node_feats, edge_feats)

    def forward(
        self,
        *,
        batched_graph: dgl.DGLGraph,
        node_feats: Tensor,
        edge_feats: Tensor,
        context_mask: Tensor,
        y_context: Tensor,
        task_type: TaskType,
    ) -> Tensor:
        """
        Args:
            batched_graph: `dgl.batch` of one graph per data point (e.g. one
                molecule per graph).
            node_feats: [n_nodes_total, node_feat_dim], all graphs concatenated.
            edge_feats: [n_edges_total, edge_feat_dim], all graphs concatenated.
            context_mask: [n_graphs] bool, True = context (labeled) graph.
            y_context: [n_context] labels, in the same order as
                `context_mask.nonzero()`.
            task_type: "regression", "binclass", or "multiclass".

        Returns:
            LimiX's raw prediction for the query graphs (~context_mask):
            [n_query] for regression, [n_query, n_classes] log-probs for
            classification.
        """
        embeddings = self.embed_graphs(batched_graph, node_feats, edge_feats)
        x_context = embeddings[context_mask]
        x_query = embeddings[~context_mask]
        return self.limix(
            x_train=x_context,
            y_train=y_context,
            x_eval=x_query,
            task_type=task_type,
        )
