"""Check whether LimiX's own feat-dim -> model-dim conversion (`encoder_x`,
i.e. `MaskEmbEncoder` built via `get_x_encoder`) produces non-degenerate
per-graph pooled embeddings when fed molecules sampled from our synthetic
prior -- a candidate first stage for a graph pooler ("each graph -> one
hidden-state representation, paired with the virtual-node label").

Deliberately does NOT use Mole-BERT: Mole-BERT's pretrained checkpoint needs
real RDKit atom-type/bond-type integers our synthetic prior doesn't produce
(established earlier this session -- molecule_skeleton.py discards the real
element/bond-order info it computes internally). LimiX's encoder_x has no
such vocabulary mismatch -- it's built generically for whatever num_features
a dataset has (exactly the "feat dim -> model dim" mechanism traced at the
very start of this investigation, vendor/limix/model/encoders.py's
get_x_encoder/MaskEmbEncoder), which is exactly our prior's native
representation (continuous SCM-generated features, no chemistry vocabulary
needed).

This uses a FRESH (randomly initialized), not pretrained, encoder_x -- the
question here is only "is the mechanism itself capable of producing
non-degenerate per-atom/per-graph embeddings from this data," not "does the
specific pretrained LimiX-16M checkpoint do so" (that would need the real
checkpoint downloaded and loaded, a separate, heavier question).

Usage: python dev/limix_encoder_pooling_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lib.graphpfn.prior.graph_level import sample_graph_level_dataset  # noqa: E402
from vendor.limix.model.encoders import get_x_encoder  # noqa: E402

SEED = 0
EMBED_DIM = 192  # matches LimiX-16M's real embed_dim (GraphPFNGraphAttentionModule(d=192, ...))
MASK_EMBEDDING_SIZE = EMBED_DIM  # mask_embedding is expand_as'd against the embedding_dim-sized
# per-feature embedding (encoders.py:349), so this must equal EMBED_DIM, not an independent size.

# LimiX's actual mechanism for handling datasets with DIFFERENT numbers of
# features (which is exactly why we want its encoder here -- the prior
# samples varying n_features per dataset): features are chunked into small
# FIXED-size groups (features_per_group, matching LimiX-16M's own config),
# padded up to a multiple of that size, and MaskEmbEncoder is built ONCE with
# num_features=features_per_group (fixed, tiny) -- NOT tied to any dataset's
# actual feature count. The number of groups then varies freely per dataset
# (n_features=16 -> 8 groups, n_features=71 -> 36 groups, etc.), with the
# SAME shared per-group encoder applied regardless. This is what lets one
# encoder instance generalize across every dataset the prior can sample --
# an earlier version of this file used num_features=n_features (one group
# holding all of a dataset's features), which only works for datasets that
# all happen to share the same feature count, defeating the point.
FEATURES_PER_GROUP = 2


def pad_to_group_multiple(atom_features: torch.Tensor, features_per_group: int = FEATURES_PER_GROUP) -> torch.Tensor:
    n_features = atom_features.shape[-1]
    remainder = n_features % features_per_group
    if remainder == 0:
        return atom_features
    pad_width = features_per_group - remainder
    pad = torch.zeros(*atom_features.shape[:-1], pad_width, dtype=atom_features.dtype, device=atom_features.device)
    return torch.cat([atom_features, pad], dim=-1)


def build_grouped_x_encoder(embed_dim: int = EMBED_DIM, features_per_group: int = FEATURES_PER_GROUP):
    """A single x_encoder instance, reusable across datasets with ANY
    n_features -- num_features is fixed to features_per_group, not to any
    particular dataset's feature count.
    """
    return get_x_encoder(
        num_features=features_per_group,
        embedding_size=embed_dim,
        mask_embedding_size=embed_dim,
        encoder_use_bias=True,
        numeric_embed_type="linear",
    )


def encode_atoms_grouped(
    x_encoder, atom_features: torch.Tensor, features_per_group: int = FEATURES_PER_GROUP
) -> torch.Tensor:
    """(n_atoms, n_features) -> (n_atoms, n_groups, embed_dim). n_groups
    depends on this dataset's n_features (padded up to a multiple of
    features_per_group); x_encoder itself is unaffected by n_features.
    """
    n_atoms = atom_features.shape[0]
    padded = pad_to_group_multiple(atom_features, features_per_group)
    n_groups = padded.shape[-1] // features_per_group
    data = padded.reshape(1, n_atoms, n_groups, features_per_group)
    nan_encoding = torch.zeros_like(data)
    out = x_encoder({"data": data, "nan_encoding": nan_encoding})
    return out["data"].squeeze(0)  # (n_atoms, n_groups, embed_dim)


def n_groups_for(n_features: int, features_per_group: int = FEATURES_PER_GROUP) -> int:
    """How many groups encode_atoms_grouped will produce for a dataset with
    this many raw features -- needed up front to size GroupFusion.
    """
    return -(-n_features // features_per_group)  # ceil division


class GroupFusion(torch.nn.Module):
    """Collapses (n_atoms, n_groups, embed_dim) -> (n_atoms, embed_dim).

    Mean-pooling groups (the first version of these scripts) turned out to
    destroy real signal: every group is encoded by the SAME small shared
    numeric_mlp/fusion_network inside MaskEmbEncoder (that's what makes it
    reusable across datasets with different n_features), and averaging their
    outputs both shrinks capacity (a 16-feature atom's information gets
    squeezed through a 2-feature-wide encoder+mean instead of the original
    approach's full 16-wide fusion_network) AND discards which group is
    which -- there's no positional signal telling the model group 0's
    embedding apart from group 5's once they're averaged together (unlike
    the real backbone's add_embeddings, which adds a per-group positional
    embedding specifically so groups stay distinguishable through the rest
    of the pipeline). Concatenating instead of averaging preserves both:
    each group's full embedding survives, in its own fixed position in the
    concatenated vector, and one Linear layer learns how to combine them.

    Unlike x_encoder itself, this IS sized to a specific dataset's n_groups
    (hence indirectly n_features) -- that's fine, this is a per-run fusion
    layer, not the reusable-across-datasets encoder the earlier scope was
    about.
    """

    def __init__(self, n_groups: int, embed_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(n_groups * embed_dim, embed_dim)

    def forward(self, atom_embeddings_grouped: torch.Tensor) -> torch.Tensor:
        # (n_atoms, n_groups, embed_dim) -> (n_atoms, n_groups * embed_dim) -> (n_atoms, embed_dim)
        return self.proj(atom_embeddings_grouped.flatten(1))

# A fixed prior config for sampling many molecules -- reuses the same
# molecule-skeleton + SCM shape as graph_level_prior_fit_test.py, just needs
# to be shaped for sample_graph_level_dataset (a "sample_configs"-ready
# distribution config, batch_size=1 internally).
BASE_PRIOR_CONFIG = {
    "sanity_check": {"min_features": 4, "min_train_ratio": 0.05, "max_train_ratio": 0.5},
    "prior": {
        "_distribution_": "choice",
        "_shared_": True,
        "values": [
            {
                "_type_": "graph_then_attributes",
                "graph": {
                    "n_nodes": {"_distribution_": "choice", "values": [3000], "_shared_": True},
                    "avg_degree": {"_distribution_": "choice", "values": [2.2], "_shared_": True},
                    "sampler": {
                        "_distribution_": "choice",
                        "values": [
                            {
                                "_type_": "multi-graph",
                                "size_jitter": 0.15,
                                "n_graphs": {"_distribution_": "choice", "values": [3000]},
                                "base_n_nodes": {"_distribution_": "choice", "values": [25]},
                                "sub_graph": {
                                    "avg_degree": {"_distribution_": "choice", "values": [2.2]},
                                    "sampler": {
                                        "_distribution_": "choice",
                                        "values": [
                                            {
                                                "_type_": "molecule-skeleton",
                                                "heavy_atom_fraction": 0.42,
                                                "min_ring_size": 3,
                                                "max_ring_size": 7,
                                            }
                                        ],
                                    },
                                },
                            }
                        ],
                    },
                },
                "scm": {
                    "_type_": "gnn",
                    "base": {
                        "_type_": "mlp",
                        "n_features": {"_distribution_": "choice", "values": [16], "_shared_": True},
                        "n_layers": {"_distribution_": "choice", "values": [3]},
                        "hidden_dim": {"_distribution_": "choice", "values": [32]},
                        "activation_type": {"_distribution_": "choice", "values": ["tanh"]},
                        "causes": {
                            "strategy": {"_distribution_": "choice", "values": ["normal"]},
                            "pre_sample_stats": {"_distribution_": "choice", "values": [False]},
                        },
                        "noise": {
                            "std": {"_distribution_": "choice", "values": [0.02]},
                            "pre_sample_std": {"_distribution_": "choice", "values": [False]},
                        },
                        "init": {
                            "std": {"_distribution_": "choice", "values": [1.0]},
                            "block_wise_dropout": {"_distribution_": "choice", "values": [False]},
                            "p_dropout": {"_distribution_": "choice", "values": [0.0]},
                            "scale_std_by_dropout": {"_distribution_": "choice", "values": [True]},
                        },
                        "causal": {
                            "enabled": {"_distribution_": "choice", "values": [False]},
                            "y_is_effect": {"_distribution_": "choice", "values": [True]},
                            "in_clique": {"_distribution_": "choice", "values": [True]},
                            "sort_features": {"_distribution_": "choice", "values": [True]},
                        },
                    },
                    "conv_type": {"_distribution_": "choice", "values": ["sage-mean"]},
                    "graph_conv_ratio": {"_distribution_": "choice", "values": [1.0]},
                    "structural": {
                        "use_degree": {"_distribution_": "choice", "values": [True]},
                        "use_pagerank": {"_distribution_": "choice", "values": [False]},
                        "lappe_k": 0,
                    },
                },
                "postprocessing": {"p_cat": 0.0, "max_categories": 256, "permute_features": False},
                "train_ratio": {"_distribution_": "choice", "values": [0.8], "_shared_": True},
            }
        ],
    },
}


def encode_and_pool(
    atom_features: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs LimiX's encoder_x (feat-dim -> model-dim, proper features_per_group
    chunking -- see build_grouped_x_encoder/encode_atoms_grouped), pools
    atoms->molecules per group, then pools groups->one vector per molecule.
    Returns (per_atom_embeddings [group-pooled, for reporting], per_molecule_pooled).
    """
    n_atoms = atom_features.shape[0]
    x_encoder = build_grouped_x_encoder(EMBED_DIM)
    x_encoder.eval()

    with torch.no_grad():
        atom_embeddings_grouped = encode_atoms_grouped(x_encoder, atom_features)  # (n_atoms, n_groups, embed_dim)
    n_groups = atom_embeddings_grouped.shape[1]
    group_fusion = GroupFusion(n_groups=n_groups, embed_dim=EMBED_DIM)
    group_fusion.eval()

    pooled_grouped = torch.zeros(n_molecules, n_groups, EMBED_DIM)
    counts = torch.zeros(n_molecules, 1, 1)
    pooled_grouped = pooled_grouped.index_add(0, molecule_id, atom_embeddings_grouped)
    counts = counts.index_add(0, molecule_id, torch.ones(n_atoms, 1, 1))
    pooled_grouped = pooled_grouped / counts.clamp(min=1)  # (n_molecules, n_groups, embed_dim)

    # Fuse groups -> one embed_dim vector per molecule (see GroupFusion's
    # docstring for why concatenation+Linear replaced a naive groups-mean).
    with torch.no_grad():
        pooled = group_fusion(pooled_grouped)
        atom_embeddings = group_fusion(atom_embeddings_grouped)  # for reporting only

    return atom_embeddings, pooled


def report_degeneracy(pooled: torch.Tensor) -> None:
    n_molecules, dim = pooled.shape
    per_dim_std = pooled.std(dim=0)
    print(f"\nper-molecule pooled embeddings: {pooled.shape}")
    print(
        f"per-dimension std across molecules: mean={per_dim_std.mean().item():.4f}, "
        f"min={per_dim_std.min().item():.4f}, max={per_dim_std.max().item():.4f}, "
        f"n_dims_near_zero_std(<1e-4)={int((per_dim_std < 1e-4).sum().item())}/{dim}"
    )

    normed = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    n_sample = min(500, n_molecules)
    idx = torch.from_numpy(np.random.choice(n_molecules, size=n_sample, replace=False))
    sub = normed[idx]
    cos_sim = sub @ sub.T
    off_diag = cos_sim[~torch.eye(n_sample, dtype=torch.bool)]
    print(
        f"pairwise cosine similarity (off-diagonal, {n_sample} sampled molecules): "
        f"mean={off_diag.mean().item():.4f}, std={off_diag.std().item():.4f}, "
        f"max={off_diag.max().item():.4f}"
    )

    centered = pooled - pooled.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    explained = (singular_values**2) / (singular_values**2).sum()
    cumulative = torch.cumsum(explained, dim=0)
    n_dims_for_95pct = int((cumulative < 0.95).sum().item()) + 1
    print(
        f"effective rank: {n_dims_for_95pct}/{min(pooled.shape)} dims needed for 95% variance "
        f"(top-5 singular values: {singular_values[:5].tolist()})"
    )

    mean_cos = off_diag.mean().item()
    if per_dim_std.mean().item() < 1e-3 or n_dims_for_95pct <= 2:
        verdict = "DEGENERATE (collapsed to ~a point or a 1-2 dim subspace)"
    elif mean_cos > 0.95:
        verdict = (
            "PARTIALLY DEGENERATE: real multi-dim per-molecule variation exists "
            f"({n_dims_for_95pct} dims), but it's small relative to a large shared/common "
            f"component every molecule's embedding has (mean cosine sim {mean_cos:.3f} -- "
            "two well-separated random vectors would be near 0). Expected for an UNtrained "
            "encoder (nothing has been optimized to push molecules apart yet, and mean-pooling "
            "over many atoms further shrinks per-molecule variance) -- this measures whether the "
            "architecture is CAPABLE of carrying signal, not whether it currently discriminates "
            "well. The real test is whether that signal grows once trained end-to-end."
        )
    else:
        verdict = "non-degenerate"
    print(f"\nverdict: {verdict}")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Sampling molecules from the synthetic prior...")
    atom_features, graph, y_per_molecule, _ = sample_graph_level_dataset(BASE_PRIOR_CONFIG)
    counts = graph.batch_num_nodes()
    n_molecules = counts.shape[0]
    molecule_id = torch.repeat_interleave(torch.arange(n_molecules), counts)
    print(f"n_atoms={graph.num_nodes()}, n_molecules={n_molecules}, n_features={atom_features.shape[-1]}")

    atom_embeddings, pooled = encode_and_pool(atom_features, molecule_id, n_molecules)
    print(f"per-atom embeddings: {atom_embeddings.shape} (LimiX encoder_x output, embed_dim={EMBED_DIM})")

    report_degeneracy(pooled)


if __name__ == "__main__":
    main()
