"""Real ZINC molecule loading for graph-level periodic evaluation.

Mirrors `qm9_data.py`, but for the standard ZINC-12k benchmark subset (the
one reported in the SAT paper's Table 1, among others): `torch_geometric`'s
`ZINC(subset=True)` splits it into train/val/test files of 10,000/1,000/1,000
molecules, which this module concatenates back into one 12,000-molecule pool
to sample from -- same "never trained on" premise as `qm9_data.py` (the
model is evaluated purely via ICL, never fit to real ZINC), so there's no
train/eval leakage to guard against by keeping the official splits separate.

Unlike QM9, ZINC has no 3D conformers (`torch_geometric.datasets.ZINC` has no
`pos`), so no `edata["distance"]` is set -- `GraphPFNGraphAttentionModule`
already treats "distance" as optional (falls back to plain topology-masked
attention, see `model.py`), so this is a no-op architecturally, not a
workaround. ZINC's `edge_attr` (bond type) is likewise dropped, matching the
QM9 loader's own choice not to feed bond order into `atom_features` either --
only node features and graph topology are used by the model.

ZINC's `x` is a single categorical atom-type index (`0..27`) rather than
QM9's 11-column mostly-one-hot feature block, but it is fed to the model
exactly as-is (cast to float), matching the QM9 loader's own "no
special-casing" convention -- `GraphLevelGraphPFN`'s `x_preprocess`/
`encoder_x` pipeline is a generic TabPFN-style feature encoder that doesn't
require one-hot input, and `pooling.py` already pads the feature width to a
multiple of `features_per_group`, so the width mismatch with QM9 (1 vs. 11)
needs no extra handling either.
"""

from __future__ import annotations

import dgl
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torch_geometric.datasets import ZINC


def load_zinc(root: str, subset: bool = True) -> ConcatDataset:
    """Loads all three official splits and concatenates them into one pool
    (10,000 + 1,000 + 1,000 = 12,000 molecules for `subset=True`, matching
    the "# GRAPHS: 12,000" reported for ZINC in the SAT paper's Table 1)."""
    splits = [ZINC(root=root, subset=subset, split=s) for s in ("train", "val", "test")]
    return ConcatDataset(splits)


def sample_one_zinc_dataset(
    dataset: ConcatDataset,
    n_graphs_range: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dgl.DGLGraph, torch.Tensor]:
    """Draw a random subset of real ZINC molecules as one eval "dataset".

    Returns (atom_features, graph, y_per_molecule) -- same shape convention
    as `sample_graph_level_dataset`/`sample_one_qm9_dataset`, so the caller's
    context/query split + standardization + GraphLevelGraphPFN.forward call
    is identical across synthetic, QM9, and ZINC data.
    """
    n_graphs = int(rng.integers(n_graphs_range[0], n_graphs_range[1] + 1))
    mol_indices = rng.choice(len(dataset), size=n_graphs, replace=False)

    feature_chunks = []
    subgraphs = []
    labels = []
    for idx in mol_indices:
        d = dataset[int(idx)]
        feature_chunks.append(d.x.float())
        src, dst = d.edge_index.numpy()
        g = dgl.graph(
            (torch.from_numpy(src).long(), torch.from_numpy(dst).long()),
            num_nodes=d.x.shape[0],
        )
        subgraphs.append(g)
        labels.append(float(d.y.item()))

    atom_features = torch.cat(feature_chunks, dim=0)
    graph = dgl.batch(subgraphs)
    y_per_molecule = torch.tensor(labels, dtype=torch.float32)
    return atom_features, graph, y_per_molecule
