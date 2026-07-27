"""Sanity-check the ORIGINAL paper's node-level prior, for comparison against
dev/graph_level_prior_fit_test.py's graph-level (virtual-node) result.

Uses the actual prior distribution from
exp/graphpfn/pretrain/main/pretrain.toml (the main/original pretraining
config: single large graph via "multi-level-sbm-with-pa", n_nodes in
[1000, 5000], SCM hyperparameters drawn from that config's own
distributions) -- not a hand-rolled config -- so this is a genuine
apples-to-apples comparison against the same prior family the original paper
pretrains on. One config is sampled (fixing one graph-sampler draw AND one
SCM instantiation together, same "fix the generating function" logic as the
graph-level test), producing ONE single connected graph with one label per
NODE (not per molecule -- no virtual node here, this is the original
mechanism: sample_attributes_gnn called directly on the graph). Train/test
split is by NODE (transductive), and the label is used as a raw continuous
regression target regardless of what task type the config's `task._type_`
distribution happened to draw (classification conversion, apply_task, is
skipped entirely) -- exactly mirroring the graph-level test's own choice to
read the raw SCM output directly.

Usage: python dev/node_level_prior_fit_test.py
"""

from __future__ import annotations

import random
import sys
import tomllib
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

from lib.graphpfn.prior.attributes import sample_attributes_gnn  # noqa: E402
from lib.graphpfn.prior.config import sample_configs  # noqa: E402
from lib.graphpfn.prior.graphs import sample_graph  # noqa: E402

CONFIG_PATH = PAPER_DIR / "exp/graphpfn/pretrain/main/pretrain.toml"

SEED = 0
TRAIN_FRACTION = 0.8
N_EPOCHS = 300
HIDDEN = 64
N_GNN_LAYERS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
LOG_EVERY = 20
MAX_SAMPLE_ATTEMPTS = 20

# If True, override just the two SCM axes identified as likely making the
# previous draw hard for a plain GNN (causal.enabled -> False: y becomes the
# SCM's direct final-layer output instead of an arbitrary shared-window slice
# of intermediate activations that need not be forward-causal; graph_conv_ratio
# -> 1.0: every hidden channel at every layer sees neighbor-aggregated
# information instead of only a random subset of channels doing so -- see
# MixedGraphLinear in attributes/layers.py). Graph topology (sampler type,
# n_nodes, avg_degree) and everything else about the SCM (n_features,
# n_layers, hidden_dim, conv_type, structural features) are left exactly as
# drawn from the real prior distribution -- this isolates whether those two
# axes specifically were the source of the difficulty, rather than switching
# to a different prior altogether.
SIMPLIFY_SCM = True


def load_prior_distribution_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)
    return data["base_config"]["prior"]


class SimpleNodeGNN(nn.Module):
    """Same backbone as graph_level_prior_fit_test.py's SimpleGNN, but with a
    direct per-node regression head instead of a pooled readout.
    """

    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            dglnn.SAGEConv(hidden, hidden, aggregator_type="mean") for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(feat)
        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(F.relu(norm(conv(g, h))))
        return self.head(h).squeeze(-1)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def sample_one_dataset(dist_config: dict) -> tuple[torch.Tensor, dgl.DGLGraph, torch.Tensor, dict]:
    """Draws ONE config (one graph-sampler + one SCM instantiation) from the
    main pretrain.toml's prior distribution, retrying on SanityCheckError-style
    failures (e.g. graph shrinks too much after largest-component extraction).
    """
    last_err = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        try:
            config = sample_configs(dist_config, batch_size=1)[0]["prior"]
            as_sampled = {
                "causal.enabled": config["scm"]["base"]["causal"]["enabled"],
                "graph_conv_ratio": config["scm"]["graph_conv_ratio"],
            }
            if SIMPLIFY_SCM:
                config["scm"]["base"]["causal"]["enabled"] = False
                config["scm"]["graph_conv_ratio"] = 1.0
                print(f"SIMPLIFY_SCM=True: overriding {as_sampled} -> "
                      f"{{'causal.enabled': False, 'graph_conv_ratio': 1.0}}")
            graph = sample_graph(config["graph"])
            features, labels = sample_attributes_gnn(graph, config["scm"])
            return features, graph, labels, config
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not sample a valid dataset after {MAX_SAMPLE_ATTEMPTS} attempts") from last_err


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)  # several SCM sampling paths (e.g. extract_features_and_labels's
    # causal-clique offset, layers.py's dropout init) use Python's own random
    # module, not numpy/torch -- without this, config *hyperparameters* still
    # reproduce (numpy-driven) but realized graph/weight/feature values don't.

    dist_config = load_prior_distribution_config()
    print(f"Sampling ONE dataset from {CONFIG_PATH.relative_to(PAPER_DIR)}'s prior "
          "(one graph draw + one fixed SCM instantiation)...")
    atom_features, graph, y_per_node, config = sample_one_dataset(dist_config)

    scm_base = config["scm"]["base"]
    print(
        f"n_nodes={graph.num_nodes()}, n_features={atom_features.shape[-1]}, "
        f"sampled task._type_={config['task']['_type_']!r} (ignored -- using raw "
        f"regression target), conv_type={config['scm']['conv_type']!r}, "
        f"graph_conv_ratio={config['scm']['graph_conv_ratio']}, "
        f"causal.enabled={scm_base['causal']['enabled']}, "
        f"n_layers={scm_base['n_layers']}, hidden_dim={scm_base['hidden_dim']}, "
        f"y mean={y_per_node.mean().item():.4f}, y std={y_per_node.std().item():.4f}"
    )

    n_nodes = graph.num_nodes()
    perm = np.random.permutation(n_nodes)
    n_train = int(n_nodes * TRAIN_FRACTION)
    train_idx = torch.from_numpy(perm[:n_train])
    test_idx = torch.from_numpy(perm[n_train:])

    y_mean = y_per_node[train_idx].mean()
    y_std = y_per_node[train_idx].std().clamp(min=1e-6)
    y_norm = (y_per_node - y_mean) / y_std

    model = SimpleNodeGNN(
        in_dim=atom_features.shape[-1], hidden=HIDDEN, n_layers=N_GNN_LAYERS, dropout=DROPOUT
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_test_r2 = -float("inf")
    best_epoch = -1

    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(graph, atom_features)
        loss = F.mse_loss(pred[train_idx], y_norm[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                pred = model(graph, atom_features)
                train_r2 = r2_score(pred[train_idx], y_norm[train_idx])
                test_r2 = r2_score(pred[test_idx], y_norm[test_idx])
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
