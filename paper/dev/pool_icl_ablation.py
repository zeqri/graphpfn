"""Ablation: pool-then-ICL architecture for graph-level prediction.

Pipeline per sampled synthetic dataset (N molecules):
  1. atoms' raw features -> per-atom embedder (Linear -> emsize)
  2. mean-pool atoms' embeddings within each molecule -> one emsize vector/molecule
  3. inject context molecules' true label (query molecules get a learned
     "no label" placeholder instead) -> one token per molecule
  4. n_blocks x [masked sample-level MHA (real ICL: query attends only to
     context, context attends to all context) -> FF
     -> full unmasked MHA (every molecule attends to every molecule,
        replacing the graph adapter's now-meaningless adjacency) -> FF]
  5. regression head -> per-molecule prediction; loss/R^2 computed on query only

Labels come from the same virtual-node + graph-conv-SCM mechanism established
earlier (graph_then_attributes prior, molecule-skeleton sub-graphs, virtual
node star-connected per molecule, GeometricConv-based SCM). Atom features fed
to THIS model are each atom's own raw feature vector (no mean-pooling at the
data level this time -- the model does its own learned pooling).
"""
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

REPO = Path("/p/project1/profound/al-zeqri1/PFN/second/graphpfn/paper")
sys.path.insert(0, str(REPO))

from lib.graphpfn.prior.config import sample_configs  # noqa: E402
from lib.graphpfn.prior.graphs.multi_graph import sample_multi_graph  # noqa: E402
from lib.graphpfn.prior.attributes import sample_attributes_gnn  # noqa: E402
from lib.graphpfn.prior.postprocessing import process_features  # noqa: E402

from sklearn.metrics import r2_score  # noqa: E402

TOML_PATH = REPO / "exp/graphpfn/pretrain/multigraph_molecule_geometric_rbf_only_warmstart/pretrain.toml"
SENTINEL_DISTANCE = 0.0
N_FEATURES_FIXED = 16  # override the sampled n_features so the embedder has a fixed input dim
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base_prior_config() -> dict:
    with open(TOML_PATH, "rb") as f:
        toml = tomllib.load(f)
    return toml["base_config"]["prior"]


def add_virtual_node_and_get_labels(graph: dgl.DGLGraph, scm_config: dict):
    """Add one star-connected virtual node per sub-graph, run the GNN-SCM over
    the augmented graph to get a graph-level label per molecule (via the
    virtual node), then return per-molecule ATOM feature lists (raw, no
    pooling) + the per-molecule labels. The virtual node itself is discarded
    after label extraction -- it never becomes model input.
    """
    counts = graph.batch_num_nodes().tolist()
    n_atoms_total = graph.num_nodes()
    n_graphs = len(counts)

    src, dst = graph.edges()
    src, dst = src.tolist(), dst.tolist()
    distance = graph.edata.get("distance")
    distance = distance.tolist() if distance is not None else [SENTINEL_DISTANCE] * len(src)

    atom_idx_per_molecule: list[np.ndarray] = []
    new_src, new_dst, new_distance = list(src), list(dst), list(distance)
    offset = 0
    for g_i, n in enumerate(counts):
        atoms = np.arange(offset, offset + n)
        atom_idx_per_molecule.append(atoms)
        virtual_idx = n_atoms_total + g_i
        for a in atoms:
            new_src.append(int(a)); new_dst.append(virtual_idx); new_distance.append(SENTINEL_DISTANCE)
            new_src.append(virtual_idx); new_dst.append(int(a)); new_distance.append(SENTINEL_DISTANCE)
        offset += n

    total_nodes = n_atoms_total + n_graphs
    graph_aug = dgl.graph((torch.tensor(new_src), torch.tensor(new_dst)), num_nodes=total_nodes)
    graph_aug.edata["distance"] = torch.tensor(new_distance, dtype=torch.float32)

    X, y = sample_attributes_gnn(graph_aug, scm_config)
    X = process_features(X, p_cat=0.0, max_categories=256, do_permute_features=False)
    X, y = X.numpy(), y.numpy()

    atom_features_per_molecule = [X[atoms] for atoms in atom_idx_per_molecule]
    y_per_molecule = y[n_atoms_total:]  # virtual-node rows, one per molecule, in molecule order
    return atom_features_per_molecule, y_per_molecule


def sample_one_dataset(base_prior_config: dict):
    configs = sample_configs(base_prior_config, batch_size=1)
    config = configs[0]["prior"]
    config["scm"]["base"]["n_features"] = N_FEATURES_FIXED  # fix embedder input dim across datasets

    sampler_cfg = config["graph"]["sampler"]
    graph = sample_multi_graph(
        n_graphs=sampler_cfg["n_graphs"],
        sub_graph=sampler_cfg["sub_graph"],
        base_n_nodes=sampler_cfg.get("base_n_nodes"),
        size_jitter=sampler_cfg.get("size_jitter", 0.0),
    )
    atom_features_per_molecule, y_per_molecule = add_virtual_node_and_get_labels(graph, config["scm"])
    train_ratio = config["train_ratio"]
    return atom_features_per_molecule, y_per_molecule, train_ratio


def context_query_split(n_graphs: int, train_ratio: float, rng: np.random.Generator):
    n_train = max(1, min(n_graphs - 1, round(n_graphs * train_ratio)))
    perm = rng.permutation(n_graphs)
    return perm[:n_train], perm[n_train:]


def standardize_by_context(y_all: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Z-score y using context-set statistics only (mirrors the real training
    loop's features_mean/std computed over train_mask, and apply_task's own
    standard_scale_labels -- both skipped in this prototype's data path)."""
    mean = y_all[train_idx].mean()
    std = y_all[train_idx].std()
    std = std if std > 1e-6 else 1.0
    return (y_all - mean) / std


class PoolICLModel(nn.Module):
    def __init__(self, n_features: int, emsize: int = 64, n_heads: int = 4, n_blocks: int = 2):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(n_features, emsize), nn.ReLU(), nn.Linear(emsize, emsize)
        )
        self.y_embed = nn.Linear(1, emsize)
        self.no_label = nn.Parameter(torch.zeros(emsize))
        self.input_norm = nn.LayerNorm(emsize)

        self.sample_mha = nn.ModuleList([nn.MultiheadAttention(emsize, n_heads, batch_first=True) for _ in range(n_blocks)])
        self.sample_ff = nn.ModuleList([nn.Sequential(nn.Linear(emsize, emsize * 2), nn.ReLU(), nn.Linear(emsize * 2, emsize)) for _ in range(n_blocks)])
        self.sample_norm1 = nn.ModuleList([nn.LayerNorm(emsize) for _ in range(n_blocks)])
        self.sample_norm2 = nn.ModuleList([nn.LayerNorm(emsize) for _ in range(n_blocks)])

        self.full_mha = nn.ModuleList([nn.MultiheadAttention(emsize, n_heads, batch_first=True) for _ in range(n_blocks)])
        self.full_ff = nn.ModuleList([nn.Sequential(nn.Linear(emsize, emsize * 2), nn.ReLU(), nn.Linear(emsize * 2, emsize)) for _ in range(n_blocks)])
        self.full_norm1 = nn.ModuleList([nn.LayerNorm(emsize) for _ in range(n_blocks)])
        self.full_norm2 = nn.ModuleList([nn.LayerNorm(emsize) for _ in range(n_blocks)])

        self.head = nn.Linear(emsize, 1)
        self.n_blocks = n_blocks

    def forward(self, atom_features_per_molecule, y_context, train_idx, test_idx):
        n_graphs = len(atom_features_per_molecule)
        pooled = []
        for atoms in atom_features_per_molecule:
            atoms_t = torch.tensor(atoms, dtype=torch.float32, device=DEVICE)
            emb = self.embedder(atoms_t)  # (n_atoms, emsize)
            pooled.append(emb.mean(dim=0))  # mean-pool -> (emsize,)
        h = torch.stack(pooled, dim=0)  # (n_graphs, emsize)

        is_context = torch.zeros(n_graphs, dtype=torch.bool)
        is_context[train_idx] = True
        label_inject = self.no_label.unsqueeze(0).repeat(n_graphs, 1).clone()
        y_ctx_t = torch.tensor(y_context, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        label_inject[is_context] = self.y_embed(y_ctx_t)
        h = self.input_norm(h + label_inject).unsqueeze(0)  # (1, n_graphs, emsize)

        # ICL mask: query rows may only attend to context rows; context rows
        # attend to all context rows. additive mask, -inf = disallowed.
        icl_mask = torch.zeros(n_graphs, n_graphs, device=DEVICE)
        icl_mask[:, ~is_context] = float("-inf")  # nobody attends to query columns
        icl_mask.fill_diagonal_(0.0)  # allow self-attention regardless

        for i in range(self.n_blocks):
            attn_out, _ = self.sample_mha[i](h, h, h, attn_mask=icl_mask)
            h = self.sample_norm1[i](h + attn_out)
            h = self.sample_norm2[i](h + self.sample_ff[i](h))

            attn_out, _ = self.full_mha[i](h, h, h)  # unmasked: everyone attends to everyone
            h = self.full_norm1[i](h + attn_out)
            h = self.full_norm2[i](h + self.full_ff[i](h))

        pred = self.head(h.squeeze(0)).squeeze(-1)  # (n_graphs,)
        return pred


def main(n_train_steps: int = 400, n_eval_datasets: int = 150, seed: int = 0):
    print(f"device = {DEVICE}")
    base_prior_config = load_base_prior_config()
    model = PoolICLModel(n_features=N_FEATURES_FIXED).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)

    print("=== training ===")
    train_losses = []
    step = 0
    attempts = 0
    while step < n_train_steps and attempts < n_train_steps * 3:
        attempts += 1
        try:
            atom_feats, y_all, train_ratio = sample_one_dataset(base_prior_config)
        except Exception:
            continue
        n_graphs = len(atom_feats)
        if n_graphs < 4:
            continue
        train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
        if len(test_idx) == 0 or len(train_idx) < 2:
            continue
        y_std = standardize_by_context(y_all, train_idx)

        model.train()
        optimizer.zero_grad()
        pred = model(atom_feats, y_std[train_idx], train_idx, test_idx)
        loss = F.mse_loss(pred[test_idx], torch.tensor(y_std[test_idx], dtype=torch.float32, device=DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
        step += 1
        if step % 50 == 0:
            print(f"[step {step}/{n_train_steps}] recent loss = {np.mean(train_losses[-50:]):.4f}")

    print(f"\n=== evaluating on {n_eval_datasets} held-out sampled datasets ===")
    model.eval()
    scores = []
    n_skipped = 0
    with torch.no_grad():
        i = 0
        attempts = 0
        while i < n_eval_datasets and attempts < n_eval_datasets * 3:
            attempts += 1
            try:
                atom_feats, y_all, train_ratio = sample_one_dataset(base_prior_config)
            except Exception:
                n_skipped += 1
                continue
            n_graphs = len(atom_feats)
            if n_graphs < 4:
                n_skipped += 1
                continue
            train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
            if len(test_idx) == 0 or len(train_idx) < 2 or np.std(y_all[test_idx]) == 0:
                n_skipped += 1
                continue
            y_std = standardize_by_context(y_all, train_idx)
            pred = model(atom_feats, y_std[train_idx], train_idx, test_idx)
            # R^2 is invariant to a shared affine transform of pred & truth,
            # so scoring in standardized space equals scoring in raw space.
            scores.append(r2_score(y_std[test_idx], pred[test_idx].cpu().numpy()))
            i += 1
            if i % 25 == 0:
                print(f"[{i}/{n_eval_datasets}] evaluated, skipped={n_skipped}")

    arr = np.array(scores)
    print(f"\n=== Pool+ICL model R^2 (n={len(arr)}, skipped={n_skipped}) ===")
    print(f"mean={arr.mean():.3f} median={np.median(arr):.3f}")
    print(f"quantiles [10,25,50,75,90]%: {np.round(np.quantile(arr, [0.1,0.25,0.5,0.75,0.9]), 3)}")
    print(f"frac R2>0.5: {(arr > 0.5).mean():.3f}  frac R2>0.8: {(arr > 0.8).mean():.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train_steps", type=int, default=400)
    parser.add_argument("--n_eval_datasets", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(n_train_steps=args.n_train_steps, n_eval_datasets=args.n_eval_datasets, seed=args.seed)
