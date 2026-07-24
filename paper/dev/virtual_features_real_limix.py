"""Real-LimiX-backbone ablation, no pooling: each molecule is represented by
exactly ONE row -- the virtual node's own feature vector and label, both
obtained directly from the graph-conv-dependent GNN-SCM (via the same
star-connected virtual node mechanism used throughout this investigation),
with no mean-pooling of atoms at all. Atoms themselves are never fed to the
model in this variant.

Since every molecule is already a single row before the model ever sees it,
there's no need to hand-replicate FeaturesTransformer.forward's internals
(as pool_icl_real_limix.py has to, to insert pooling mid-forward) -- this
script just calls the real, unmodified GraphPFN.forward() directly. The
graph-adapter's message-passing attention is repurposed to full (fully
connected) attention among all molecules by simply passing a complete graph
over the N virtual-node rows as the `graph` argument -- no code changes to
model.py needed, exactly as established for the graph-adapter->full-attention
swap earlier.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import dgl

REPO = Path("/p/project1/profound/al-zeqri1/PFN/second/graphpfn/paper")
sys.path.insert(0, str(REPO))

# >>> Point the checkpoint loader at the local copy instead of HF Hub (no
# network access from compute nodes).
import lib.tfm.limix as limix_mod  # noqa: E402
_LOCAL_CKPT = str(REPO / "checkpoints/LimiX-16M.ckpt")
limix_mod._download_limix_checkpoint = lambda: _LOCAL_CKPT

from lib.graphpfn.model import GraphPFN  # noqa: E402
from lib.graphpfn.prior.config import sample_configs  # noqa: E402
from lib.graphpfn.prior.graphs.multi_graph import sample_multi_graph  # noqa: E402
from lib.graphpfn.prior.attributes import sample_attributes_gnn  # noqa: E402
from lib.graphpfn.prior.postprocessing import process_features  # noqa: E402
from lib.util import TaskType  # noqa: E402

import tomllib  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

TOML_PATH = REPO / "exp/graphpfn/pretrain/multigraph_molecule_geometric_rbf_only_warmstart/pretrain.toml"
SENTINEL_DISTANCE = 0.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base_prior_config() -> dict:
    with open(TOML_PATH, "rb") as f:
        toml = tomllib.load(f)
    return toml["base_config"]["prior"]


def add_virtual_node_and_get_features_and_labels(graph: dgl.DGLGraph, scm_config: dict):
    """Star-connect one virtual node per sub-graph, run the GNN-SCM over the
    augmented graph, and return BOTH the features and labels straight from
    the virtual nodes' own rows -- no mean-pooling, no atom features used at
    all. The virtual node's own X is just as graph-derived as its y (see
    earlier discussion: every recorded SCM layer output has already passed
    through >=2 graph-conv-capable layers), so this is a legitimate, if
    different, way to obtain a per-molecule feature vector.
    """
    counts = graph.batch_num_nodes().tolist()
    n_atoms_total = graph.num_nodes()
    n_graphs = len(counts)

    src, dst = graph.edges()
    src, dst = src.tolist(), dst.tolist()
    distance = graph.edata.get("distance")
    distance = distance.tolist() if distance is not None else [SENTINEL_DISTANCE] * len(src)

    new_src, new_dst, new_distance = list(src), list(dst), list(distance)
    offset = 0
    for g_i, n in enumerate(counts):
        virtual_idx = n_atoms_total + g_i
        for a in range(offset, offset + n):
            new_src.append(a); new_dst.append(virtual_idx); new_distance.append(SENTINEL_DISTANCE)
            new_src.append(virtual_idx); new_dst.append(a); new_distance.append(SENTINEL_DISTANCE)
        offset += n

    total_nodes = n_atoms_total + n_graphs
    graph_aug = dgl.graph((torch.tensor(new_src), torch.tensor(new_dst)), num_nodes=total_nodes)
    graph_aug.edata["distance"] = torch.tensor(new_distance, dtype=torch.float32)

    X, y = sample_attributes_gnn(graph_aug, scm_config)
    X = process_features(X, p_cat=0.0, max_categories=256, do_permute_features=False)
    X, y = X.numpy(), y.numpy()

    X_virtual = X[n_atoms_total:]  # each virtual node's own SCM-generated feature vector
    y_virtual = y[n_atoms_total:]  # each virtual node's own SCM-generated label
    return X_virtual, y_virtual


def sample_one_dataset(base_prior_config: dict):
    configs = sample_configs(base_prior_config, batch_size=1)
    config = configs[0]["prior"]

    sampler_cfg = config["graph"]["sampler"]
    graph = sample_multi_graph(
        n_graphs=sampler_cfg["n_graphs"],
        sub_graph=sampler_cfg["sub_graph"],
        base_n_nodes=sampler_cfg.get("base_n_nodes"),
        size_jitter=sampler_cfg.get("size_jitter", 0.0),
    )
    X_virtual, y_virtual = add_virtual_node_and_get_features_and_labels(graph, config["scm"])
    train_ratio = config["train_ratio"]
    return X_virtual, y_virtual, train_ratio


def context_query_split(n_graphs: int, train_ratio: float, rng: np.random.Generator):
    n_train = max(1, min(n_graphs - 1, round(n_graphs * train_ratio)))
    perm = rng.permutation(n_graphs)
    return perm[:n_train], perm[n_train:]


def standardize_by_context(y_all: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = y_all[train_idx].mean()
    std = y_all[train_idx].std()
    std = std if std > 1e-6 else 1.0
    return (y_all - mean) / std


def fully_connected_graph(n: int) -> dgl.DGLGraph:
    """Complete graph (with self-loops) over n molecule-rows -- this is what
    turns the graph-adapter's message-passing attention into full attention
    among all molecules, with no code changes to model.py: it just builds
    its dense attn_mask straight from this graph's edges.
    """
    idx = torch.arange(n)
    src, dst = torch.meshgrid(idx, idx, indexing="ij")
    return dgl.graph((src.reshape(-1), dst.reshape(-1)), num_nodes=n)


def build_model(random_adapter_init: bool = True) -> GraphPFN:
    model = GraphPFN(
        edge_head=None,
        feat_head=False,
        layer_ids=list(range(12)),
        freeze_tfm=True,
        random_init_tfm=False,
    )
    if random_adapter_init:
        for idx in range(12):
            layer = model.tfm.module.transformer_encoder.layers[idx]
            for module in [layer.conv.base, layer.mlp.base]:
                for p in module.parameters():
                    if p.dim() >= 2:
                        torch.nn.init.xavier_uniform_(p, gain=0.1)
                    else:
                        torch.nn.init.zeros_(p)
    model = model.to(DEVICE)
    model.eval()
    return model


def forward_one_dataset(model: GraphPFN, X_virtual, y_context, train_idx, n_graphs):
    features_t = torch.tensor(X_virtual, dtype=torch.float32, device=DEVICE)
    train_mask = torch.zeros(n_graphs, dtype=torch.bool, device=DEVICE)
    train_mask[train_idx] = True
    y_train_t = torch.tensor(y_context, dtype=torch.float32, device=DEVICE)
    graph = fully_connected_graph(n_graphs).to(DEVICE)

    out = model(
        graph=graph,
        features=features_t,
        y_train=y_train_t,
        train_mask=train_mask,
        task_type=TaskType.REGRESSION,
    )
    return out["predictions"]  # (n_graphs,), context positions are dummy zeros


def train_adapters(model: GraphPFN, base_prior_config: dict, n_steps: int, rng: np.random.Generator,
                    lr: float = 1e-3, max_norm: float = 1.0):
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"training {n_trainable}/{n_total} params (adapters only, backbone frozen)")
    optimizer = torch.optim.Adam(trainable, lr=lr)
    model.train()

    losses = []
    step = 0
    attempts = 0
    while step < n_steps and attempts < n_steps * 3:
        attempts += 1
        try:
            X_virtual, y_all, train_ratio = sample_one_dataset(base_prior_config)
        except Exception:
            continue
        n_graphs = len(y_all)
        if n_graphs < 4:
            continue
        train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
        if len(test_idx) == 0 or len(train_idx) < 2:
            continue
        y_std = standardize_by_context(y_all, train_idx)

        optimizer.zero_grad()
        pred = forward_one_dataset(model, X_virtual, y_std[train_idx], train_idx, n_graphs)
        loss = torch.nn.functional.mse_loss(
            pred[test_idx], torch.tensor(y_std[test_idx], dtype=torch.float32, device=DEVICE)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_norm)
        optimizer.step()
        losses.append(loss.item())
        step += 1
        if step % 10 == 0:
            print(f"[train step {step}/{n_steps}] recent loss = {np.mean(losses[-10:]):.4f}")

    model.eval()
    return losses


def main(n_eval_datasets: int = 60, seed: int = 0, random_adapter_init: bool = True, n_train_steps: int = 0):
    print(f"device = {DEVICE}")
    base_prior_config = load_base_prior_config()
    model = build_model(random_adapter_init=random_adapter_init and n_train_steps == 0)
    rng = np.random.default_rng(seed)

    if n_train_steps > 0:
        train_adapters(model, base_prior_config, n_train_steps, rng)

    scores = []
    n_skipped = 0
    i = 0
    attempts = 0
    with torch.no_grad():
        while i < n_eval_datasets and attempts < n_eval_datasets * 3:
            attempts += 1
            try:
                X_virtual, y_all, train_ratio = sample_one_dataset(base_prior_config)
            except Exception:
                n_skipped += 1
                continue
            n_graphs = len(y_all)
            if n_graphs < 4:
                n_skipped += 1
                continue
            train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
            if len(test_idx) == 0 or len(train_idx) < 2 or np.std(y_all[test_idx]) == 0:
                n_skipped += 1
                continue
            y_std = standardize_by_context(y_all, train_idx)
            try:
                pred = forward_one_dataset(model, X_virtual, y_std[train_idx], train_idx, n_graphs)
            except Exception as e:
                print(f"forward failed: {type(e).__name__}: {e}")
                n_skipped += 1
                continue
            scores.append(r2_score(y_std[test_idx], pred[test_idx].cpu().numpy()))
            i += 1
            if i % 10 == 0:
                print(f"[{i}/{n_eval_datasets}] evaluated, skipped={n_skipped}")

    arr = np.array(scores)
    print(f"\n=== Real-LimiX virtual-node-features (no pooling) R^2 (n={len(arr)}, skipped={n_skipped}) ===")
    if len(arr) == 0:
        print("no successful evaluations")
        return
    print(f"mean={arr.mean():.3f} median={np.median(arr):.3f}")
    print(f"quantiles [10,25,50,75,90]%: {np.round(np.quantile(arr, [0.1,0.25,0.5,0.75,0.9]), 3)}")
    print(f"frac R2>0.5: {(arr > 0.5).mean():.3f}  frac R2>0.8: {(arr > 0.8).mean():.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval_datasets", type=int, default=60)
    parser.add_argument("--n_train_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zero_init_adapter", action="store_true")
    args = parser.parse_args()
    main(
        n_eval_datasets=args.n_eval_datasets,
        seed=args.seed,
        random_adapter_init=not args.zero_init_adapter,
        n_train_steps=args.n_train_steps,
    )
