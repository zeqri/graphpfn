"""Architecture-only smoke test for graph-level regression via a virtual node.

Checks whether the existing GraphPFN virtual-node + adapter mechanism (no new
model code) can compute and regress a whole-graph structural statistic, using
one FIXED deterministic labeling function shared across all sampled graphs
(mean node degree). This deliberately removes any in-context-learning
requirement -- the model learns the function in its adapter weights across
steps, rather than inferring it per-dataset from labeled context. The point is
to isolate the architecture question ("can structure reach a query node and
get decoded into a scalar") from the prior-learning question.

Graph sizes are kept small on purpose: the backbone's ICL attention is O(n^2)
over the node/sample axis, and here *every* real node is context (only the
virtual node is query), so there is no train_ratio < 1 to shrink the context
side the way node-level pretraining does.

Each optimizer step averages the loss over --n-grad-accum-steps freshly
sampled graphs, and the LR linearly warms up over --n-warmup-steps optimizer
steps -- both mirror the real pretraining config (pretrain.toml), since raw
single-sample SGD is noisy enough to collapse the model into predicting the
label's marginal mean regardless of input.

Usage (after GPU allocation, run with cwd=paper/):
    python -m bin.graphpfn.graph_level_smoke_test --n-steps 500 --device cuda
"""

import argparse

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from lib.graphpfn.model import GraphPFN
from lib.graphpfn.prior.graphs import sample_graph
from lib.graphpfn.prior.prior_typings import GraphConfig
from lib.util import TaskType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-nodes-min", type=int, default=30)
    parser.add_argument("--n-nodes-max", type=int, default=80)
    parser.add_argument("--avg-degree-min", type=float, default=2.0)
    parser.add_argument("--avg-degree-max", type=float, default=8.0)
    parser.add_argument("--n-groups-min", type=int, default=2)
    parser.add_argument("--n-groups-max", type=int, default=4)
    parser.add_argument("--n-feature-dim", type=int, default=8)
    parser.add_argument(
        "--n-steps", type=int, default=500, help="Number of optimizer steps."
    )
    parser.add_argument(
        "--n-grad-accum-steps",
        type=int,
        default=16,
        help="Graphs averaged into the loss per optimizer step.",
    )
    parser.add_argument(
        "--n-warmup-steps",
        type=int,
        default=100,
        help="Optimizer steps over which the LR linearly warms up.",
    )
    parser.add_argument(
        "--n-calib-graphs",
        type=int,
        default=500,
        help="Graphs sampled up-front to standardize the label.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sample_graph_config(args: argparse.Namespace) -> GraphConfig:
    n_nodes = int(np.random.randint(args.n_nodes_min, args.n_nodes_max + 1))
    avg_degree = float(np.random.uniform(args.avg_degree_min, args.avg_degree_max))
    n_groups = int(np.random.randint(args.n_groups_min, args.n_groups_max + 1))
    return {
        "n_nodes": n_nodes,
        "avg_degree": avg_degree,
        "sampler": {
            "_type_": "sbm",
            "n_groups": n_groups,
            "offdiagonal_coef": 0.1,
        },
    }


def average_degree(graph: dgl.DGLGraph) -> float:
    """Fixed graph-level labeling function shared by ALL sampled graphs."""
    return graph.in_degrees().float().mean().item()


def calibrate_label_stats(args: argparse.Namespace) -> tuple[float, float]:
    """Estimate mean/std of the fixed label over the graph-generation config."""
    values = [
        average_degree(sample_graph(sample_graph_config(args)))
        for _ in range(args.n_calib_graphs)
    ]
    values = np.array(values)
    return float(values.mean()), float(values.std() + 1e-8)


def build_virtual_node_input(
    graph: dgl.DGLGraph,
    n_feature_dim: int,
    device: torch.device,
) -> tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    """Augments the graph with one virtual node connected to all real nodes.

    The virtual node is appended last and is the sole train_mask=False (query)
    position; all real nodes are train_mask=True context. There is no
    per-node label in this task, so their y_train is i.i.d. noise (set by the
    caller) -- only the virtual node's prediction is supervised. A constant
    y_train (e.g. all zeros) would itself be a consistent context pattern
    ("y = 0 for every x"), and an in-context learner is rational to copy that
    pattern at eval time regardless of x_eval; i.i.d. noise carries no
    learnable x -> y mapping, so it can't be copied as a shortcut.
    """
    n_real = graph.num_nodes()
    src, dst = graph.edges()

    virtual_idx = n_real
    vn_src = torch.arange(n_real, dtype=src.dtype)
    vn_dst = torch.full((n_real,), virtual_idx, dtype=dst.dtype)

    new_src = torch.cat([src, vn_src, vn_dst])
    new_dst = torch.cat([dst, vn_dst, vn_src])
    augmented = dgl.graph((new_src, new_dst), num_nodes=n_real + 1)

    features = torch.randn(n_real + 1, n_feature_dim)
    features[virtual_idx] = 0.0

    train_mask = torch.zeros(n_real + 1, dtype=torch.bool)
    train_mask[:n_real] = True

    return augmented.to(device), features.to(device), train_mask.to(device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("Calibrating label statistics (mean degree) over the graph prior...")
    label_mean, label_std = calibrate_label_stats(args)
    print(f"  label mean={label_mean:.4f}, std={label_std:.4f}")

    model = GraphPFN(edge_head=None, feat_head=False, freeze_tfm=True).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"n_trainable_params={n_trainable:,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / args.n_warmup_steps)
    )

    losses = []
    preds_log = []
    labels_log = []
    for step in range(1, args.n_steps + 1):
        optimizer.zero_grad()
        step_losses = []
        for _ in range(args.n_grad_accum_steps):
            graph = sample_graph(sample_graph_config(args))
            label = (average_degree(graph) - label_mean) / label_std

            aug_graph, features, train_mask = build_virtual_node_input(
                graph, args.n_feature_dim, device
            )
            n_train = int(train_mask.sum().item())
            y_train = torch.randn(n_train, dtype=torch.float32, device=device)

            out = model(
                graph=aug_graph,
                features=features,
                y_train=y_train,
                train_mask=train_mask,
                task_type=TaskType.REGRESSION,
                n_random_features=0,
            )

            virtual_idx = aug_graph.num_nodes() - 1
            pred = out["predictions"][virtual_idx]
            target = torch.tensor(label, dtype=torch.float32, device=device)
            loss = F.mse_loss(pred, target)

            (loss / args.n_grad_accum_steps).backward()
            step_losses.append(loss.item())
            preds_log.append(pred.item())
            labels_log.append(label)

        optimizer.step()
        scheduler.step()

        losses.append(float(np.mean(step_losses)))
        if step % args.log_every == 0:
            recent_loss = float(np.mean(losses[-args.log_every :]))
            recent_n = args.log_every * args.n_grad_accum_steps
            recent_preds = np.array(preds_log[-recent_n:])
            recent_labels = np.array(labels_log[-recent_n:])
            corr = np.corrcoef(recent_preds, recent_labels)[0, 1]
            print(
                f"step {step:5d} | lr={scheduler.get_last_lr()[0]:.2e} "
                f"| loss (last {args.log_every})={recent_loss:.4f} "
                f"| pred mean={recent_preds.mean():+.3f} std={recent_preds.std():.3f} "
                f"| label mean={recent_labels.mean():+.3f} std={recent_labels.std():.3f} "
                f"| corr(pred, label)={corr:+.3f} "
                "| predict-prior-mean baseline MSE=1.0 (label is standardized)"
            )

    print("Done.")


if __name__ == "__main__":
    main()
