"""Real-QM9-data version of pool_icl_real_limix.py: same architecture (real
pretrained LimiX-16M backbone, pooling atoms into one token per molecule
right after encoder_x, graph-adapter repurposed as full attention among
pooled molecule tokens), but the "prior" is replaced entirely by random
subsets of real QM9 molecules, regressing on one of QM9's 19 standard
targets instead of a synthetic-SCM-generated label.

QM9 (via torch_geometric) already gives us, per molecule, exactly what the
synthetic prior had to manufacture: real per-atom features (data.x, PyG's
own 11-dim one-hot+extra featurization), real bonds (data.edge_index), real
3D coordinates (data.pos, so real bond distances instead of a QM9-fit
Gaussian prior), and one real graph-level scalar label per molecule
(data.y) -- no virtual-node/SCM machinery needed at all, since the label
already exists.

"Easy" target default: cv (heat capacity, QM9 target index 11) -- widely
noted in the QM9 benchmarking literature as one of the properties best
explained by simple molecular composition/size, as opposed to properties
like homo/lumo/gap that need real electronic-structure reasoning. Override
with --target.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import dgl

REPO = Path("/p/project1/profound/al-zeqri1/PFN/second/graphpfn/paper")
sys.path.insert(0, str(REPO))

import lib.tfm.limix as limix_mod  # noqa: E402
_LOCAL_CKPT = str(REPO / "checkpoints/LimiX-16M.ckpt")
limix_mod._download_limix_checkpoint = lambda: _LOCAL_CKPT

from lib.graphpfn.model import GraphPFN, SDPAInput  # noqa: E402
from lib.util import TaskType  # noqa: E402

from sklearn.metrics import r2_score  # noqa: E402
from torch_geometric.datasets import QM9  # noqa: E402

QM9_ROOT = "/p/project1/profound/al-zeqri1/PFN/graphpfn/data/QM9"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard QM9 target order (as provided by torch_geometric.datasets.QM9).
QM9_TARGETS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298",
    "g298", "cv", "u0_atom", "u298_atom", "h298_atom", "g298_atom", "A", "B", "C",
]


def load_qm9():
    return QM9(root=QM9_ROOT)


def split_molecule_pools(n_molecules: int, eval_fraction: float, seed: int):
    """Partition QM9 into disjoint train/eval molecule pools up front, so
    training and evaluation draws can NEVER overlap at the molecule level --
    unlike the synthetic prior (an infinite generator, where every sampled
    dataset is fresh by construction), QM9 is a fixed 130,831-molecule set,
    so without this split, adapter training could simply memorize specific
    molecules that later reappear as eval queries, inflating R^2 without
    reflecting genuine in-context generalization.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_molecules)
    n_eval = int(n_molecules * eval_fraction)
    return perm[n_eval:], perm[:n_eval]  # train_pool, eval_pool


def sample_one_dataset(dataset, pool: np.ndarray, n_graphs_range: tuple[int, int], target_idx: int, rng: np.random.Generator):
    """Draw a random subset of real QM9 molecules as one 'dataset', mirroring
    the synthetic prior's multi-graph sampling: N independent molecules,
    each contributing its own atom features + one graph-level label.
    `pool` restricts sampling to a fixed subset of molecule indices (the
    train or eval pool from split_molecule_pools), so this dataset can never
    draw a molecule from the other pool.
    """
    n_graphs = int(rng.integers(n_graphs_range[0], n_graphs_range[1] + 1))
    mol_indices = rng.choice(pool, size=n_graphs, replace=False)
    atom_features_per_molecule = []
    distances_per_molecule = []
    edges_per_molecule = []
    labels = []
    for idx in mol_indices:
        d = dataset[int(idx)]
        atom_features_per_molecule.append(d.x.numpy())
        pos = d.pos.numpy()
        src, dst = d.edge_index.numpy()
        dist = np.linalg.norm(pos[src] - pos[dst], axis=-1)
        distances_per_molecule.append(dist)
        edges_per_molecule.append((src, dst))
        labels.append(float(d.y[0, target_idx]))
    return atom_features_per_molecule, distances_per_molecule, edges_per_molecule, np.array(labels)


def context_query_split(n_graphs: int, train_ratio: float, rng: np.random.Generator):
    n_train = max(1, min(n_graphs - 1, round(n_graphs * train_ratio)))
    perm = rng.permutation(n_graphs)
    return perm[:n_train], perm[n_train:]


def standardize_by_context(y_all: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = y_all[train_idx].mean()
    std = y_all[train_idx].std()
    std = std if std > 1e-6 else 1.0
    return (y_all - mean) / std


def build_model(random_adapter_init: bool = True) -> GraphPFN:
    """Same as pool_icl_real_limix.py's build_model -- see that file's NOTE
    for why random_adapter_init=True is needed for a meaningful one-shot
    (untrained) check: the adapter's own output projection is zero-init by
    default, which would make it a pure no-op otherwise.
    """
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


def pooled_forward(model: GraphPFN, atom_features_per_molecule, y_context, train_idx, test_idx):
    """Identical in structure to pool_icl_real_limix.py's pooled_forward:
    manually replicates FeaturesTransformer.forward, pooling atoms into one
    token per molecule right after encoder_x, and repurposing the
    graph-adapter's message-passing slot as full attention among the pooled
    molecule tokens (their real-QM9-bond distances never reach this slot --
    they were only ever used, if at all, inside per-atom processing, which
    this variant doesn't use since encoder_x has no cross-atom mixing).
    """
    tfm = model.tfm.module
    n_graphs = len(atom_features_per_molecule)
    is_context = torch.zeros(n_graphs, dtype=torch.bool, device=DEVICE)
    is_context[train_idx] = True

    perm = torch.argsort((~is_context).float(), stable=True)
    ordered_atom_feats = [atom_features_per_molecule[i] for i in perm.tolist()]
    n_context = int(is_context.sum().item())

    pooled_tokens = []
    for atoms in ordered_atom_feats:
        x = torch.tensor(atoms, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        batch_size, seq_len, num_feature = x.shape
        feature_to_add = num_feature % tfm.features_per_group
        if feature_to_add > 0:
            x = torch.cat([x, torch.zeros(batch_size, seq_len, feature_to_add, device=x.device, dtype=x.dtype)], dim=-1)
        x_dict = {"data": x, "mask": torch.isnan(x).to(torch.int32)}
        x_dict = {k: v.reshape(batch_size, seq_len, v.shape[2] // tfm.features_per_group, tfm.features_per_group) for k, v in x_dict.items()}
        x_dict["eval_pos"] = seq_len
        preprocessed = tfm.x_preprocess(x_dict)
        preprocessed = tfm.process_4_x(preprocessed)
        x_encoder_result = tfm.encoder_x(preprocessed)
        x_emb = x_encoder_result["data"]
        pooled_tokens.append(x_emb.mean(dim=1))

    embedded_x = torch.cat(pooled_tokens, dim=0).unsqueeze(0)
    embedded_x = tfm.add_embeddings(embedded_x)

    y_full = torch.full((1, n_graphs), float("nan"), device=DEVICE, dtype=torch.float32)
    y_context_ordered = torch.tensor(y_context, dtype=torch.float32, device=DEVICE)
    y_full[0, :n_context] = y_context_ordered
    y_dict = {"data": y_full.unsqueeze(-1)}
    y_type = torch.ones_like(y_dict["data"])
    embedded_y = tfm.mixed_y_embedding(y_dict, y_type=y_type, eval_pos=n_context)

    embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)

    attn_mask = torch.ones(n_graphs, n_graphs, dtype=torch.bool, device=DEVICE)
    zero_degree_mask = torch.zeros(n_graphs, dtype=torch.bool, device=DEVICE)
    model._graph_holder.graph = SDPAInput(attn_mask=attn_mask, zero_degree_mask=zero_degree_mask, edge_distance=None)

    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
    ):
        encoder_out = tfm.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=n_context)[0]
    model._graph_holder.graph = None

    encoder_out = tfm.encoder_out_norm(encoder_out)
    test_encoder_out = encoder_out[:, n_context:, -1]
    test_y_type = y_type[:, n_context:, 0]
    _, reg_pred = tfm.y_decoder(test_encoder_out, test_y_type)
    pred_ordered = reg_pred.float().squeeze(0).squeeze(-1)

    query_order_orig_idx = perm[n_context:]
    pred_by_orig_idx = torch.zeros(n_graphs, device=DEVICE)
    pred_by_orig_idx[query_order_orig_idx] = pred_ordered
    return pred_by_orig_idx[test_idx]


def train_adapters(model: GraphPFN, dataset, train_pool, n_graphs_range, target_idx, n_steps: int, rng: np.random.Generator,
                    lr: float = 1e-3, max_norm: float = 1.0, train_ratio: float = 0.3):
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"training {sum(p.numel() for p in trainable)}/{sum(p.numel() for p in model.parameters())} params (adapters only, backbone frozen)")
    print(f"sampling training datasets from a fixed pool of {len(train_pool)} molecules")
    optimizer = torch.optim.Adam(trainable, lr=lr)
    model.train()

    losses = []
    for step in range(1, n_steps + 1):
        atom_feats, _dist, _edges, y_all = sample_one_dataset(dataset, train_pool, n_graphs_range, target_idx, rng)
        n_graphs = len(atom_feats)
        train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
        if len(test_idx) == 0 or len(train_idx) < 2:
            continue
        y_std = standardize_by_context(y_all, train_idx)

        optimizer.zero_grad()
        pred = pooled_forward(model, atom_feats, y_std[train_idx], train_idx, test_idx)
        loss = torch.nn.functional.mse_loss(
            pred, torch.tensor(y_std[test_idx], dtype=torch.float32, device=DEVICE)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_norm)
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"[train step {step}/{n_steps}] recent loss = {np.mean(losses[-10:]):.4f}")

    model.eval()
    return losses


def main(
    n_eval_datasets: int = 60,
    seed: int = 0,
    random_adapter_init: bool = True,
    n_train_steps: int = 0,
    target: str = "cv",
    n_graphs_min: int = 10,
    n_graphs_max: int = 30,
    train_ratio: float = 0.3,
    eval_pool_fraction: float = 0.2,
):
    print(f"device = {DEVICE}")
    target_idx = QM9_TARGETS.index(target)
    print(f"regressing on QM9 target '{target}' (index {target_idx})")

    dataset = load_qm9()
    print(f"loaded QM9: {len(dataset)} molecules")
    train_pool, eval_pool = split_molecule_pools(len(dataset), eval_pool_fraction, seed=seed)
    print(f"molecule-level split: {len(train_pool)} in train pool, {len(eval_pool)} in eval pool (disjoint, no overlap)")

    model = build_model(random_adapter_init=random_adapter_init and n_train_steps == 0)
    rng = np.random.default_rng(seed)

    if n_train_steps > 0:
        train_adapters(model, dataset, train_pool, (n_graphs_min, n_graphs_max), target_idx, n_train_steps, rng, train_ratio=train_ratio)

    scores = []
    with torch.no_grad():
        for i in range(1, n_eval_datasets + 1):
            atom_feats, _dist, _edges, y_all = sample_one_dataset(dataset, eval_pool, (n_graphs_min, n_graphs_max), target_idx, rng)
            n_graphs = len(atom_feats)
            train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
            if len(test_idx) == 0 or len(train_idx) < 2 or np.std(y_all[test_idx]) == 0:
                continue
            y_std = standardize_by_context(y_all, train_idx)
            pred = pooled_forward(model, atom_feats, y_std[train_idx], train_idx, test_idx)
            scores.append(r2_score(y_std[test_idx], pred.cpu().numpy()))
            if i % 10 == 0:
                print(f"[{i}/{n_eval_datasets}] evaluated")

    arr = np.array(scores)
    print(f"\n=== QM9 '{target}' Pool+ICL R^2 (n={len(arr)}) ===")
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
    parser.add_argument("--target", type=str, default="cv", choices=QM9_TARGETS)
    parser.add_argument("--n_graphs_min", type=int, default=10)
    parser.add_argument("--n_graphs_max", type=int, default=30)
    parser.add_argument("--train_ratio", type=float, default=0.3)
    parser.add_argument("--eval_pool_fraction", type=float, default=0.2)
    args = parser.parse_args()
    main(
        n_eval_datasets=args.n_eval_datasets,
        seed=args.seed,
        random_adapter_init=not args.zero_init_adapter,
        n_train_steps=args.n_train_steps,
        target=args.target,
        n_graphs_min=args.n_graphs_min,
        n_graphs_max=args.n_graphs_max,
        train_ratio=args.train_ratio,
        eval_pool_fraction=args.eval_pool_fraction,
    )
