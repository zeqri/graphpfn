"""Node-level regression on REAL QM9NMR per-atom isotropic shielding,
CARBON ATOMS ONLY, using the exact same idea/pipeline as
node_level_city_roads_icl.py (real labeled nodes serve as the PFN context
directly, no virtual node), combined with qm9nmr_carbon_only_checkpoint_
eval.py's carbon-only masking convention and qm9nmr_node_level_icl_matched_
city_roads.py's city-roads-M-matched scale.

Carbon-only convention (see qm9nmr_carbon_only_checkpoint_eval.py's
docstring for the full rationale): the graph keeps EVERY real atom and bond
of every sampled molecule -- no pruning -- but only CARBON atoms are ever
given a real label / used as context / scored. Heteroatoms (H, N, O, F) are
structural-only: present in the graph (so the model can still see a
carbon's true bonded environment via the graph adapter's message-passing),
but never fed a label and never appear in train/val/test masks. This avoids
an earlier bug where pruning heteroatoms out entirely deleted exactly the
local chemical environment (e.g. C-OH vs. C-NH2 vs. C-F) that determines a
carbon's NMR shift.

Scale-matching (see qm9nmr_node_level_icl_matched_city_roads.py): molecules
are sampled in random order and greedily assigned whole-molecule-at-a-time
to train, then val, then test, until each split's CARBON atom count reaches
its target -- city-roads-M's own RL split sizes (3610/3611/28886) by
default. A molecule's heteroatoms tag along with it (always structural-only)
regardless of which split its carbons land in.

GEOMETRY: unlike the original version of this script (which used the
RELEASED graphpfn package's predict_icl, matching node_level_city_roads_
icl.py's own import path), this now uses this repo's INTERNAL
lib.graphpfn.model.GraphPFN directly. The released package's model
(graphpfn.model.graphpfn.GraphPFN, installed from the sibling
../graphpfn/src repo) has no edge-feature/distance mechanism at all -- it's
pure topology-based attention, same as this repo's model.py before the
geometric-prior work (see dev/README.md sections 9-13). Real bond distances
are computed here from QM9's mol.pos (3D coordinates) and attached as
graph.edata["distance"], which the internal model's EdgeDistanceEncoder
(model.py) uses as an attention bias -- the whole point of pointing this at
a checkpoint trained with the geometric prior
(multigraph_molecule_geometric_rbf_or_linear by default).

This also fixes what was a latent bug in the released-package version:
that model's GraphPFN.from_pretrained does `torch.load(path)["state_dict"]`
with no fallback, then asserts every checkpoint key already exists in its
own architecture. This repo's own training checkpoints are saved in
bin/graphpfn/pretrain.py's {"model": ..., "model_ema": ...} format (no
top-level "state_dict" key at all) and contain distance_encoder.* keys the
released model doesn't define -- pointing the old code at
DEFAULT_CHECKPOINT_PATH would have raised a KeyError before even reaching
inference. load_checkpoint_icl below handles this repo's own format
directly (same convention as qm9nmr_carbon_only_streaming_full_finetune.py's
load_checkpoint_full_finetune).

ICL only, no finetuning: no gradient step happens anywhere in this script
(everything runs under torch.inference_mode()) -- the model's weights are
exactly whatever the checkpoint specifies, matching node_level_city_roads_
icl.py's "real labeled nodes serve as context directly" ICL evaluation.
Ensembling (--n-members, default 10, matching the paper's own ICL
convention) is a plain loop of independent forward passes -- n_random_
features > 0 already draws fresh random features from the global torch RNG
on every call, which is the model's own source of ensemble diversity (no
batched-ensemble mechanism exists in the internal model, unlike the
released package's GraphPFN wrapper, so this is N times the compute of a
single forward pass, not a free lunch).

CAUTION -- graph size: this evaluates the WHOLE matched dataset (train +
val + test carbons and all structural heteroatoms) in a single forward
pass, no streaming/micro-batching (same design as the original released-
package version). At city-roads-M's default scale (3610/3611/28886 carbon
atoms), total atom count likely lands well past model.py's
MAX_SDPA_GRAPH_SIZE=10_000, forcing the sparse DGL attention path, and
could be memory-heavy for a single non-batched pass -- this risk already
existed in the original script (same total node count, same no-batching
design), it isn't new here. Reduce --n-train/val/test-carbons for a smoke
test before running at full scale.

Usage (run with cwd=paper/, inside a GPU allocation):
    python -m bin.graphpfn.qm9nmr_node_level_icl_carbon_only_matched_city_roads --device cuda
"""

from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path

import numpy as np

DEFAULT_QM9_ROOT = "/p/project1/profound/al-zeqri1/PFN/graphpfn/data/QM9"
DEFAULT_NMR_PATH = "/p/project1/profound/al-zeqri1/PFN/graphpfn/data/SI_DFT_NMR.txt"
DEFAULT_NMR_CACHE_PATH = (
    "/p/project1/profound/al-zeqri1/PFN/graphpfn/data/qm9nmr_shieldings_cache.npz"
)
# This repo's own training checkpoint (backbone + adapters + edge head,
# {"model_ema": ...}), NOT the released package's adapters-only checkpoint
# -- see load_checkpoint_icl. Pick whichever geometric-prior run you want to
# evaluate; defaults to the most exposed/cleaned variant discussed in
# dev/README.md sections 12-13.
DEFAULT_CHECKPOINT_PATH = Path(
    "/p/project1/profound/al-zeqri1/PFN/graphpfn_EGNN/graphpfn/paper/exp/graphpfn/"
    "pretrain/multigraph_molecule_geometric_rbf_or_linear/pretrain/checkpoint.pt"
)
DEFAULT_TOML_PATH = Path(
    "/p/project1/profound/al-zeqri1/PFN/graphpfn_EGNN/graphpfn/paper/exp/graphpfn/"
    "pretrain/multigraph_molecule_geometric_rbf_or_linear/pretrain.toml"
)

NMR_PHASES = ["gas", "ccl4", "thf", "acetone", "methanol", "dmso"]

# city-roads-M's own RL-split labeled node counts (see
# node_level_city_roads_icl.py's printed "split RL: 3610 train / 3611 val /
# 28886 test labeled nodes") -- the targets this script's molecule-carbon
# accumulation tries to match as closely as whole-molecule granularity allows.
CITY_ROADS_M_TRAIN = 3610
CITY_ROADS_M_VAL = 3611
CITY_ROADS_M_TEST = 28886


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qm9-root", default=DEFAULT_QM9_ROOT)
    parser.add_argument("--nmr-path", default=DEFAULT_NMR_PATH)
    parser.add_argument("--nmr-cache-path", default=DEFAULT_NMR_CACHE_PATH)
    parser.add_argument("--nmr-phase", default="gas", choices=NMR_PHASES,
                        help="Same per-atom regression target used by "
                             "qm9nmr_carbon_only_checkpoint_eval.py.")
    parser.add_argument("--n-train-carbons", type=int, default=CITY_ROADS_M_TRAIN,
                        help="Target labeled train CARBON atoms -- matches "
                             "city-roads-M's RL split by default.")
    parser.add_argument("--n-val-carbons", type=int, default=CITY_ROADS_M_VAL,
                        help="Target labeled val carbon atoms.")
    parser.add_argument("--n-test-carbons", type=int, default=CITY_ROADS_M_TEST,
                        help="Target labeled test carbon atoms.")
    parser.add_argument("--n-members", type=int, default=10,
                        help="Ensemble members (paper ICL setting uses 10). "
                             "Each is an independent full forward pass.")
    parser.add_argument("--n-random-features", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH,
                        help="Local path to this repo's own bin/graphpfn/"
                             "pretrain.py training checkpoint (backbone + "
                             "adapters + edge head, ~334 tensors under "
                             "model_ema) -- NOT the released package's "
                             "adapters-only checkpoint format.")
    parser.add_argument("--toml", type=Path, default=DEFAULT_TOML_PATH,
                        help="pretrain.toml the checkpoint was produced "
                             "under -- only base_config['model'] (e.g. "
                             "edge_head) is actually read from it.")
    return parser.parse_args()


def load_qm9nmr_shieldings(nmr_path: str, cache_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (shieldings [total_atoms, 6] float32, offsets [n_mols + 1] int64)."""
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["shieldings"], cached["offsets"]

    print(f"Parsing {nmr_path} (one-time; caching to {cache_path}) ...")
    shieldings_list = []
    offsets = [0]
    with open(nmr_path) as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        header = lines[i].strip()
        if header == "":
            i += 1
            continue
        natoms = int(header)
        data_lines = lines[i + 2: i + 2 + natoms]
        for dl in data_lines:
            parts = dl.split()
            shieldings_list.append([float(x) for x in parts[1:1 + len(NMR_PHASES)]])
        offsets.append(offsets[-1] + natoms)
        i += 2 + natoms

    shieldings = np.array(shieldings_list, dtype=np.float32)
    offsets = np.array(offsets, dtype=np.int64)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(cache_path, shieldings=shieldings, offsets=offsets)
    print(f"Parsed {len(offsets) - 1} molecules, {shieldings.shape[0]} atoms total; "
          f"cached to {cache_path}.")
    return shieldings, offsets


def build_matched_carbon_only_dataset(args, phase_idx: int):
    """Greedily assigns whole QM9 molecules (in random order) to train, then
    val, then test, counting only each molecule's CARBON atoms against the
    split's target -- so the resulting graph's train/val/test labeled
    CARBON counts closely match city-roads-M's RL split. Every atom of
    every included molecule (carbon and heteroatoms alike) stays in the
    graph with its real bonds; heteroatoms are simply never marked in any
    of the three masks (structural-only, matching qm9nmr_carbon_only_
    checkpoint_eval.py's convention).

    Returns (graph, features, targets_raw, masks): `graph` is a plain
    dgl.DGLGraph with real bond distances (from QM9's mol.pos) attached as
    graph.edata["distance"]; `features` is a torch.Tensor [n_nodes,
    n_features]; `targets_raw` is real ppm shielding values (numpy, one per
    atom, meaningless for heteroatoms); `masks` is {"train"/"val"/"test":
    boolean numpy array over nodes}.
    """
    import dgl
    import torch
    from torch_geometric.datasets import QM9

    full = QM9(root=args.qm9_root)
    print(f"Loaded QM9: {len(full)} molecules; per-atom target = NMR shielding "
          f"[{NMR_PHASES[phase_idx]}], CARBON ATOMS ONLY")

    nmr_shieldings, nmr_offsets = load_qm9nmr_shieldings(args.nmr_path, args.nmr_cache_path)
    assert len(nmr_offsets) - 1 == len(full), (
        f"QM9NMR has {len(nmr_offsets) - 1} molecules, PyG QM9 has {len(full)} -- "
        "positional alignment assumption broken."
    )

    targets = [args.n_train_carbons, args.n_val_carbons, args.n_test_carbons]

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(full))

    atom_features = []
    atom_targets = []
    atom_is_carbon = []
    atom_split = []  # 0=train, 1=val, 2=test, per atom (heteroatoms get discarded below)
    edge_src_parts = []
    edge_dst_parts = []
    edge_distance_parts = []

    offset = 0
    split_idx = 0
    split_carbon_count = 0
    for j in perm:
        if split_idx >= len(targets):
            break  # all splits filled

        j = int(j)
        mol = full[j]
        n_atoms = int(mol.x.shape[0])

        nmr_n_atoms = int(nmr_offsets[j + 1] - nmr_offsets[j])
        assert nmr_n_atoms == n_atoms, (
            f"atom-count mismatch at QM9 idx {j}: PyG has {n_atoms} atoms, "
            f"QM9NMR block has {nmr_n_atoms}."
        )

        # mol.z: atomic numbers -- NOT mol.x[:, 0] (see qm9nmr_carbon_only_
        # checkpoint_eval.py's "FIXED VERSION" note).
        is_carbon = (mol.z.numpy() == 6)
        n_carbon = int(is_carbon.sum())

        atom_features.append(mol.x.numpy().astype(np.float32))
        atom_targets.append(
            nmr_shieldings[nmr_offsets[j]: nmr_offsets[j + 1], phase_idx].copy()
        )
        atom_is_carbon.append(is_carbon)
        # Every atom of this molecule is tagged with the CURRENT split, but
        # only carbon positions will actually be turned into a mask below --
        # heteroatoms get discarded regardless of this molecule's split.
        atom_split.append(np.full(n_atoms, split_idx, dtype=np.int64))

        ei = mol.edge_index.numpy()
        edge_src_parts.append(ei[0] + offset)
        edge_dst_parts.append(ei[1] + offset)
        # Real bond distance from QM9's 3D coordinates -- Euclidean distance
        # between each bonded pair's atomic positions, not a sampled prior.
        pos = mol.pos.numpy()
        edge_distance_parts.append(
            np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1).astype(np.float32)
        )

        offset += n_atoms
        split_carbon_count += n_carbon
        if split_carbon_count >= targets[split_idx]:
            split_idx += 1
            split_carbon_count = 0

    n_nodes = offset
    features = torch.from_numpy(np.concatenate(atom_features, axis=0))
    targets_raw = np.concatenate(atom_targets, axis=0).astype(np.float32)
    is_carbon_all = np.concatenate(atom_is_carbon, axis=0)
    split_of_atom = np.concatenate(atom_split, axis=0)
    edge_src = np.concatenate(edge_src_parts)
    edge_dst = np.concatenate(edge_dst_parts)
    edge_distance = torch.from_numpy(np.concatenate(edge_distance_parts).astype(np.float32))

    graph = dgl.graph(data=(edge_src, edge_dst), num_nodes=n_nodes, idtype=torch.int32)
    graph.edata["distance"] = edge_distance

    # Heteroatoms never enter any mask, regardless of which split their
    # molecule landed in -- structural-only, per qm9nmr_carbon_only_
    # checkpoint_eval.py's convention.
    masks = {
        "train": (split_of_atom == 0) & is_carbon_all,
        "val": (split_of_atom == 1) & is_carbon_all,
        "test": (split_of_atom == 2) & is_carbon_all,
    }
    n_heteroatoms = n_nodes - int(is_carbon_all.sum())

    print(f"Split (carbon-only, molecule-accumulated, targets "
          f"{args.n_train_carbons}/{args.n_val_carbons}/{args.n_test_carbons} "
          f"carbon atoms): {masks['train'].sum()} train / {masks['val'].sum()} val / "
          f"{masks['test'].sum()} test labeled CARBON atoms "
          f"({n_nodes} atoms total incl. {n_heteroatoms} structural-only heteroatoms, "
          f"{graph.num_edges()} directed edges, "
          f"distance range=[{edge_distance.min():.3f}, {edge_distance.max():.3f}] A).")

    return graph, features, targets_raw, masks


def load_checkpoint_icl(
    config: dict, checkpoint_path: Path, device, verbose: bool = True
):
    """Constructs this repo's internal GraphPFN and loads --checkpoint onto
    it non-strict. No finetuning happens anywhere in this script (pure ICL,
    everything under torch.inference_mode()), so freeze_tfm's value has no
    actual effect here -- left at the model's own default.

    Handles either a full bin/graphpfn/pretrain.py training checkpoint
    (backbone + adapters + edge head, {"model_ema": ...}) or the released
    adapters-only checkpoint ({"state_dict": ...}), same convention as
    qm9nmr_carbon_only_streaming_full_finetune.py's
    load_checkpoint_full_finetune. Any keys the checkpoint doesn't cover
    keep their construction-time values (e.g. LimiX-16M.ckpt for the
    backbone, if an adapters-only checkpoint is passed instead)."""
    import torch

    from lib.graphpfn.model import GraphPFN

    model_kwargs = dict(config.get("model", dict()))
    graphpfn = GraphPFN(**model_kwargs).to(device)

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_ema" in raw:
        module_state = raw["model_ema"]
        state_dict = {
            k[len("module."):]: v for k, v in module_state.items() if k.startswith("module.")
        }
    elif "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        raise ValueError(
            f"Unrecognized checkpoint format: expected a 'model_ema' or "
            f"'state_dict' key, got top-level keys={list(raw.keys())}"
        )

    missing, unexpected = graphpfn.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected keys when loading checkpoint: {unexpected}"
    if verbose:
        print(f"Loaded {len(state_dict)} tensors from {checkpoint_path} "
              f"({len(missing)} keys left at construction-time values).")
    return graphpfn.eval()


def main() -> None:
    args = parse_args()

    import torch

    from lib.metrics import calculate_metrics
    from lib.util import TaskType

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.toml, "rb") as f:
        toml_config = tomllib.load(f)
    config = toml_config["base_config"]

    device = torch.device(args.device)
    amp_enabled = (
        config.get("amp", False) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    print(f"device={device} amp_enabled={amp_enabled}")

    phase_idx = NMR_PHASES.index(args.nmr_phase)
    graph, features, targets_raw, masks = build_matched_carbon_only_dataset(args, phase_idx)
    graph = graph.to(device)
    features = features.to(device)

    train_mask_np = masks["train"]
    label_mean = float(targets_raw[train_mask_np].mean())
    label_std = float(targets_raw[train_mask_np].std() + 1e-8)
    print(f"Label stats (train carbons only): mean={label_mean:.4f}, std={label_std:.4f}")

    graphpfn = load_checkpoint_icl(config, args.checkpoint, device)

    train_mask = torch.from_numpy(train_mask_np).to(device)
    y_train = torch.from_numpy(
        (targets_raw[train_mask_np] - label_mean) / label_std
    ).to(dtype=torch.float32, device=device)

    # Ensembling: n_random_features > 0 draws fresh random features from the
    # global torch RNG every call, so N independent forward passes already
    # gives N different ensemble members -- no batched-ensemble mechanism
    # exists on this internal model (unlike the released package's
    # GraphPFN wrapper), so this really is N times the compute of one pass.
    all_preds = []
    with torch.inference_mode():
        for member in range(args.n_members):
            with torch.autocast(
                device.type, enabled=amp_enabled, dtype=torch.bfloat16 if amp_enabled else None
            ):
                out = graphpfn(
                    graph=graph, features=features, y_train=y_train,
                    train_mask=train_mask, task_type=TaskType.REGRESSION,
                    n_random_features=args.n_random_features,
                )
            all_preds.append(out["predictions"].float().cpu().numpy())
            print(f"  ensemble member {member + 1}/{args.n_members} done")

    pred_std = np.mean(all_preds, axis=0)
    preds = label_mean + label_std * pred_std

    test_mask = masks["test"]
    metrics = calculate_metrics(
        targets_raw[test_mask], preds[test_mask], "regression", "labels"
    )

    baseline_mae = float(np.abs(targets_raw[test_mask] - label_mean).mean())

    print(f"\n=== GraphPFN ICL (internal model + real bond distances) on "
          f"QM9NMR carbon-only [{args.nmr_phase}], matched to city-roads-M scale ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"ensemble members: {args.n_members}, seed: {args.seed}")
    print(f"R2:  {metrics['r2']:.4f}")
    print(f"MAE: {metrics['mae']:.4f} ppm")
    print(f"MAE of constant (train-mean) baseline: {baseline_mae:.4f} ppm")


if __name__ == "__main__":
    main()
