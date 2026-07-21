"""Node-level regression on REAL QM9NMR per-atom isotropic shielding,
CARBON ATOMS ONLY, using the same pipeline as qm9nmr_node_level_icl_carbon_
only_matched_city_roads.py, but matched to the QM9-NMR paper's OWN
train/test protocol instead of city-roads-M's split sizes -- so the printed
MAE is directly comparable to that paper's reported numbers, rather than to
an unrelated RL benchmark's scale.

Paper being matched: Gupta, Chakraborty & Ramakrishnan, "Revving up 13C NMR
shielding predictions across chemical space" (Mach. Learn.: Sci. Technol. 2
035010, 2021). Section 4.2 / table 1: FCHL-based KRR trained on a random set
of 100k of the dataset's 812k labeled 13C atoms, evaluated on a disjoint
50k-atom hold-out set, gas phase, MAE = 1.88 ppm (direct ML) / 1.36 ppm
(Delta-ML). Those two numbers are this script's target to beat/match, and
are printed alongside this run's own MAE at the end.

WHY THIS ISN'T A LITERAL 100k/50k REPRODUCTION (proxy, not exact match):
  1. Split unit. The paper's KRR treats every atom as an independent
     training/test entity (a local descriptor with a distance cutoff), so
     sibling carbons of the same molecule can freely land on opposite sides
     of the split. This repo's GraphPFN is graph-ICL: context ("train") and
     query ("test") nodes sit in the SAME message-passing graph, so a query
     carbon could see a same-molecule context carbon's label leaking
     through the bond graph. To keep the comparison honest for a graph
     model (no such leakage), this script assigns WHOLE molecules to a
     single split, same convention as the city-roads-matched script --
     carbon counts land close to the targets but not exact.
  2. Scale. The paper's 100k train + 50k test carbons implies ~24k
     molecules and roughly 300k-450k total atoms (carbons + structural
     H/N/O/F). This script (like the city-roads-matched one) runs the WHOLE
     graph through a single non-streaming forward pass per ensemble member
     -- no micro-batching exists here -- so 100k/50k is very likely
     infeasible on one GPU. Defaults below instead target a smaller but
     still substantial carbon count (20k train / 10k test, preserving the
     paper's 2:1 train:test ratio) as a feasible PROXY for the full-scale
     number; override --n-train-carbons/--n-test-carbons upward if you have
     the memory for it (the printed total-atom count tells you what you're
     about to forward-pass).
  3. No validation split needed. ICL only -- no gradient step, no
     early stopping -- so unlike the city-roads-matched script this
     defaults --n-val-carbons to 0 (paper has no val split either).

GEOMETRY, checkpoint format, ensembling, and the carbon-only/structural-
heteroatom masking convention are unchanged from qm9nmr_node_level_icl_
carbon_only_matched_city_roads.py -- see that script's docstring for the
full rationale (real bond distances from QM9's mol.pos feed the internal
model's EdgeDistanceEncoder; checkpoints are this repo's own bin/graphpfn/
pretrain.py {"model_ema": ...} format; n_random_features>0 is the model's
own source of ensemble diversity across --n-members independent forward
passes).

Usage (run with cwd=paper/, inside a GPU allocation):
    python -m bin.graphpfn.qm9nmr_node_level_icl_carbon_only_matched_qm9nmr_paper --device cuda
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

# Gupta, Chakraborty & Ramakrishnan 2021 (MLST 2 035010), table 1 + section
# 4.2: FCHL-based KRR / Delta-ML, 100k train / 50k hold-out 13C atoms,
# random atom-level split, mean absolute error in ppm. Gas phase is this
# script's default --nmr-phase; the other phases' paper numbers are the
# remaining rows of table 1.
PAPER_MAE_BY_PHASE = {
    "gas": {"ml": 1.88, "delta_ml": 1.36},
    "ccl4": {"ml": 1.91, "delta_ml": 1.38},
    "thf": {"ml": 1.99, "delta_ml": 1.48},
    "acetone": {"ml": 1.93, "delta_ml": 1.42},
    "methanol": {"ml": 1.94, "delta_ml": 1.42},
    "dmso": {"ml": 1.99, "delta_ml": 1.49},
}
PAPER_N_TRAIN_ATOMS = 100_000
PAPER_N_TEST_ATOMS = 50_000

# Feasible proxy for the paper's 100k/50k atom-level split -- see the
# module docstring's "Scale" note. Keeps the paper's 2:1 train:test ratio.
DEFAULT_N_TRAIN_CARBONS = 20_000
DEFAULT_N_VAL_CARBONS = 0
DEFAULT_N_TEST_CARBONS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qm9-root", default=DEFAULT_QM9_ROOT)
    parser.add_argument("--nmr-path", default=DEFAULT_NMR_PATH)
    parser.add_argument("--nmr-cache-path", default=DEFAULT_NMR_CACHE_PATH)
    parser.add_argument("--nmr-phase", default="gas", choices=NMR_PHASES,
                        help="Same per-atom regression target used by the "
                             "city-roads-matched script; also selects which "
                             "row of the paper's table 1 to compare against.")
    parser.add_argument("--n-train-carbons", type=int, default=DEFAULT_N_TRAIN_CARBONS,
                        help="Target labeled train (context) CARBON atoms. "
                             f"Paper uses {PAPER_N_TRAIN_ATOMS:,}; default here "
                             "is a feasible proxy -- raise it if you have the "
                             "memory for a single non-streamed forward pass "
                             "over the resulting graph.")
    parser.add_argument("--n-val-carbons", type=int, default=DEFAULT_N_VAL_CARBONS,
                        help="Target labeled val carbon atoms (unused by "
                             "metrics; ICL has no early stopping). Paper has "
                             "no val split either -- default 0.")
    parser.add_argument("--n-test-carbons", type=int, default=DEFAULT_N_TEST_CARBONS,
                        help="Target labeled test (query) carbon atoms. "
                             f"Paper uses {PAPER_N_TEST_ATOMS:,}; default here "
                             "is a feasible proxy.")
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
    CARBON counts closely match the paper's own 100k/0/50k atom-level split
    sizes (or whatever proxy sizes were passed in), while keeping every
    molecule's carbons together on one side of the split (see module
    docstring point 1 -- avoids context/query leakage through the bond
    graph, which the paper's per-atom KRR split didn't need to worry about).
    A zero-target split (e.g. val by default) is skipped entirely -- no
    molecule is wasted on it.

    Every atom of every included molecule (carbon and heteroatoms alike)
    stays in the graph with its real bonds; heteroatoms are simply never
    marked in any of the three masks (structural-only, matching qm9nmr_
    node_level_icl_carbon_only_matched_city_roads.py's convention).

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

    def _skip_zero_targets(idx: int) -> int:
        while idx < len(targets) and targets[idx] == 0:
            idx += 1
        return idx

    offset = 0
    split_idx = _skip_zero_targets(0)
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

        # mol.z: atomic numbers -- NOT mol.x[:, 0].
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
            split_idx = _skip_zero_targets(split_idx + 1)
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
    # molecule landed in -- structural-only, per qm9nmr_node_level_icl_
    # carbon_only_matched_city_roads.py's convention.
    masks = {
        "train": (split_of_atom == 0) & is_carbon_all,
        "val": (split_of_atom == 1) & is_carbon_all,
        "test": (split_of_atom == 2) & is_carbon_all,
    }
    n_heteroatoms = n_nodes - int(is_carbon_all.sum())

    print(f"Split (carbon-only, molecule-accumulated, targets "
          f"{args.n_train_carbons}/{args.n_val_carbons}/{args.n_test_carbons} "
          f"carbon atoms -- paper uses {PAPER_N_TRAIN_ATOMS:,}/0/{PAPER_N_TEST_ATOMS:,} "
          f"atom-level): {masks['train'].sum()} train / {masks['val'].sum()} val / "
          f"{masks['test'].sum()} test labeled CARBON atoms "
          f"({n_nodes} atoms total incl. {n_heteroatoms} structural-only heteroatoms, "
          f"{graph.num_edges()} directed edges, "
          f"distance range=[{edge_distance.min():.3f}, {edge_distance.max():.3f}] A).")
    if n_nodes > 150_000:
        print(f"WARNING: {n_nodes} total atoms in a single non-streamed forward "
              f"pass -- this may be slow or run out of memory. Lower "
              f"--n-train-carbons/--n-test-carbons if this fails.")

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
    adapters-only checkpoint ({"state_dict": ...})."""
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
    # exists on this internal model, so this really is N times the compute
    # of one pass.
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
    paper_mae = PAPER_MAE_BY_PHASE[args.nmr_phase]

    print(f"\n=== GraphPFN ICL (internal model + real bond distances) on "
          f"QM9NMR carbon-only [{args.nmr_phase}], matched to the QM9-NMR "
          f"paper's own train/test protocol ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"ensemble members: {args.n_members}, seed: {args.seed}")
    print(f"train/test carbons this run: {int(train_mask_np.sum())}/{int(test_mask.sum())} "
          f"(paper: {PAPER_N_TRAIN_ATOMS:,}/{PAPER_N_TEST_ATOMS:,} -- see module "
          f"docstring's 'Scale' note for why this run is smaller)")
    print(f"R2:  {metrics['r2']:.4f}")
    print(f"MAE: {metrics['mae']:.4f} ppm")
    print(f"MAE of constant (train-mean) baseline: {baseline_mae:.4f} ppm")
    print(f"--- paper's FCHL-KRR benchmark at 100k/50k, [{args.nmr_phase}] "
          f"(Gupta, Chakraborty & Ramakrishnan 2021, table 1) ---")
    print(f"paper direct ML MAE:  {paper_mae['ml']:.2f} ppm")
    print(f"paper Delta-ML MAE:   {paper_mae['delta_ml']:.2f} ppm")
    print(f"NOTE: this run used a smaller train/test carbon count than the "
          f"paper (proxy, not a literal reproduction) and no Delta-ML "
          f"baseline correction -- treat the comparison as directional, not "
          f"an apples-to-apples number.")


if __name__ == "__main__":
    main()
