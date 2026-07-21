"""Node-level regression on REAL QM9NMR per-atom isotropic shielding,
CARBON ATOMS ONLY, using the same pipeline as qm9nmr_node_level_icl_carbon_
only_matched_qm9nmr_paper.py, but with an ATOM-LEVEL random train/val/test
split instead of a molecule-disjoint one.

WHY THIS SCRIPT EXISTS -- LEAKAGE, ON PURPOSE: the sibling *_matched_
qm9nmr_paper.py script (and *_matched_city_roads.py before it) assigns
WHOLE molecules to a single split, so a test carbon's molecule is entirely
unseen by the model. The paper being compared against (Gupta, Chakraborty &
Ramakrishnan 2021, MLST 2 035010) does the opposite: its KRR treats every
13C atom as an independent training/test entity with no molecule-disjointness
constraint at all ("a random set of 100k entries for training... a separate
subset of 50k nuclei -- not overlapping with the 100k training entries").
Concretely, that means carbon #3 of some molecule can be in their training
set while carbon #7 of THE SAME molecule is in their test set -- the model
(or here, the ICL context) gets to see most of that molecule's local
chemical environment already, just not that one specific atom's label.

This script reproduces that same leakage pattern: it samples CARBON ATOMS
uniformly at random (not whole molecules) into train/val/test, while still
keeping every atom of every touched molecule (carbon and heteroatoms alike)
in the graph with real bonds intact, exactly like the other QM9NMR ICL
scripts. Structurally, a query carbon's own molecule is very likely to have
1+ of its OTHER carbons sitting in the ICL context (train_mask) -- directly
reachable via the bond graph the model attends/message-passes over. That is
the leakage: it should make this run's MAE noticeably lower than the
molecule-disjoint sibling script's MAE on the *same* checkpoint and *same*
total carbon-count scale, and the SIZE of that drop is itself the useful
number -- it estimates how much of the paper's reported accuracy is coming
from "free" within-molecule information their split allows, versus how much
reflects genuine chemical generalization.

Practical implication for comparing to the paper: this run is now
apples-to-apples with their SPLIT METHODOLOGY (atom-level, no molecule-
disjointness), but still not with their MODEL (their FCHL-KRR is a
descriptor purpose-built and cross-validated for this one task; here it's a
general pretrained graph model doing zero-gradient ICL) or their SCALE (see
below). Treat the resulting MAE as the fairest "how good is our model
relative to theirs" number available so far, not a guarantee of parity.

SCALE: same feasibility constraint as the molecule-disjoint sibling script
applies (single non-streamed forward pass, no micro-batching) -- defaults
below are DELIBERATELY KEPT IDENTICAL to *_matched_qm9nmr_paper.py's
defaults (20k train / 0 val / 10k test carbons, i.e. the same total pool
size and therefore roughly the same total graph size), so a run of this
script is a controlled, like-for-like comparison against a run of that
script: same checkpoint, same carbon-count budget, the ONLY difference is
molecule-disjoint vs. atom-level (leaky) splitting. Override --n-train-
carbons/--n-test-carbons together if you want a different scale, but keep
them matched across the two scripts if you want the leakage effect to stay
isolated.

GEOMETRY, checkpoint format, ensembling, and the carbon-only/structural-
heteroatom masking convention are unchanged from the sibling scripts -- see
qm9nmr_node_level_icl_carbon_only_matched_city_roads.py's docstring for the
full rationale (real bond distances from QM9's mol.pos feed the internal
model's EdgeDistanceEncoder; checkpoints are this repo's own bin/graphpfn/
pretrain.py {"model_ema": ...} format; n_random_features>0 is the model's
own source of ensemble diversity across --n-members independent forward
passes).

Usage (run with cwd=paper/, inside a GPU allocation):
    python -m bin.graphpfn.qm9nmr_node_level_icl_carbon_only_matched_qm9nmr_paper_atom_level --device cuda
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
# ATOM-LEVEL random split (what this script now matches), mean absolute
# error in ppm. Gas phase is this script's default --nmr-phase; the other
# phases' paper numbers are the remaining rows of table 1.
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

# Deliberately identical to *_matched_qm9nmr_paper.py's defaults -- see
# module docstring's "SCALE" note: keeping these matched across the two
# scripts is what makes a leaky-vs-molecule-disjoint comparison controlled.
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
                             "other QM9NMR ICL scripts; also selects which "
                             "row of the paper's table 1 to compare against.")
    parser.add_argument("--n-train-carbons", type=int, default=DEFAULT_N_TRAIN_CARBONS,
                        help="Exact labeled train (context) CARBON atom "
                             "count, sampled uniformly at random across ALL "
                             "carbons in the assembled molecule pool (atom-"
                             "level, may include sibling carbons of test "
                             "atoms' own molecules -- see module docstring). "
                             f"Paper uses {PAPER_N_TRAIN_ATOMS:,}.")
    parser.add_argument("--n-val-carbons", type=int, default=DEFAULT_N_VAL_CARBONS,
                        help="Exact labeled val carbon atom count (unused by "
                             "metrics; ICL has no early stopping). Paper has "
                             "no val split either -- default 0.")
    parser.add_argument("--n-test-carbons", type=int, default=DEFAULT_N_TEST_CARBONS,
                        help="Exact labeled test (query) carbon atom count. "
                             f"Paper uses {PAPER_N_TEST_ATOMS:,}.")
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


def build_atom_level_carbon_only_dataset(args, phase_idx: int):
    """Samples whole molecules (in random order) into a pool until the
    pool's total CARBON atom count reaches --n-train-carbons +
    --n-val-carbons + --n-test-carbons, keeping every atom (carbon and
    heteroatoms alike) of every touched molecule in the graph with real
    bonds intact -- same graph-construction convention as the other QM9NMR
    ICL scripts.

    Then, UNLIKE those sibling scripts, the train/val/test assignment
    itself is done by permuting ALL carbon atoms in the pool (across every
    molecule, ignoring molecule boundaries) and slicing off exactly
    --n-train-carbons / --n-val-carbons / --n-test-carbons of them -- this
    is the atom-level (leaky) split the module docstring describes: a
    molecule can, and very likely will, have some carbons in train and
    others in test. Any leftover carbons (the pool usually overshoots the
    target slightly, from the last molecule that pushed it over) are left
    unlabeled -- structural-only, same treatment as heteroatoms.

    Returns (graph, features, targets_raw, masks), same shapes/semantics as
    the sibling scripts' build_matched_carbon_only_dataset.
    """
    import dgl
    import torch
    from torch_geometric.datasets import QM9

    full = QM9(root=args.qm9_root)
    print(f"Loaded QM9: {len(full)} molecules; per-atom target = NMR shielding "
          f"[{NMR_PHASES[phase_idx]}], CARBON ATOMS ONLY, ATOM-LEVEL split (leakage allowed)")

    nmr_shieldings, nmr_offsets = load_qm9nmr_shieldings(args.nmr_path, args.nmr_cache_path)
    assert len(nmr_offsets) - 1 == len(full), (
        f"QM9NMR has {len(nmr_offsets) - 1} molecules, PyG QM9 has {len(full)} -- "
        "positional alignment assumption broken."
    )

    total_target_carbons = args.n_train_carbons + args.n_val_carbons + args.n_test_carbons

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(full))

    atom_features = []
    atom_targets = []
    atom_is_carbon = []
    edge_src_parts = []
    edge_dst_parts = []
    edge_distance_parts = []

    offset = 0
    pool_carbon_count = 0
    for j in perm:
        if pool_carbon_count >= total_target_carbons:
            break  # pool has enough carbons to fill all three splits

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
        pool_carbon_count += n_carbon

    n_nodes = offset
    features = torch.from_numpy(np.concatenate(atom_features, axis=0))
    targets_raw = np.concatenate(atom_targets, axis=0).astype(np.float32)
    is_carbon_all = np.concatenate(atom_is_carbon, axis=0)
    edge_src = np.concatenate(edge_src_parts)
    edge_dst = np.concatenate(edge_dst_parts)
    edge_distance = torch.from_numpy(np.concatenate(edge_distance_parts).astype(np.float32))

    graph = dgl.graph(data=(edge_src, edge_dst), num_nodes=n_nodes, idtype=torch.int32)
    graph.edata["distance"] = edge_distance

    # ATOM-LEVEL random split across every carbon in the pool, ignoring
    # which molecule each one came from -- this is the leakage: a carbon
    # sitting in train_mask can be the bonded neighbor (or a few hops away,
    # same ring/functional group) of a carbon sitting in test_mask.
    carbon_atom_ids = np.flatnonzero(is_carbon_all)
    assert len(carbon_atom_ids) >= total_target_carbons, (
        f"molecule pool only yielded {len(carbon_atom_ids)} carbon atoms, "
        f"need {total_target_carbons} -- this shouldn't happen with QM9's "
        f"812k total carbons unless targets are set unreasonably high."
    )
    carbon_perm = rng.permutation(carbon_atom_ids)
    train_ids = carbon_perm[:args.n_train_carbons]
    val_ids = carbon_perm[args.n_train_carbons: args.n_train_carbons + args.n_val_carbons]
    test_ids = carbon_perm[
        args.n_train_carbons + args.n_val_carbons:
        args.n_train_carbons + args.n_val_carbons + args.n_test_carbons
    ]
    # Any carbons beyond the exact target counts (the pool typically
    # overshoots slightly, since the last molecule that crossed
    # total_target_carbons is kept whole) are left unlabeled -- structural-
    # only, same treatment as heteroatoms.

    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[train_ids] = True
    val_mask[val_ids] = True
    test_mask[test_ids] = True
    masks = {"train": train_mask, "val": val_mask, "test": test_mask}

    n_heteroatoms = n_nodes - int(is_carbon_all.sum())
    n_unlabeled_carbons = int(is_carbon_all.sum()) - int(
        train_mask.sum() + val_mask.sum() + test_mask.sum()
    )

    print(f"Split (carbon-only, ATOM-LEVEL random, leakage allowed, exact targets "
          f"{args.n_train_carbons}/{args.n_val_carbons}/{args.n_test_carbons} "
          f"carbon atoms -- paper uses {PAPER_N_TRAIN_ATOMS:,}/0/{PAPER_N_TEST_ATOMS:,}): "
          f"{int(train_mask.sum())} train / {int(val_mask.sum())} val / "
          f"{int(test_mask.sum())} test labeled CARBON atoms "
          f"({n_nodes} atoms total incl. {n_heteroatoms} structural-only heteroatoms and "
          f"{n_unlabeled_carbons} unlabeled leftover carbons, "
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
    graph, features, targets_raw, masks = build_atom_level_carbon_only_dataset(args, phase_idx)
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
          f"QM9NMR carbon-only [{args.nmr_phase}], ATOM-LEVEL split "
          f"(leakage allowed, matches the QM9-NMR paper's split methodology) ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"ensemble members: {args.n_members}, seed: {args.seed}")
    print(f"train/test carbons this run: {int(train_mask_np.sum())}/{int(test_mask.sum())} "
          f"(paper: {PAPER_N_TRAIN_ATOMS:,}/{PAPER_N_TEST_ATOMS:,} -- see module "
          f"docstring's 'SCALE' note for why this run is smaller)")
    print(f"R2:  {metrics['r2']:.4f}")
    print(f"MAE: {metrics['mae']:.4f} ppm")
    print(f"MAE of constant (train-mean) baseline: {baseline_mae:.4f} ppm")
    print(f"--- paper's FCHL-KRR benchmark at 100k/50k, [{args.nmr_phase}] "
          f"(Gupta, Chakraborty & Ramakrishnan 2021, table 1) ---")
    print(f"paper direct ML MAE:  {paper_mae['ml']:.2f} ppm")
    print(f"paper Delta-ML MAE:   {paper_mae['delta_ml']:.2f} ppm")
    print(f"NOTE: this run's split methodology (atom-level, leakage allowed) "
          f"now matches the paper's; scale ({int(train_mask_np.sum())}/"
          f"{int(test_mask.sum())} vs. their 100k/50k) and model class (general "
          f"pretrained graph-ICL vs. a descriptor purpose-built and cross-"
          f"validated for this task) still differ. Compare this run's MAE "
          f"against the *_matched_qm9nmr_paper.py (molecule-disjoint) script's "
          f"MAE at the SAME --n-train-carbons/--n-test-carbons to isolate how "
          f"much of the paper's reported accuracy leakage alone can explain.")


if __name__ == "__main__":
    main()
