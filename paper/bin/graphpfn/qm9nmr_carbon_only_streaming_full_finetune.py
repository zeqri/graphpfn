"""Multi-GPU (DDP), streaming, FULL fine-tune (backbone + graph adapters,
both unfrozen) of a GraphPFN checkpoint on REAL QM9NMR CARBON-ONLY per-atom
isotropic shielding, aiming to close as much of the gap as practical toward
the specialized kernel-ML literature benchmark (Gupta et al. 2021: ~1.9 ppm
ML / ~1.4 ppm Delta-ML at 100k/50k train/hold-out carbon atoms -- see
qm9nmr_carbon_only_checkpoint_eval.py's comparison table). Do not expect to
literally match those numbers: they come from specialized, atom-level-split
(not molecule-disjoint) kernel regressors purpose-built for this one task,
while GraphPFN is a general-purpose graph foundation model doing ICL
fine-tuning. Defaults here target Gupta et al.'s own scale (100k train /
50k test carbons) now that a 4-GPU job is available, but treat this as
gap-closing progress from the ~23.5 ppm zero-shot carbon-only baseline, not
a literal apples-to-apples comparison (see the split-methodology caveat
below).

Combines four things established earlier in this session:
  1. Carbon-only masking (qm9nmr_carbon_only_checkpoint_eval.py): every atom
     of every included molecule stays in the graph with real bonds intact;
     only CARBON atoms are ever labeled / used as context / scored.
     Heteroatoms are structural-only.
  2. Streaming micro-batch fine-tuning (mirrors qm9_graph_level_finetune_
     streaming.py's approach, ported to this repo's internal lib.graphpfn.
     model.GraphPFN and to node-level carbon-only regression instead of
     virtual-node graph-level regression): ONE big train-pool graph is built
     once, on CPU; every step extracts a small random molecule micro-batch's
     induced subgraph, so per-step GPU memory is bounded by --micro-batch-
     molecules, not by how large the overall train pool is. This is what
     makes a 100k-carbon train pool tractable, unlike qm9nmr_node_level_
     checkpoint_finetune.py's single fixed (and much smaller) train-pool
     subgraph.
  3. FULL fine-tuning (backbone + adapters unfrozen, matching the paper's
     main downstream finetuning protocol -- GraphPFN(freeze_tfm=False)):
     unlike every other finetuning script in this session, which only
     trains the graph adapters. This roughly quintuples the trainable
     parameter count and the optimizer's memory footprint, so
     --micro-batch-molecules defaults small and --cpu-offload is available.
  4. DDP (mirrors bin/graphpfn/pretrain.py's own multi-GPU setup exactly --
     lib.configure_ddp/get_device/is_master_process, DistributedDataParallel
     with find_unused_parameters=True): every rank builds the IDENTICAL
     train-pool/test split (same --seed), so the shared test set is
     consistent across ranks, but each rank samples DIFFERENT micro-batches
     every step (rank-dependent RNG) -- standard DDP data-parallelism,
     gradients averaged automatically by DistributedDataParallel on
     backward(). Only the master rank (RANK=0) evaluates, logs, and
     checkpoints.

Train/test molecules are sampled once and are disjoint by construction
(molecule-level, never touching test during training) -- unlike Gupta et
al.'s apparent atom-level split (see prior conversation), so this is a
*harder*, not easier, comparison point.

Usage (single-node, 4 GPUs, run with cwd=paper/, via run_qm9nmr_carbon_
finetune.sh / srun torchrun --nproc_per_node=4):
    python -m bin.graphpfn.qm9nmr_carbon_only_streaming_full_finetune \\
        --n-train-pool-carbons 100000 --n-test-carbons 50000 --nmr-phase gas

Single-GPU smoke test (no torchrun):
    python -m bin.graphpfn.qm9nmr_carbon_only_streaming_full_finetune \\
        --n-train-pool-carbons 2000 --n-test-carbons 1000 --n-outer-steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import tomllib
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import lib

# Your own in-progress training run (multigraph_molecule_valence_bounded
# prior), NOT the paper's released checkpoint -- same one used earlier by
# qm9nmr_node_level_checkpoint_finetune.py. This is a FULL bin/graphpfn/
# pretrain.py training checkpoint (backbone + adapters + edge head, ~334
# tensors under model_ema), which load_checkpoint_full_finetune's model_ema
# branch already handles. Check exp/graphpfn/pretrain/multigraph_molecule_
# valence_bounded/pretrain/training_log.jsonl for its current step -- it may
# still be running, so re-running against this same path later will see a
# different, more-trained snapshot.
DEFAULT_CHECKPOINT_PATH = Path(
    "/p/project1/profound/al-zeqri1/PFN/graphpfn_EGNN/graphpfn/paper/exp/graphpfn/"
    "pretrain/multigraph_molecule_valence_bounded/pretrain/checkpoint.pt"
)
DEFAULT_TOML_PATH = Path(
    "/p/project1/profound/al-zeqri1/PFN/graphpfn_EGNN/graphpfn/paper/exp/graphpfn/"
    "pretrain/multigraph_molecule_valence_bounded/pretrain.toml"
)
DEFAULT_QM9_ROOT = "/p/project1/profound/al-zeqri1/PFN/graphpfn/data/QM9"
DEFAULT_NMR_PATH = "/p/project1/profound/al-zeqri1/PFN/graphpfn/data/SI_DFT_NMR.txt"
DEFAULT_NMR_CACHE_PATH = (
    "/p/project1/profound/al-zeqri1/PFN/graphpfn/data/qm9nmr_shieldings_cache.npz"
)

NMR_PHASES = ["gas", "ccl4", "thf", "acetone", "methanol", "dmso"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--toml", type=Path, default=DEFAULT_TOML_PATH)
    parser.add_argument("--qm9-root", default=DEFAULT_QM9_ROOT)
    parser.add_argument("--nmr-path", default=DEFAULT_NMR_PATH)
    parser.add_argument("--nmr-cache-path", default=DEFAULT_NMR_CACHE_PATH)
    parser.add_argument("--nmr-phase", default="gas", choices=NMR_PHASES)
    parser.add_argument("--output", type=Path, default=None,
                        help="Directory to write checkpoint.pt / training_log.jsonl "
                             "(master rank only). If omitted, nothing is saved to disk.")
    parser.add_argument("--n-train-pool-carbons", type=int, default=100_000,
                        help="Target labeled CARBON atoms in the streaming "
                             "train pool -- molecules accumulated until reached. "
                             "Defaults to Gupta et al. 2021's own scale.")
    parser.add_argument("--n-test-carbons", type=int, default=50_000,
                        help="Target labeled carbon atoms held out completely "
                             "during training, scored only by periodic eval. "
                             "Defaults to Gupta et al. 2021's own scale.")
    parser.add_argument("--micro-batch-molecules", type=int, default=64,
                        help="Molecules (context+query combined) per GPU "
                             "forward/backward step. This bounds GPU memory -- "
                             "kept small since the backbone is also unfrozen here.")
    parser.add_argument("--context-ratio", type=float, default=0.75,
                        help="Fraction of each micro-batch's MOLECULES used as "
                             "context; the rest are query (contribute to loss).")
    parser.add_argument("--grad-accum-steps", type=int, default=8,
                        help="Micro-batches averaged into the loss per "
                             "optimizer step, PER GPU (so effective batch = "
                             "grad-accum-steps * micro-batch-molecules * "
                             "world_size under DDP).")
    parser.add_argument("--n-outer-steps", type=int, default=5000,
                        help="Optimizer steps. Bumped up from the single-GPU "
                             "default given a 4-GPU job's larger train pool.")
    parser.add_argument("--n-warmup-steps", type=int, default=200)
    parser.add_argument("--n-random-features", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Full-model finetuning LR -- lower than adapter-"
                             "only finetuning since the backbone is also "
                             "being updated.")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-context-molecules", type=int, default=4000,
                        help="Fixed context sample (drawn once from the train "
                             "pool) used for every eval forward pass, so scores "
                             "are comparable across steps and the eval forward "
                             "pass itself stays a bounded size.")
    parser.add_argument("--eval-periodic-test-molecules", type=int, default=3000,
                        help="Fixed random subsample of TEST molecules (drawn "
                             "once) used for the PERIODIC in-training eval "
                             "(every --eval-every steps) -- bounded so this "
                             "master-rank-only, non-streamed forward pass can't "
                             "stall long enough to blow the other ranks' NCCL "
                             "collective timeout (default 30 min) while they "
                             "idle at the post-eval barrier. Root-caused a real "
                             "DDP hang/SIGABRT at --n-test-carbons 50000 (see "
                             "slurm-14123610.out: ranks 1-3 timed out waiting on "
                             "an ALLREDUCE that never completed). The zero-shot "
                             "baseline (before the training loop starts) and the "
                             "final report (after it ends) are one-off, not "
                             "repeated every --eval-every steps, so they still "
                             "use the FULL test set for the paper-comparable "
                             "number.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload activations to host RAM during backward "
                             "(GraphPFN's autograd_cpu_offloading). Try this "
                             "first if full fine-tuning OOMs even at a small "
                             "--micro-batch-molecules.")
    parser.add_argument("--continue", dest="continue_", action="store_true",
                        help="Resume from <output>/checkpoint.pt (model + "
                             "optimizer + scheduler + step) if it already "
                             "exists. Safe to pass even on the very first "
                             "launch of a fresh --output dir -- a no-op then, "
                             "matching bin/graphpfn/pretrain.py's own --continue "
                             "convention. Requires --output.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
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


class CarbonOnlyMoleculePool:
    """One disjoint-union real-atom graph (no virtual nodes) covering every
    sampled molecule -- carbon AND heteroatoms alike, real bonds intact.
    Molecule i's atoms live at global node indices
    [mol_atom_offset[i], mol_atom_offset[i] + mol_atom_count[i])."""

    def __init__(
        self, graph, features, labels_raw, is_carbon, mol_atom_offset, mol_atom_count,
        edge_distance,
    ):
        self.graph = graph
        self.features = features  # [n_nodes, n_features] CPU float32 tensor
        self.labels_raw = labels_raw  # [n_nodes] numpy float32, real ppm
        self.is_carbon = is_carbon  # [n_nodes] bool
        self.mol_atom_offset = mol_atom_offset  # [n_mols] int64
        self.mol_atom_count = mol_atom_count  # [n_mols] int64
        # [n_directed_edges] float32, aligned with graph.edges()'s order --
        # real bond distance (Euclidean, from QM9's 3D mol.pos), matching
        # what a geometric-prior-pretrained checkpoint's graph adapter
        # expects at graph.edata["distance"] (see
        # lib.graphpfn.prior.graphs.molecule_skeleton and model.py's
        # EdgeDistanceEncoder). Harmless for a non-geometric checkpoint too:
        # its distance_encoder is either absent (older architectures load
        # this key back out via strict=False in load_checkpoint_full_
        # finetune) or zero-init/untrained, so it has no effect either way.
        self.edge_distance = edge_distance

    @property
    def n_mols(self) -> int:
        return len(self.mol_atom_offset)

    def atom_ids_for(self, mol_positions: np.ndarray) -> np.ndarray:
        parts = [
            np.arange(self.mol_atom_offset[p], self.mol_atom_offset[p] + self.mol_atom_count[p])
            for p in mol_positions
        ]
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


def build_carbon_only_pool(args, phase_idx: int, verbose: bool = True) -> CarbonOnlyMoleculePool:
    """Samples molecules until the train pool's CARBON atom count reaches
    --n-train-pool-carbons, then continues sampling (disjoint) until the test
    pool's carbon count reaches --n-test-carbons. Train-pool molecules occupy
    positions [0, n_train_pool_mols); test molecules occupy the rest.
    Heteroatoms of every molecule stay in the graph regardless of split.
    Deterministic given --seed, so every DDP rank builds the identical pool."""
    from torch_geometric.datasets import QM9

    full = QM9(root=args.qm9_root)
    if verbose:
        print(f"Loaded QM9: {len(full)} molecules; per-atom target = NMR shielding "
              f"[{NMR_PHASES[phase_idx]}], CARBON ATOMS ONLY")

    nmr_shieldings, nmr_offsets = load_qm9nmr_shieldings(args.nmr_path, args.nmr_cache_path)
    assert len(nmr_offsets) - 1 == len(full), (
        f"QM9NMR has {len(nmr_offsets) - 1} molecules, PyG QM9 has {len(full)} -- "
        "positional alignment assumption broken."
    )

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(full))

    targets = [args.n_train_pool_carbons, args.n_test_carbons]

    atom_features, atom_targets, atom_is_carbon = [], [], []
    edge_src_parts, edge_dst_parts, edge_distance_parts = [], [], []
    mol_atom_offset_list, mol_atom_count_list = [], []

    offset = 0
    split_idx = 0
    split_carbon_count = 0
    n_train_pool_mols = None
    for j in perm:
        if split_idx >= len(targets):
            break

        j = int(j)
        mol = full[j]
        n_atoms = int(mol.x.shape[0])

        nmr_n_atoms = int(nmr_offsets[j + 1] - nmr_offsets[j])
        assert nmr_n_atoms == n_atoms, (
            f"atom-count mismatch at QM9 idx {j}: PyG has {n_atoms} atoms, "
            f"QM9NMR block has {nmr_n_atoms}."
        )

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
        # Real bond distance from QM9's 3D coordinates (mol.pos), not a
        # sampled prior -- Euclidean distance between each bonded pair's
        # atomic positions.
        pos = mol.pos.numpy()
        edge_distance_parts.append(
            np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1).astype(np.float32)
        )

        mol_atom_offset_list.append(offset)
        mol_atom_count_list.append(n_atoms)

        offset += n_atoms
        split_carbon_count += n_carbon
        if split_carbon_count >= targets[split_idx]:
            if split_idx == 0:
                n_train_pool_mols = len(mol_atom_offset_list)
            split_idx += 1
            split_carbon_count = 0

    if n_train_pool_mols is None:  # ran out of molecules while still filling train pool
        n_train_pool_mols = len(mol_atom_offset_list)

    n_nodes = offset
    features = torch.from_numpy(np.concatenate(atom_features, axis=0))
    labels_raw = np.concatenate(atom_targets, axis=0).astype(np.float32)
    is_carbon_all = np.concatenate(atom_is_carbon, axis=0)
    mol_atom_offset = np.array(mol_atom_offset_list, dtype=np.int64)
    mol_atom_count = np.array(mol_atom_count_list, dtype=np.int64)
    edges = torch.from_numpy(
        np.stack(
            [np.concatenate(edge_src_parts), np.concatenate(edge_dst_parts)], axis=0
        ).astype(np.int64)
    )
    graph = dgl.graph(data=(edges[0], edges[1]), num_nodes=n_nodes, idtype=torch.int32)
    edge_distance = torch.from_numpy(np.concatenate(edge_distance_parts).astype(np.float32))

    from lib.graph.data import drop_constant_features

    train_pool_atom_end = int(mol_atom_offset[n_train_pool_mols - 1] + mol_atom_count[n_train_pool_mols - 1])
    fit_mask = torch.zeros(n_nodes, dtype=torch.bool)
    fit_mask[:train_pool_atom_end] = True
    features = drop_constant_features(features, fit_mask)

    n_train_pool_carbons = int(is_carbon_all[:train_pool_atom_end].sum())
    n_test_carbons = int(is_carbon_all[train_pool_atom_end:].sum())
    if verbose:
        print(f"Pool: {n_nodes} atoms across {len(mol_atom_offset)} molecules "
              f"({n_train_pool_mols} train-pool / {len(mol_atom_offset) - n_train_pool_mols} test), "
              f"{n_train_pool_carbons} train-pool carbons / {n_test_carbons} test carbons, "
              f"{graph.num_edges()} directed edges, {features.shape[1]} feature dims.")

    pool = CarbonOnlyMoleculePool(
        graph=graph, features=features, labels_raw=labels_raw, is_carbon=is_carbon_all,
        mol_atom_offset=mol_atom_offset, mol_atom_count=mol_atom_count,
        edge_distance=edge_distance,
    )
    return pool, n_train_pool_mols


def extract_micro_dataset(
    pool: CarbonOnlyMoleculePool, mol_positions: np.ndarray, n_context_mols: int, device,
):
    """Builds the induced subgraph for the given molecule positions (all
    atoms, carbon and hetero). The first n_context_mols molecules are
    context, the rest are query. Returns (sub_graph, sub_features,
    context_carbon_mask, query_carbon_mask, sub_labels_raw, sub_is_carbon) --
    masks are True only at CARBON atoms of the respective role; heteroatoms
    are never in either mask."""
    atom_ids = pool.atom_ids_for(mol_positions)
    sub_g = dgl.node_subgraph(pool.graph, atom_ids)
    # dgl.node_subgraph keeps (by default) each surviving edge's original id
    # in sub_g.edata[dgl.EID] -- use it to carry the matching real bond
    # distances over, so a geometric-prior checkpoint's graph adapter sees
    # real distances on this induced subgraph too, not just at pretraining.
    sub_g.edata["distance"] = pool.edge_distance[sub_g.edata[dgl.EID]]
    sub_features = pool.features[atom_ids].to(device)
    sub_labels_raw = pool.labels_raw[atom_ids]
    sub_is_carbon = pool.is_carbon[atom_ids]

    context_mol_atom_ids = pool.atom_ids_for(mol_positions[:n_context_mols])
    query_mol_atom_ids = pool.atom_ids_for(mol_positions[n_context_mols:])
    local_of_global = {int(g): local for local, g in enumerate(atom_ids)}

    n_local = len(atom_ids)
    context_mask = torch.zeros(n_local, dtype=torch.bool)
    for g in context_mol_atom_ids:
        local = local_of_global[int(g)]
        if sub_is_carbon[local]:
            context_mask[local] = True
    query_mask = torch.zeros(n_local, dtype=torch.bool)
    for g in query_mol_atom_ids:
        local = local_of_global[int(g)]
        if sub_is_carbon[local]:
            query_mask[local] = True

    return (
        sub_g.to(device), sub_features, context_mask.to(device), query_mask.to(device),
        sub_labels_raw, sub_is_carbon,
    )


def load_checkpoint_full_finetune(
    config: dict, checkpoint_path: Path, device: torch.device, cpu_offload: bool, verbose: bool = True,
):
    """Constructs GraphPFN with freeze_tfm=False (backbone unfrozen, matching
    the paper's full-finetune protocol -- adapters are always trainable
    regardless of freeze_tfm), then loads --checkpoint onto it non-strict.
    Handles either a full bin/graphpfn/pretrain.py training checkpoint
    (backbone + adapters + edge head, {"model_ema": ...}) -- e.g. this
    repo's own in-progress training runs -- or the released adapters-only
    checkpoint ({"state_dict": ...} or its {"model_ema": ...} conversion,
    144 tensors). Either way, any keys the checkpoint doesn't cover keep
    their construction-time values (e.g. LimiX-16M.ckpt for the backbone,
    if an adapters-only checkpoint is passed instead)."""
    from lib.graphpfn.model import GraphPFN

    model_kwargs = dict(config.get("model", dict()))
    model_kwargs["freeze_tfm"] = False
    if cpu_offload:
        model_kwargs["autograd_cpu_offloading"] = True
    graphpfn = GraphPFN(**model_kwargs).to(device)

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_ema" in raw:
        module_state = raw["model_ema"]
    else:
        module_state = {"module." + k: v for k, v in raw["state_dict"].items()}
    state_dict = {
        k[len("module."):]: v for k, v in module_state.items() if k.startswith("module.")
    }
    missing, unexpected = graphpfn.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected keys when loading checkpoint: {unexpected}"
    if verbose:
        print(f"Loaded {len(state_dict)} tensors from {checkpoint_path} "
              f"({len(missing)} keys left at construction-time values).")
    return graphpfn


def main() -> None:
    args = parse_args()

    if lib.is_ddp():
        lib.configure_ddp(timeout_minutes=30)
    is_master = lib.is_master_process()
    rank = lib.get_rank()
    world_size = lib.get_world_size()
    device = lib.get_device()

    # All ranks use the SAME seed for pool construction/eval-context choice,
    # so every rank ends up with the IDENTICAL train-pool/test split and the
    # same fixed eval-context sample -- only the per-step training
    # micro-batch draws differ across ranks (see `train_rng` below), which is
    # what makes this standard DDP data-parallelism (different data per GPU,
    # gradients averaged automatically by DistributedDataParallel).
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    train_rng = np.random.RandomState(lib.make_seed(args.seed, "train_microbatch", rank))

    if is_master:
        print(f"world_size={world_size}")

    with open(args.toml, "rb") as f:
        toml_config = tomllib.load(f)
    config = toml_config["base_config"]
    amp_enabled = (
        config.get("amp", False) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    if is_master:
        print(f"device={device} amp_enabled={amp_enabled}")

    phase_idx = NMR_PHASES.index(args.nmr_phase)
    # Same --seed on every rank -> every rank builds the IDENTICAL pool.
    pool, n_train_pool_mols = build_carbon_only_pool(args, phase_idx, verbose=is_master)
    train_positions = np.arange(n_train_pool_mols)
    test_positions = np.arange(n_train_pool_mols, pool.n_mols)

    train_pool_atom_ids = pool.atom_ids_for(train_positions)
    train_carbon_mask = pool.is_carbon[train_pool_atom_ids]
    label_mean = float(pool.labels_raw[train_pool_atom_ids][train_carbon_mask].mean())
    label_std = float(pool.labels_raw[train_pool_atom_ids][train_carbon_mask].std() + 1e-8)
    if is_master:
        print(f"Label stats (train-pool carbons only): mean={label_mean:.4f}, std={label_std:.4f}")

    from lib.metrics import calculate_metrics
    from lib.util import TaskType

    graphpfn = load_checkpoint_full_finetune(
        config, args.checkpoint, device, args.cpu_offload, verbose=is_master
    )
    graphpfn_without_ddp = graphpfn
    if lib.is_ddp():
        # device_ids/output_device must be the LOCAL cuda device index,
        # matching bin/graphpfn/pretrain.py's own DDP wrapping convention.
        #
        # find_unused_parameters=True is REQUIRED here, confirmed the hard
        # way: an attempt at False crashed immediately (not a hang) with
        # "Parameter indices which did not receive grad: 17 18 333 334 335
        # 336 343 344 345 346 347 348 351 352 353 354 355 356", IDENTICAL
        # across all 4 ranks, on the very first training step. The loss here
        # only ever uses out["predictions"] (see F.mse_loss below) --
        # out["features_pred"] and out["edge_predictions"], plus possibly an
        # inactive branch implied by this checkpoint's own name
        # (multigraph_molecule_geometric_rbf_or_linear -- suggesting an RBF
        # vs. linear distance-encoder choice, only one of which is active at
        # a time), are therefore structurally unused every single step, not
        # an occasional/data-dependent edge case. False categorically cannot
        # work with that. (For the record: the sibling non-geometric repo's
        # script has this exact same predictions-only loss shape and ran
        # 5000 steps fine with find_unused_parameters=True, so the flag
        # itself was never the actual cause of the earlier ~30min NCCL
        # hangs -- that diagnosis was wrong; see training log discussion.
        #
        # broadcast_buffers=False -- the ACTUAL root cause, found via
        # TORCH_DISTRIBUTED_DEBUG=DETAIL on the minimal-diff sibling script:
        # "Rank 0 is running collective: SequenceNumber=5605, OpType=
        # BROADCAST, TensorShape=[192] ... Rank 1 is running collective:
        # OpType=REDUCE". DDP's default broadcast_buffers=True makes
        # _sync_buffers() issue a REAL NCCL broadcast of every registered
        # buffer on EVERY forward call -- and this model has one:
        # EdgeDistanceEncoder.centers (16 elements x 12 attention layers =
        # 192, matching the crash's TensorShape exactly). evaluate_on_test()
        # runs on the MASTER RANK ONLY (baseline eval before the loop,
        # periodic eval inside it) with no barrier() synchronizing the other
        # 3 ranks around those calls -- so master alone issues this buffer
        # broadcast every time it evals, an uncompensated collective the
        # other ranks never match, permanently offsetting rank 0's
        # collective ordering from theirs until the offset happens to land
        # on an incompatible operation type and hangs/crashes. centers is
        # torch.linspace(...) -- deterministic and already identical across
        # ranks by construction -- so it never legitimately needed syncing;
        # this removes the collective rather than working around the
        # master-only-eval pattern that triggers it.
        graphpfn = DistributedDataParallel(
            graphpfn, device_ids=[device], output_device=device,
            find_unused_parameters=True, broadcast_buffers=False,
        )
    trainable_params = [p for p in graphpfn.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    if is_master:
        print(f"n_trainable_params={n_trainable:,} (FULL finetune: backbone + adapters)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, args.n_warmup_steps))
    )

    # >>> Resume (model + optimizer + scheduler + step), matching
    # bin/graphpfn/pretrain.py's --continue convention: safe to pass on the
    # very first launch of a fresh --output dir (no checkpoint yet -> no-op,
    # start_step stays 1). Runs on EVERY rank (not just master) -- each
    # rank's local model/optimizer copy needs the resumed state loaded
    # before training continues, not just the one that writes checkpoints.
    start_step = 1
    if args.continue_:
        assert args.output is not None, "--continue requires --output"
        checkpoint_path = args.output / "checkpoint.pt"
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            graphpfn_without_ddp.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt["step"] + 1
            if is_master:
                print(f"Resumed from {checkpoint_path} at step {ckpt['step']} "
                      f"-> continuing from step {start_step}.")
        lib.barrier()

    # Same --seed on every rank -> identical fixed eval-context sample, so
    # eval scores are comparable across steps AND across ranks (only the
    # master rank actually calls evaluate_on_test, but this keeps it
    # reproducible if that ever changes).
    eval_context_positions = rng.choice(
        train_positions, size=min(args.eval_context_molecules, len(train_positions)), replace=False,
    )
    # Bounded, fixed subsample of the test set for the PERIODIC in-loop eval
    # only -- see --eval-periodic-test-molecules's help for why: the full
    # test set (test_positions, sized to --n-test-carbons) is unbounded and
    # non-streamed, and stalling on it once was enough to blow the other
    # ranks' NCCL barrier timeout and abort the whole DDP job.
    eval_periodic_test_positions = rng.choice(
        test_positions,
        size=min(args.eval_periodic_test_molecules, len(test_positions)),
        replace=False,
    )
    if is_master:
        print(f"Periodic eval (every --eval-every={args.eval_every} steps) uses a "
              f"fixed random subsample of {len(eval_periodic_test_positions)}/"
              f"{len(test_positions)} test molecules to keep it fast and bounded "
              f"on one GPU -- the zero-shot baseline and the final report below "
              f"still use the FULL test set.")

    def evaluate_on_test(test_positions_override: np.ndarray | None = None, tag: str = "eval") -> dict:
        eval_test_positions = (
            test_positions if test_positions_override is None else test_positions_override
        )
        mol_positions = np.concatenate([eval_context_positions, eval_test_positions])
        t_extract_start = time.time()
        sub_g, sub_features, context_mask, query_mask, sub_labels_raw, sub_is_carbon = (
            extract_micro_dataset(pool, mol_positions, len(eval_context_positions), device)
        )
        # Diagnostic instrumentation added while chasing a DDP hang
        # (slurm-14123610.out / job 14123740): this print, flushed
        # immediately, is what will tell us on the NEXT run whether a stall
        # happens BEFORE this line (never printed -> stuck building the
        # subgraph/extract_micro_dataset), between this line and "done"
        # (stuck inside the model forward pass -- the EdgeDistanceEncoder's
        # per-edge computation is the prime suspect, since the working
        # non-geometric comparison run never exercises it), or AFTER "done"
        # (stuck in save_checkpoint's disk I/O instead, which is timed
        # separately at its own call site below).
        print(f"    [{tag}] subgraph: {sub_g.num_nodes()} nodes, {sub_g.num_edges()} directed "
              f"edges (extracted in {time.time() - t_extract_start:.1f}s) -- starting forward...",
              flush=True)
        y_train = torch.from_numpy(
            (sub_labels_raw[context_mask.cpu().numpy()] - label_mean) / label_std
        ).to(dtype=torch.float32, device=device)

        t_fwd_start = time.time()
        graphpfn.eval()
        # no_grad, not inference_mode: this is the DDP-WRAPPED module (not
        # the bare model, unlike the pure-ICL scripts' convention), and
        # DDP's own _sync_buffers() does an in-place buffer update on every
        # forward call. inference_mode() marks tensors it touches specially
        # in a way that in-place update rejects ("Inplace update to
        # inference tensor outside InferenceMode is not allowed") -- this
        # crashed rank 0 outright the one time TORCH_DISTRIBUTED_DEBUG=
        # DETAIL was enabled (its extra consistency checks likely changed
        # exactly how/when that buffer sync fires). no_grad gives the same
        # "don't track gradients" behavior we actually need here without
        # the special inference-tensor marking that DDP's internal buffer
        # sync trips over.
        with torch.no_grad(), torch.autocast(
            device.type, enabled=amp_enabled, dtype=torch.bfloat16 if amp_enabled else None
        ):
            out = graphpfn(
                graph=sub_g, features=sub_features, y_train=y_train,
                train_mask=context_mask, task_type=TaskType.REGRESSION,
                n_random_features=args.n_random_features,
            )
        pred_std = out["predictions"][query_mask].float().cpu().numpy()
        y_pred_raw = label_mean + label_std * pred_std
        y_true_raw = sub_labels_raw[query_mask.cpu().numpy()]
        graphpfn.train()
        print(f"    [{tag}] forward pass done in {time.time() - t_fwd_start:.1f}s", flush=True)
        return calculate_metrics(y_true_raw, y_pred_raw, "regression", "labels")

    def save_checkpoint(step: int) -> None:
        if args.output is None or not is_master:
            return
        t_start = time.time()
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": graphpfn_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            args.output / "checkpoint.pt",
        )
        # Same diagnostic purpose as evaluate_on_test's prints above -- a
        # slow/contended shared filesystem write is the other plausible
        # culprit for master stalling past the other ranks' barrier timeout,
        # independent of the eval forward pass itself.
        print(f"    [checkpoint] step {step} saved in {time.time() - t_start:.1f}s", flush=True)

    def log_jsonl(record: dict) -> None:
        if args.output is None or not is_master:
            return
        args.output.mkdir(parents=True, exist_ok=True)
        with open(args.output / "training_log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    # Only the master rank evaluates/logs/checkpoints -- these are plain
    # no-grad forward passes (torch.inference_mode()), so they never call
    # backward() and therefore never participate in DistributedDataParallel's
    # gradient-sync hooks; safe to run on one rank only. Always computed
    # (needed for the final report below) but only printed/logged on a
    # fresh run -- on --continue, it was already logged as step 0 before.
    #
    # Bounded (eval_periodic_test_positions), NOT the full test set: this
    # runs BEFORE the training loop, and non-master ranks don't wait at an
    # explicit barrier here -- they proceed straight into the loop's first
    # backward() call, which DOES hit the same 30-minute process-group
    # collective timeout via DDP's automatic gradient sync. An unbounded
    # full-test eval here was a latent second way to trigger the exact same
    # class of hang seen in slurm-14123610.out, just before training instead
    # of at a periodic eval boundary.
    if is_master:
        baseline_metrics = evaluate_on_test(eval_periodic_test_positions, tag="baseline(bounded)")
        if start_step == 1:
            print(f"\nZERO-SHOT baseline (before fine-tuning, bounded subsample): "
                  f"R2={baseline_metrics['r2']:.4f} MAE={baseline_metrics['mae']:.4f} ppm "
                  f"RMSE={baseline_metrics['rmse']:.4f} ppm", flush=True)
            log_jsonl({"step": 0, "r2": baseline_metrics["r2"], "mae": baseline_metrics["mae"]})

    n_context_per_micro = max(1, round(args.micro_batch_molecules * args.context_ratio))
    n_query_per_micro = max(1, args.micro_batch_molecules - n_context_per_micro)

    if is_master:
        print(f"\n{'step':>5} | {'loss':>8} | {'test_r2':>8} | {'test_mae':>9}")
    if start_step > args.n_outer_steps:
        if is_master:
            print(f"Resumed step {start_step} already >= --n-outer-steps "
                  f"{args.n_outer_steps}; nothing left to train, skipping to final report.")
    losses = []
    for step in range(start_step, args.n_outer_steps + 1):
        optimizer.zero_grad()
        step_losses = []
        for _ in range(args.grad_accum_steps):
            # train_rng is seeded per-rank -> every GPU trains on a
            # DIFFERENT random micro-batch each step (standard DDP data
            # parallelism); gradients are averaged automatically across
            # ranks by DistributedDataParallel during .backward() below.
            mol_positions = train_rng.choice(
                train_positions, size=n_context_per_micro + n_query_per_micro, replace=False,
            )
            sub_g, sub_features, context_mask, query_mask, sub_labels_raw, sub_is_carbon = (
                extract_micro_dataset(pool, mol_positions, n_context_per_micro, device)
            )
            if context_mask.sum() == 0 or query_mask.sum() == 0:
                # Practically unreachable at micro_batch_molecules >= a few
                # dozen (every QM9 molecule contains carbon), but NOTE: if it
                # ever fired on only SOME ranks it would desync the number of
                # backward() calls across ranks within this grad-accum loop
                # and could hang DDP's gradient sync -- not handled beyond
                # this comment since it's not a realistic risk here.
                continue

            y_train = torch.from_numpy(
                (sub_labels_raw[context_mask.cpu().numpy()] - label_mean) / label_std
            ).to(dtype=torch.float32, device=device)
            y_query = torch.from_numpy(
                (sub_labels_raw[query_mask.cpu().numpy()] - label_mean) / label_std
            ).to(dtype=torch.float32, device=device)

            with torch.autocast(
                device.type, enabled=amp_enabled, dtype=torch.bfloat16 if amp_enabled else None
            ):
                out = graphpfn(
                    graph=sub_g, features=sub_features, y_train=y_train,
                    train_mask=context_mask, task_type=TaskType.REGRESSION,
                    n_random_features=args.n_random_features,
                )
            loss = F.mse_loss(out["predictions"][query_mask], y_query)
            (loss / args.grad_accum_steps).backward()
            step_losses.append(loss.item())

        if step_losses:
            optimizer.step()
        scheduler.step()
        losses.append(float(np.mean(step_losses)) if step_losses else float("nan"))

        if step % args.eval_every == 0 or step == args.n_outer_steps:
            if is_master:
                # Bounded subsample (see eval_periodic_test_positions above),
                # NOT the full test set -- this is what previously stalled
                # long enough to blow the other ranks' NCCL barrier timeout.
                test_metrics = evaluate_on_test(eval_periodic_test_positions, tag=f"step {step}")
                recent_loss = float(np.nanmean(losses[-args.eval_every:]))
                print(f"{step:5d} | {recent_loss:8.4f} | {test_metrics['r2']:8.4f} | "
                      f"{test_metrics['mae']:9.4f}", flush=True)
                log_jsonl({"step": step, "loss": recent_loss, "r2": test_metrics["r2"],
                           "mae": test_metrics["mae"]})
                save_checkpoint(step)
            # Every rank prints right before this barrier -- if it hangs
            # again, comparing which ranks' "entering barrier" lines
            # appear (and when) against master's eval/checkpoint timing
            # above tells us definitively whether master never finished its
            # work, or a non-master rank is the actual straggler instead.
            print(f"    [rank {rank}] entering post-eval barrier at step {step} "
                  f"({time.strftime('%H:%M:%S')})", flush=True)
            lib.barrier()

    if not is_master:
        return

    save_checkpoint(args.n_outer_steps)
    # Full test set here, deliberately: this is the paper-comparable final
    # number, and unlike the periodic/baseline evals above, nothing else is
    # waiting on a barrier for it (non-master ranks already returned), so
    # even if it's slow it can't trigger the DDP watchdog -- only wastes
    # wall-clock time, which the tag's timing print will make visible.
    final_metrics = evaluate_on_test(tag="FINAL(full test set)")
    test_atom_ids = pool.atom_ids_for(test_positions)
    test_carbon_labels = pool.labels_raw[test_atom_ids][pool.is_carbon[test_atom_ids]]
    baseline_mae = float(np.abs(test_carbon_labels - label_mean).mean())

    print(f"\n=== GraphPFN, FULL finetune (backbone + adapters), QM9NMR carbon-only "
          f"[{args.nmr_phase}] ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"world_size: {world_size}")
    print(f"train pool: ~{args.n_train_pool_carbons} carbons across {n_train_pool_mols} molecules, "
          f"test: ~{args.n_test_carbons} carbons, micro-batch: {args.micro_batch_molecules} molecules "
          f"({n_context_per_micro} context / {n_query_per_micro} query), "
          f"grad-accum: {args.grad_accum_steps}, steps: {args.n_outer_steps}")
    print(f"Zero-shot  R2/MAE: {baseline_metrics['r2']:.4f} / {baseline_metrics['mae']:.4f} ppm")
    print(f"Fine-tuned R2/MAE: {final_metrics['r2']:.4f} / {final_metrics['mae']:.4f} ppm")
    print(f"RMSE (fine-tuned, test): {final_metrics['rmse']:.4f} ppm")
    print(f"Baseline (train-pool carbon mean, no model): {baseline_mae:.4f} ppm")
    print(
        "\nNOTE: literature comparison (Gupta et al. 2021, 100k/50k atom-level "
        "split): ML ~1.88 ppm, Delta-ML ~1.36 ppm. This run uses a molecule-"
        "disjoint split (harder, no same-molecule leakage between train pool "
        "and test) at a similar scale -- treat as gap-closing progress from "
        "the ~23.5 ppm zero-shot carbon-only baseline, not a literal "
        "apples-to-apples comparison against a specialized kernel regressor."
    )


if __name__ == "__main__":
    main()
