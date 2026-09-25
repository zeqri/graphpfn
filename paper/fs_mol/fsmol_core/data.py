"""FS-Mol task IO, per-molecule feature building, the stratified support/query split and the
FS-Mol graph -> pooler-input ("raw dict") conversion. Shared by every trainer and evaluator.

Task files are FS-Mol's <task>.jsonl.gz: one molecule per line with `graph` (32-col `node_features`
+ 3 typed `adjacency_lists`), `Property` (label), `SMILES`, `fingerprints` (2048-dim ECFP counts) and
`descriptors` (200 RDKit phys-chem values).
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FP_LEN = 2048     # ECFP count-fingerprint width stored in FS-Mol .jsonl.gz
DESC_LEN = 200    # RDKit phys-chem descriptor width stored in FS-Mol .jsonl.gz


# --------------------------------------------------------------------------------------------------
# task IO
# --------------------------------------------------------------------------------------------------

@dataclass
class TaskData:
    """One FS-Mol task, in file order.

    graphs[i]   {"adjacency_lists": [[[a, b], ...] x 3], "n_atoms": V[, "node_features": [[...]*32]*V]}
    labels      int64 [N]
    fps, descs  float32 [N, 2048] / [N, 200]      (only if read with permol=True)
    smiles      list[str]                         (only if read with smiles=True)
    atom_feats  per-molecule float tensors [V_i, C] replacing the stored node_features (or None: use them)
    permol      float32 [N, F] per-molecule vector to splice onto the pooled atom representation (or None)
    repair_status  per-molecule SMILES-repair status from robust_mol (x9 recipe only)
    """
    graphs: list
    labels: np.ndarray
    fps: np.ndarray | None = None
    descs: np.ndarray | None = None
    smiles: list | None = None
    atom_feats: list | None = None
    permol: np.ndarray | None = None
    repair_status: list | None = None

    @property
    def n_pos(self) -> int:
        return int(self.labels.sum())

    @property
    def degenerate(self) -> bool:
        """Fewer than 4 molecules or single-class -- nothing to evaluate / train on."""
        return len(self.graphs) < 4 or self.n_pos == 0 or self.n_pos == len(self.graphs)


def read_task(path: Path, *, node_features: bool, permol: bool, smiles: bool) -> TaskData:
    """Read one <task>.jsonl.gz. `node_features` keeps FS-Mol's own atom features in each graph,
    `permol` also reads the fingerprint + descriptor vectors, `smiles` the SMILES strings."""
    graphs, labels, fps, descs, smiles_list = [], [], [], [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            g = d["graph"]
            graph = {"adjacency_lists": g["adjacency_lists"], "n_atoms": len(g["node_features"])}
            if node_features:
                graph["node_features"] = g["node_features"]
            graphs.append(graph)
            labels.append(int(bool(float(d["Property"]))))
            if permol:
                fp = d.get("fingerprints")
                fps.append(np.zeros(FP_LEN, np.float32) if fp is None else np.asarray(fp, np.float32))
                de = d.get("descriptors")
                descs.append(np.full(DESC_LEN, np.nan, np.float32) if de is None else np.asarray(de, np.float32))
            if smiles:
                smiles_list.append(d["SMILES"])
    return TaskData(
        graphs=graphs,
        labels=np.asarray(labels, dtype=np.int64),
        fps=np.stack(fps, axis=0) if permol else None,
        descs=np.stack(descs, axis=0) if permol else None,
        smiles=smiles_list if smiles else None,
    )


def load_task_lists(task_list_json: Path) -> dict:
    """{"train": [...], "valid": [...], "test": [...]} task names from fsmol-0.1.json."""
    return json.loads(task_list_json.read_text())


# --------------------------------------------------------------------------------------------------
# per-molecule feature vector (spliced onto the pooled atom representation)
# --------------------------------------------------------------------------------------------------

def fold_fingerprint(fp, n_folded: int):
    """Classic ECFP folding: folded[:, j % n_folded] += fp[:, j].  fp: [N, D] -> [N, n_folded].
    n_folded <= 0 or >= D returns fp unchanged. D is zero-padded to a multiple of n_folded first."""
    n, d = fp.shape
    if n_folded <= 0 or n_folded >= d:
        return fp.astype(np.float32)
    pad = (-d) % n_folded
    if pad:
        fp = np.concatenate([fp, np.zeros((n, pad), fp.dtype)], axis=1)
    return fp.reshape(n, -1, n_folded).sum(axis=1).astype(np.float32)


def build_permol_features(fps, descs, mode: str, fp_fold: int):
    """Per-molecule feature matrix [N, F] to splice. fingerprint -> folded ECFP; descriptors ->
    raw RDKit descriptors with +/-inf -> NaN (so the encoder's NaN mask flags them as missing);
    both -> concatenation."""
    parts = []
    if mode in ("fingerprint", "both"):
        parts.append(fold_fingerprint(fps, fp_fold))
    if mode in ("descriptors", "both"):
        d = np.array(descs, dtype=np.float32)
        d[np.isinf(d)] = np.nan
        parts.append(d)
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------------------------------
# stratified support/query split -- byte-for-byte reimplementation of
# fs_mol.data.fsmol_task_sampler.StratifiedTaskSampler.sample(train_size=k, valid_size=0,
# test_size=None, allow_smaller_test=True)
# --------------------------------------------------------------------------------------------------

class SplitTooSmall(Exception):
    """Stand-in for FS-Mol's DatasetTooSmall / FoldTooSmall -- the caller skips the run."""


def stratified_support_query(labels, support_size: int, seed: int):
    from sklearn.model_selection import StratifiedShuffleSplit

    n = len(labels)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    order = np.concatenate([pos, neg])                                        # FS-Mol `samples` order
    strat = np.concatenate([np.zeros(len(pos), int), np.ones(len(neg), int)])  # FS-Mol `labels`

    num_test = n - support_size
    if support_size >= n or num_test < 2:
        raise SplitTooSmall(f"n={n}, support={support_size}, query={num_test}")
    sss = StratifiedShuffleSplit(n_splits=1, train_size=support_size, test_size=num_test, random_state=seed)
    try:
        tr, te = next(iter(sss.split(np.arange(n), strat)))
    except ValueError as e:
        raise SplitTooSmall(str(e))

    support_idx, query_idx = order[tr], order[te]
    if len(query_idx) < 2:
        raise SplitTooSmall("query fold < 2")
    for nm, sl in (("support", support_idx), ("query", query_idx)):
        npos = int(labels[sl].sum())
        if not (0 < npos < len(sl)):
            raise SplitTooSmall(f"{nm} fold single-class ({npos}/{len(sl)})")
    return support_idx, query_idx


def cap_query(query_idx, max_query: int | None, rng):
    """Random subset (sorted) of at most `max_query` query molecules; unchanged if already small."""
    if max_query is not None and len(query_idx) > max_query:
        return np.sort(rng.choice(query_idx, size=max_query, replace=False))
    return query_idx


# --------------------------------------------------------------------------------------------------
# FS-Mol graphs -> raw dict for encode_raw_dataset_on_gpu
# --------------------------------------------------------------------------------------------------

def graphs_to_raw(graphs, atom_feats, labels, n_context: int, features_per_group: int, bidirectional: bool):
    """Concatenate `graphs` (already in final support-then-query order) into one batched raw dict.

      * atom features  `atom_feats[i]` ([V_i, C] tensors) if given, else each graph's stored
                       `node_features`; zero-padded up to a multiple of `features_per_group`.
      * edges          union of the 3 typed adjacency lists (bond TYPE dropped -- the pooler has no
                       edge features), made BIDIRECTIONAL unless `bidirectional` is False.
      * isolated atoms every zero-in-degree atom gets a src==dst self-loop (nothing is dropped).
    """
    import torch

    af_list, es_list, ed_list, mid_list, y_list = [], [], [], [], []
    atom_offset = 0
    eval_pos_atoms = None

    for mol_idx, (g, y) in enumerate(zip(graphs, labels)):
        if mol_idx == n_context:
            eval_pos_atoms = atom_offset
        if atom_feats is None:
            feats = torch.from_numpy(np.asarray(g["node_features"], dtype=np.float32))
        else:
            feats = atom_feats[mol_idx]
        V = feats.shape[0]
        af_list.append(feats)

        src_parts, dst_parts = [], []
        for adj in g["adjacency_lists"]:               # 3 typed lists; type is dropped
            if adj is None or len(adj) == 0:
                continue
            a = np.asarray(adj, dtype=np.int64)        # [E, 2]
            src_parts.append(a[:, 0]); dst_parts.append(a[:, 1])
            if bidirectional:
                src_parts.append(a[:, 1]); dst_parts.append(a[:, 0])
        src = np.concatenate(src_parts) if src_parts else np.zeros(0, np.int64)
        dst = np.concatenate(dst_parts) if dst_parts else np.zeros(0, np.int64)

        indeg = np.zeros(V, dtype=np.int64)
        if dst.size:
            np.add.at(indeg, dst, 1)
        iso = np.flatnonzero(indeg == 0)               # zero-in-degree atoms -> self-loop
        if iso.size:
            src = np.concatenate([src, iso]); dst = np.concatenate([dst, iso])

        es_list.append(torch.from_numpy(src + atom_offset))
        ed_list.append(torch.from_numpy(dst + atom_offset))
        mid_list.append(torch.full((V,), mol_idx, dtype=torch.long))
        y_list.append(torch.tensor([float(y)], dtype=torch.float32))
        atom_offset += V
    if eval_pos_atoms is None:
        eval_pos_atoms = atom_offset

    atom_features = torch.cat(af_list, dim=0).float()   # [n_atoms, C]
    n_atoms, n_features = atom_features.shape
    rem = n_features % features_per_group
    if rem:
        atom_features = torch.cat(
            [atom_features, torch.zeros(n_atoms, features_per_group - rem, dtype=atom_features.dtype)], dim=-1
        )

    return {
        "atom_features": atom_features,
        "edges_src": torch.cat(es_list, dim=0).long(),
        "edges_dst": torch.cat(ed_list, dim=0).long(),
        "n_atoms": n_atoms,
        "molecule_id": torch.cat(mid_list, dim=0),
        "n_molecules": len(graphs),
        "y": torch.cat(y_list, dim=0),
        "eval_pos_atoms": eval_pos_atoms,
        "eval_pos_molecules": n_context,
    }


def make_episode(task: TaskData, support_idx, query_idx, features_per_group: int, bidirectional: bool):
    """(raw, permol_ordered) for one in-context episode: support molecules first, then query.
    `permol_ordered` is the per-molecule splice matrix in the same order (None if the task has none)."""
    order = np.concatenate([support_idx, query_idx])
    raw = graphs_to_raw(
        [task.graphs[i] for i in order],
        None if task.atom_feats is None else [task.atom_feats[i] for i in order],
        task.labels[order], n_context=len(support_idx),
        features_per_group=features_per_group, bidirectional=bidirectional,
    )
    return raw, (None if task.permol is None else task.permol[order])
