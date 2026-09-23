"""Atom-feature builders for the evaluation scripts.

Two atom-feature variants are used everywhere:
  - extra_features_only   -- the 31 RDKit columns in EXTRA_FEATURE_NAMES.
  - x_plus_extra_features -- [9 MoleculeNet-style from_smiles columns (X_COLUMNS) | 31 extra] = 40.
None of the 31 extra columns duplicates a from_smiles column (a few, e.g. atomic_mass, are
functions of atomic_num).

Every builder returns a float32 [n_atoms, n_features] tensor in the dataset's own atom order and
raises on an atom-count mismatch instead of silently misaligning. Sources:
  - MoleculeNet: data.x + features parsed from data.smiles.
  - ZINC: no SMILES -- an RDKit mol is rebuilt from ZINC's own atom/bond vocabulary graph.
  - AQSOL: no SMILES -- each graph is matched to its data_curated.csv SMILES (cached), with a
    topology-rebuild fallback.
"""

from __future__ import annotations

import math
import os

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, Crippen
from rdkit.Chem.EState import EState
from rdkit.Chem import rdMolDescriptors

EXTRA_FEATURE_NAMES = [
    # Valence
    "total_valence",

    # Bond environment
    "n_single_bonds",
    "n_double_bonds",
    "n_triple_bonds",
    "n_aromatic_bonds",
    "n_conjugated_bonds",
    "n_ring_bonds",

    # Ring environment
    "in_3_ring",
    "in_4_ring",
    "in_5_ring",
    "in_6_ring",
    "smallest_ring_size",
    "num_rings_containing_atom",

    # Derived chemical descriptors
    "gasteiger_charge",
    "crippen_logp_contrib",
    "crippen_mr_contrib",
    "tpsa_contrib",
    "estate_index",

    # Chemical roles
    "is_hbond_donor",
    "is_hbond_acceptor",

    # Periodic-table properties
    "atomic_mass",
    "covalent_radius",
    "vdw_radius",
    "num_outer_electrons",

    # Neighbour composition
    "num_carbon_neighbors",
    "num_nitrogen_neighbors",
    "num_oxygen_neighbors",
    "num_sulfur_neighbors",
    "num_phosphorus_neighbors",
    "num_halogen_neighbors",
    "num_heteroatom_neighbors",
]
N_EXTRA_FEATURES = len(EXTRA_FEATURE_NAMES)  # 31

# The 9 torch_geometric `from_smiles` columns, in data.x's own column order (documentation only --
# data.x is used as-is, this list is never used to re-derive it).
X_COLUMNS = [
    "atomic_num", "chirality", "degree_total", "formal_charge", "num_hs_total",
    "num_radical_electrons", "hybridization", "is_aromatic", "is_in_ring",
]
X_PLUS_EXTRA_FEATURE_COLUMNS = X_COLUMNS + EXTRA_FEATURE_NAMES
N_X_PLUS_EXTRA_FEATURES = len(X_PLUS_EXTRA_FEATURE_COLUMNS)  # 9 + 31 = 40


def safe_float(value, default=0.0):
    """Replace NaN and infinite descriptor values with a safe default."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    return value if math.isfinite(value) else default


def get_donor_acceptor_atoms(mol):
    """Return sets containing RDKit donor and acceptor atom indices."""
    feature_definition = os.path.join(
        RDConfig.RDDataDir,
        "BaseFeatures.fdef",
    )
    feature_factory = ChemicalFeatures.BuildFeatureFactory(
        feature_definition
    )

    donor_atoms = set()
    acceptor_atoms = set()

    for feature in feature_factory.GetFeaturesForMol(mol):
        atom_ids = feature.GetAtomIds()

        if feature.GetFamily() == "Donor":
            donor_atoms.update(atom_ids)
        elif feature.GetFamily() == "Acceptor":
            acceptor_atoms.update(atom_ids)

    return donor_atoms, acceptor_atoms


def get_atom_ring_sizes(mol):
    """Return the sizes of all rings containing each atom."""
    ring_sizes = [[] for _ in range(mol.GetNumAtoms())]

    for ring in mol.GetRingInfo().AtomRings():
        size = len(ring)

        for atom_idx in ring:
            ring_sizes[atom_idx].append(size)

    return ring_sizes


def featurize_extra_atoms(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    return featurize_extra_atoms_from_mol(mol)


def featurize_extra_atoms_from_mol(mol):
    """The 31 extra columns for an already-built, sanitized RDKit mol (no SMILES round-trip, so
    the mol's own atom order is kept)."""
    # Calculates and stores _GasteigerCharge on every atom.
    AllChem.ComputeGasteigerCharges(mol)

    crippen_contribs = Crippen._GetAtomContribs(mol)
    estate_indices = EState.EStateIndices(mol)
    tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol)

    donor_atoms, acceptor_atoms = get_donor_acceptor_atoms(mol)
    atom_ring_sizes = get_atom_ring_sizes(mol)

    periodic_table = Chem.GetPeriodicTable()
    rows = []

    halogen_atomic_numbers = {9, 17, 35, 53}  # F, Cl, Br, I

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()

        bond_counts = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 0,
            Chem.BondType.TRIPLE: 0,
            Chem.BondType.AROMATIC: 0,
        }

        n_conjugated_bonds = 0
        n_ring_bonds = 0

        for bond in atom.GetBonds():
            bond_type = bond.GetBondType()

            if bond_type in bond_counts:
                bond_counts[bond_type] += 1

            if bond.GetIsConjugated():
                n_conjugated_bonds += 1

            if bond.IsInRing():
                n_ring_bonds += 1

        neighbor_atomic_numbers = [
            neighbor.GetAtomicNum()
            for neighbor in atom.GetNeighbors()
        ]

        num_carbon_neighbors = neighbor_atomic_numbers.count(6)
        num_nitrogen_neighbors = neighbor_atomic_numbers.count(7)
        num_oxygen_neighbors = neighbor_atomic_numbers.count(8)
        num_phosphorus_neighbors = neighbor_atomic_numbers.count(15)
        num_sulfur_neighbors = neighbor_atomic_numbers.count(16)

        num_halogen_neighbors = sum(
            z in halogen_atomic_numbers
            for z in neighbor_atomic_numbers
        )

        # Every non-carbon and non-hydrogen neighbour is a heteroatom.
        num_heteroatom_neighbors = sum(
            z not in {1, 6}
            for z in neighbor_atomic_numbers
        )

        ring_sizes = atom_ring_sizes[atom_idx]
        smallest_ring_size = min(ring_sizes) if ring_sizes else 0

        logp_contrib, mr_contrib = crippen_contribs[atom_idx]

        gasteiger_charge = safe_float(
            atom.GetProp("_GasteigerCharge")
            if atom.HasProp("_GasteigerCharge")
            else 0.0
        )

        row = [
            # Valence
            atom.GetTotalValence(),

            # Bond environment
            bond_counts[Chem.BondType.SINGLE],
            bond_counts[Chem.BondType.DOUBLE],
            bond_counts[Chem.BondType.TRIPLE],
            bond_counts[Chem.BondType.AROMATIC],
            n_conjugated_bonds,
            n_ring_bonds,

            # Ring environment
            float(3 in ring_sizes),
            float(4 in ring_sizes),
            float(5 in ring_sizes),
            float(6 in ring_sizes),
            smallest_ring_size,
            len(ring_sizes),

            # Derived descriptors
            gasteiger_charge,
            safe_float(logp_contrib),
            safe_float(mr_contrib),
            safe_float(tpsa_contribs[atom_idx]),
            safe_float(estate_indices[atom_idx]),

            # Chemical roles
            float(atom_idx in donor_atoms),
            float(atom_idx in acceptor_atoms),

            # Periodic-table properties
            periodic_table.GetAtomicWeight(atomic_num),
            periodic_table.GetRcovalent(atomic_num),
            periodic_table.GetRvdw(atomic_num),
            periodic_table.GetNOuterElecs(atomic_num),

            # Neighbour composition
            num_carbon_neighbors,
            num_nitrogen_neighbors,
            num_oxygen_neighbors,
            num_sulfur_neighbors,
            num_phosphorus_neighbors,
            num_halogen_neighbors,
            num_heteroatom_neighbors,
        ]

        rows.append(row)

    extra_x = torch.tensor(rows, dtype=torch.float32)

    return mol, extra_x, EXTRA_FEATURE_NAMES


def extra_atom_features_from_smiles(smiles: str, n_atoms_expected: int) -> torch.Tensor:
    """[n_atoms, 31] in RDKit atom order -- the same order torch_geometric's from_smiles uses for
    data.x / edge_index. Raises if the atom count differs from n_atoms_expected."""
    mol, extra_x, _ = featurize_extra_atoms(smiles)
    if mol.GetNumAtoms() != n_atoms_expected:
        raise ValueError(
            f"RDKit atom count {mol.GetNumAtoms()} != data.x atom count {n_atoms_expected} for {smiles!r} "
            "-- atom order would not line up with the existing bond graph / self-loops / molecule_id."
        )
    return extra_x


def extra_atom_features_from_mol(mol, n_atoms_expected: int) -> torch.Tensor:
    """[n_atoms, 31] for a mol already in the dataset's atom order. Raises if the atom count
    differs from n_atoms_expected."""
    _, extra_x, _ = featurize_extra_atoms_from_mol(mol)
    if mol.GetNumAtoms() != n_atoms_expected:
        raise ValueError(
            f"RDKit atom count {mol.GetNumAtoms()} != expected atom count {n_atoms_expected} -- "
            "atom order would not line up with the existing bond graph / self-loops / molecule_id."
        )
    return extra_x


def x_plus_extra_atom_features(data) -> torch.Tensor:
    """[n_atoms, 40] for a MoleculeNet molecule: data.x's 9 from_smiles columns | 31 extra columns."""
    x_cols = data.x.float()
    extra_cols = extra_atom_features_from_smiles(data.smiles, n_atoms_expected=data.x.shape[0])
    return torch.cat([x_cols, extra_cols], dim=-1)


# --------------------------------------------------------------------------------------------------
# 9 MoleculeNet-style columns from an RDKit mol (for datasets whose PyG .x is NOT from_smiles)
# --------------------------------------------------------------------------------------------------

def atom_features_9dim_from_mol(mol) -> torch.Tensor:
    """[n_atoms, 9] float32 -- the exact per-atom read torch_geometric.utils.smiles.from_smiles does,
    through the same x_map tables, so the encoding matches MoleculeNet's own data.x column-for-column.
    A categorical value missing from its table falls back to the table's last slot instead of raising."""
    from torch_geometric.utils.smiles import x_map

    def idx(key: str, value) -> int:
        table = x_map[key]
        return table.index(value) if value in table else len(table) - 1

    rows = [
        [
            idx("atomic_num", atom.GetAtomicNum()),
            idx("chirality", str(atom.GetChiralTag())),
            idx("degree", atom.GetTotalDegree()),
            idx("formal_charge", atom.GetFormalCharge()),
            idx("num_hs", atom.GetTotalNumHs()),
            idx("num_radical_electrons", atom.GetNumRadicalElectrons()),
            idx("hybridization", str(atom.GetHybridization())),
            idx("is_aromatic", atom.GetIsAromatic()),
            idx("is_in_ring", atom.IsInRing()),
        ]
        for atom in mol.GetAtoms()
    ]
    return torch.tensor(rows, dtype=torch.float32)


def x_plus_extra_from_mol(mol, n_atoms_expected: int) -> torch.Tensor:
    """[n_atoms, 40] = [9 MoleculeNet-style columns | 31 extra columns], both read off the SAME
    sanitized mol, so atom order is shared with whatever graph the mol was rebuilt from."""
    x9 = atom_features_9dim_from_mol(mol)
    extra31 = extra_atom_features_from_mol(mol, n_atoms_expected=n_atoms_expected)
    x40 = torch.cat([x9, extra31], dim=-1)
    if x40.shape != (n_atoms_expected, N_X_PLUS_EXTRA_FEATURES):
        raise ValueError(f"rebuilt {tuple(x40.shape)} != expected ({n_atoms_expected}, {N_X_PLUS_EXTRA_FEATURES})")
    return x40


# --------------------------------------------------------------------------------------------------
# ZINC (subset): no SMILES -- rebuild an RDKit mol from ZINC's own atom/bond vocab graph
# --------------------------------------------------------------------------------------------------

class _ZincDictionaryStub:
    """Unpickle stand-in for the '__main__.Dictionary' class ZINC's raw atom_dict.pickle /
    bond_dict.pickle were pickled from (benchmarking-gnns data prep)."""


def load_zinc_vocab(zinc_root) -> tuple[list[str], list[str]]:
    """(atom idx2word, bond idx2word) from <zinc_root>/raw/{atom,bond}_dict.pickle -- ZINC's 28-entry
    atom vocab ('C', 'N H1 +', 'O -', ...) and 4-entry bond vocab (NONE/SINGLE/DOUBLE/TRIPLE)."""
    import __main__
    import pickle
    from pathlib import Path

    raw_dir = Path(zinc_root) / "raw"
    __main__.Dictionary = _ZincDictionaryStub
    with open(raw_dir / "atom_dict.pickle", "rb") as f:
        atom_dict = pickle.load(f)
    with open(raw_dir / "bond_dict.pickle", "rb") as f:
        bond_dict = pickle.load(f)
    return list(atom_dict.idx2word), list(bond_dict.idx2word)


def _parse_zinc_atom_token(token: str) -> tuple[str, int, int]:
    """'N H1 +' -> ('N', 1, +1); 'O -' -> ('O', 0, -1); 'C' -> ('C', 0, 0)."""
    parts = token.split()
    n_explicit_h, formal_charge = 0, 0
    for p in parts[1:]:
        if p.startswith("H"):
            n_explicit_h = int(p[1:])
        elif p == "+":
            formal_charge = 1
        elif p == "-":
            formal_charge = -1
    return parts[0], n_explicit_h, formal_charge


def zinc_data_to_rdkit_mol(data, atom_vocab: list[str], bond_vocab: list[str]):
    """Rebuild + sanitize an RDKit mol from one ZINC Data object (own-vocab .x / edge_attr), atom
    order == ZINC node order. ZINC is kekulized; SanitizeMol re-perceives rings, aromaticity,
    hybridization and implicit Hs."""
    bond_order = {"SINGLE": Chem.BondType.SINGLE, "DOUBLE": Chem.BondType.DOUBLE, "TRIPLE": Chem.BondType.TRIPLE}
    rw = Chem.RWMol()
    for atom_id in data.x.view(-1).tolist():
        symbol, n_h, charge = _parse_zinc_atom_token(atom_vocab[int(atom_id)])
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(charge)
        if n_h:
            atom.SetNumExplicitHs(n_h)
        rw.AddAtom(atom)
    seen: set[tuple[int, int]] = set()
    for (src, dst), b in zip(data.edge_index.t().tolist(), data.edge_attr.view(-1).tolist()):
        if (src, dst) in seen or (dst, src) in seen:  # both directions stored
            continue
        seen.add((src, dst))
        name = bond_vocab[int(b)]
        if name != "NONE":
            rw.AddBond(int(src), int(dst), bond_order[name])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def zinc_x_plus_extra_atom_features(data, atom_vocab: list[str], bond_vocab: list[str]) -> torch.Tensor:
    """[n_atoms, 40] for one ZINC molecule, off the rebuilt mol (no SMILES round-trip, so no
    canonical-SMILES atom reordering). Validated on all 12k ZINC-subset molecules with zero failures."""
    return x_plus_extra_from_mol(zinc_data_to_rdkit_mol(data, atom_vocab, bond_vocab), data.x.shape[0])


# --------------------------------------------------------------------------------------------------
# AQSOL: PyG graphs carry no SMILES -- match each graph to its data_curated.csv row (primary), or
# rebuild an RDKit mol from its own element/bond graph (fallback)
# --------------------------------------------------------------------------------------------------

_WL_ROUNDS = 3


def _aqsol_vocab_index(aqsol_root) -> tuple[dict[str, int], dict[str, int]]:
    from torch_geometric.datasets import AQSOL

    ds = AQSOL(root=str(aqsol_root), split="train")
    return {s: i for i, s in enumerate(ds.atoms())}, {s: i for i, s in enumerate(ds.bonds())}


def _wl_canon(n: int, node_lab: list[int], adj) -> tuple[list[int], tuple]:
    """(canonical node order, final WL labels in that order) -- 3-round WL colour refinement over
    (atom-vocab id, sorted (bond-vocab id, neighbour label) multiset)."""
    import hashlib

    lab = [str(x) for x in node_lab]
    for _ in range(_WL_ROUNDS):
        lab = [
            hashlib.md5(repr((lab[u], tuple(sorted((b, lab[v]) for v, b in adj[u])))).encode()).hexdigest()[:12]
            for u in range(n)
        ]
    order = sorted(range(n), key=lambda u: (lab[u], node_lab[u], len(adj[u])))
    return order, tuple(lab[u] for u in order)


def _adj(n: int, edges: list[tuple[int, int, int]]) -> list[list[tuple[int, int]]]:
    a: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, b in edges:
        a[u].append((v, b))
    return a


def _aqsol_encoding_from_smiles(smi: str, atom_idx: dict, bond_idx: dict):
    """SMILES -> (mol, atom-vocab ids, (u, v, bond-vocab id) both directions) in RDKit atom order,
    or None if unparseable / outside AQSOL's element vocab."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        x = [atom_idx[a.GetSymbol()] for a in m.GetAtoms()]
    except KeyError:
        return None
    if not x:
        return None
    edges: list[tuple[int, int, int]] = []
    for b in m.GetBonds():
        bt = str(b.GetBondType())
        bt = bt if bt in bond_idx else "SINGLE"
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [(u, v, bond_idx[bt]), (v, u, bond_idx[bt])]
    return m, x, edges


def build_aqsol_x_plus_extra_from_smiles(aqsol_root, smiles_csv, splits=("train", "test")) -> dict[str, list]:
    """{split: [FloatTensor[n_i, 40] | None]} aligned to AQSOL(root, split=...) iteration order.
    Each graph is matched to a data_curated.csv row by (n_nodes, n_edges, sorted atom ids, WL
    signature), ties broken by y ~= Solubility; the node permutation is recovered from the two
    canonical orders and verified against the atom ids; then [from_smiles 9 cols | 31 extra cols],
    both computed from that SMILES, are permuted into PyG node order. None = no verified match."""
    import csv
    from collections import defaultdict

    from rdkit import RDLogger
    from torch_geometric.datasets import AQSOL
    from torch_geometric.utils.smiles import from_smiles

    RDLogger.DisableLog("rdApp.*")
    atom_idx, bond_idx = _aqsol_vocab_index(aqsol_root)

    bucket: dict[tuple, list] = defaultdict(list)
    with open(smiles_csv) as f:
        for r in csv.DictReader(f):
            res = _aqsol_encoding_from_smiles(r["SMILES"], atom_idx, bond_idx)
            if res is None:
                continue
            _m, x, edges = res
            order, sig = _wl_canon(len(x), x, _adj(len(x), edges))
            bucket[(len(x), len(edges) // 2, tuple(sorted(x)), sig)].append((r, order))

    out: dict[str, list] = {}
    for sp in splits:
        col: list = []
        for g in AQSOL(root=str(aqsol_root), split=sp):
            x = g.x.view(-1).tolist()
            n = len(x)
            edges = [(u, v, b) for (u, v), b in zip(g.edge_index.t().tolist(), g.edge_attr.view(-1).tolist())]
            order_g, sig_g = _wl_canon(n, x, _adj(n, edges))
            cands = bucket.get((n, len(edges) // 2, tuple(sorted(x)), sig_g), [])
            if len(cands) > 1:
                yv = g.y.item()
                cands = [c for c in cands if abs(float(c[0]["Solubility"]) - yv) < 1e-4] or cands
            picked = None
            for r, order_csv in cands:
                perm = [0] * n
                for i in range(n):
                    perm[order_g[i]] = order_csv[i]  # pyg node -> csv (RDKit) atom
                mol = Chem.MolFromSmiles(r["SMILES"])
                if not all(atom_idx[mol.GetAtomWithIdx(perm[u]).GetSymbol()] == x[u] for u in range(n)):
                    continue
                x40_full = torch.cat([
                    from_smiles(r["SMILES"]).x.float(),
                    extra_atom_features_from_smiles(r["SMILES"], n_atoms_expected=mol.GetNumAtoms()),
                ], dim=-1)
                picked = x40_full[perm].contiguous()
                if picked.shape == (n, N_X_PLUS_EXTRA_FEATURES):
                    break
                picked = None
            col.append(picked)
        out[sp] = col
        print(f"  AQSOL {sp}: {sum(t is not None for t in col)}/{len(col)} graphs matched to a CSV SMILES")
    return out


def load_aqsol_x_plus_extra_from_smiles(aqsol_root, smiles_csv, cache_path, splits=("train", "test")) -> dict[str, list]:
    """Cached build_aqsol_x_plus_extra_from_smiles. Delete cache_path to force a rebuild."""
    from pathlib import Path

    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        if all(s in cached for s in splits):
            return cached
    built = build_aqsol_x_plus_extra_from_smiles(aqsol_root, smiles_csv, splits)
    torch.save(built, cache_path)
    return built


def _two_tier_sanitize(mol) -> str:
    """Full SanitizeMol; on failure (AQSOL skeletons carry no charges/explicit Hs, so hypervalent or
    charged S/N/P and NH-heteroaromatics can violate RDKit's valence model) retry with the valence
    check and kekulization off so ring/aromaticity/degree perception still runs."""
    from rdkit.Chem import SanitizeFlags as SF

    try:
        Chem.SanitizeMol(mol)
        return "strict"
    except Exception:  # noqa: BLE001
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, sanitizeOps=SF.SANITIZE_ALL ^ SF.SANITIZE_KEKULIZE ^ SF.SANITIZE_PROPERTIES)
        Chem.GetSymmSSSR(mol)
        return "relaxed"


def aqsol_data_to_rdkit_mol(data, atom_vocab: list[str], bond_vocab: list[str]):
    """Rebuild + (two-tier) sanitize an RDKit mol from one AQSOL Data object, atom order == node
    order. Every atom is neutral (AQSOL stores no charges / explicit Hs)."""
    bond_order = {
        "SINGLE": Chem.BondType.SINGLE, "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE, "AROMATIC": Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for atom_id in data.x.view(-1).tolist():
        rw.AddAtom(Chem.Atom(atom_vocab[int(atom_id)]))
    seen: set[tuple[int, int]] = set()
    aromatic_atoms: set[int] = set()
    for (src, dst), b in zip(data.edge_index.t().tolist(), data.edge_attr.view(-1).tolist()):
        if (src, dst) in seen or (dst, src) in seen:
            continue
        seen.add((src, dst))
        name = bond_vocab[int(b)]
        if name == "NONE":
            continue
        rw.AddBond(int(src), int(dst), bond_order[name])
        if name == "AROMATIC":
            aromatic_atoms.update((int(src), int(dst)))
    mol = rw.GetMol()
    for idx in aromatic_atoms:
        mol.GetAtomWithIdx(idx).SetIsAromatic(True)
    _two_tier_sanitize(mol)
    return mol


def aqsol_x_plus_extra_rebuild(data, atom_vocab: list[str], bond_vocab: list[str]) -> torch.Tensor | None:
    """Topology-rebuild FALLBACK for an AQSOL graph with no CSV match: [n_atoms, 40] off the rebuilt
    mol (formal_charge neutral, chirality 0), or None if even the relaxed sanitize fails."""
    try:
        return x_plus_extra_from_mol(aqsol_data_to_rdkit_mol(data, atom_vocab, bond_vocab), data.x.shape[0])
    except Exception:  # noqa: BLE001 -- AQSOL metals/salts can genuinely fail RDKit sanitize
        return None
