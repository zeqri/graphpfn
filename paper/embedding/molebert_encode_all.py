"""
Encode every molecule in the MoleculeNet SMILES datasets + AQSOL + ZINC into
a graph-level hidden-state representation using the pretrained Mole-BERT GIN
encoder -- the Mole-BERT analogue of
embedding_scripts/moldeberta_encode_moleculenet.py
(+ moldeberta_encode_zinc.py), covering every dataset in ONE script instead
of nine separate ones.

Datasets:
  scaffold-split (raw CSV with a 'smiles' column + dataset/moleculenet/<name>/
  split/scaffold_split.json), parsed via RDKit and converted to Mole-BERT's
  own atom/bond feature encoding with Mole-BERT/loader.py:
  mol_to_graph_data_obj_simple:
      regression:     esol, freesolv, lipo
      classification: bace, bbbp, clintox, sider
  aqsol / zinc: no SMILES in their PyG graphs, so no RDKit parsing is
      possible or needed -- Mole-BERT operates on the graph directly. Each
      dataset's own atom/bond vocab is remapped straight to Mole-BERT's
      feature scheme (atomic number - 1 + chirality=0 for atoms; bond-type
      index + direction=0 for bonds; see build_aqsol_vocab_maps /
      build_zinc_vocab_maps). This is a deterministic 1:1 remap of every
      graph in torch_geometric.datasets.AQSOL(split=...) /
      ZINC(subset=True, split=...) -- unlike MolDeBERTa's aqsol path (which
      has to WL-match graphs back to a CSV for SMILES text), there is no
      "unmatched" case here, only the rare per-graph conversion/encode
      failure.

Row order / index space:
  - For the scaffold-split datasets, split_idx values are indices into the
    source_csv row order (0-indexed, header excluded) -- same convention as
    moldeberta_encode_moleculenet.py.
  - For aqsol/zinc, split_idx values are LOCAL indices into that split's own
    PyG dataset (0..len(split)-1) -- same convention as
    moldeberta_encode_zinc.py's meta (see each dataset's meta.json
    "index_space_note").

Failed rows are DROPPED, never stored as zero vectors, for EVERY dataset.
A failed row (unparseable/empty SMILES for the scaffold-split sets, or a
graph whose vocab remap/GIN forward pass raises for aqsol/zinc) is removed
from split_idx before the per-split embeddings are sliced out, so it never
appears in split_idx or in any saved embeddings array -- no split file ever
contains a placeholder zero row. meta["failed_indices"] records the dropped
rows for provenance only; meta["failed_per_split"] is always empty by
construction.

Usage:
    python molebert_encode_all.py
    python molebert_encode_all.py --datasets esol lipo aqsol zinc
    python molebert_encode_all.py --device cuda --batch-size 512

Output (per dataset, under <out-dir>/<name>/ -- default <out-dir> is
<graphpfn>/embeddings/Molbert/, where the evaluation scripts read it):
    <name>_embeddings_{train,valid,test}.npy  -- float32 [n_split, 300],
                                                  failed rows already dropped
    <name>_meta.json                          -- run metadata, split sizes/indices, failed rows

The output directory alone identifies the encoder (Mole-BERT) -- file names
carry no encoder tag, matching embeddings/moldeberta/'s layout exactly
(same <name>_embeddings_{train,valid,test}.npy / <name>_meta.json scheme).
"""
import argparse
import json
import os
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data

# model.py imports torch_scatter.scatter_add unconditionally, but it is only
# ever called from GCNConv, which we never instantiate (gnn_type='gin' only
# uses GINConv). Stub the module out instead of fighting torch_scatter's
# from-source build (it needs torch visible at build time, which pip's build
# isolation hides).
if 'torch_scatter' not in sys.modules:
    _stub = types.ModuleType('torch_scatter')

    def scatter_add(src, index, dim=0, out=None, dim_size=None):
        if dim_size is None:
            dim_size = int(index.max()) + 1
        if out is None:
            size = list(src.size())
            size[dim] = dim_size
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.index_add_(dim, index, src)

    _stub.scatter_add = scatter_add
    sys.modules['torch_scatter'] = _stub

SCRIPT_DIR = Path(__file__).resolve().parent            # .../graphpfn/paper/embedding
REPO_ROOT = SCRIPT_DIR.parent.parent                     # .../graphpfn
CHECKPOINTS_ROOT = REPO_ROOT / 'embeddings' / 'checkpoints'
# Official Mole-BERT code (https://github.com/junxia97/Mole-BERT: loader.py, model.py);
# defaults to <graphpfn>/embeddings/checkpoints/Mole-BERT, override with $MOLEBERT_DIR.
MOLEBERT_DIR = Path(os.environ.get('MOLEBERT_DIR', CHECKPOINTS_ROOT / 'Mole-BERT'))
sys.path.insert(0, str(MOLEBERT_DIR))

from loader import mol_to_graph_data_obj_simple  # noqa: E402
from model import GNN  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit import RDLogger  # noqa: E402

from split_source import check_split_source, read_csv_smiles  # noqa: E402

RDLogger.DisableLog('rdApp.*')

# Pretrained Mole-BERT GIN weights, shipped inside the Mole-BERT repo as model_gin/Mole-BERT.pth;
# override with $MOLEBERT_CHECKPOINT.
CKPT = Path(os.environ.get('MOLEBERT_CHECKPOINT', MOLEBERT_DIR / 'model_gin' / 'Mole-BERT.pth'))
MOLECULENET_ROOT = REPO_ROOT / 'datasets' / 'moleculenet'
AQSOL_ROOT = REPO_ROOT / 'datasets' / 'aqsol'
ZINC_ROOT = REPO_ROOT / 'datasets' / 'zinc'
OUT_ROOT = REPO_ROOT / 'embeddings' / 'Molbert'
EMB_DIM = 300
NUM_LAYER = 5

# Mole-BERT's own bond-type vocabulary (Mole-BERT/loader.py allowable_features).
MOLEBERT_BOND_TYPE = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}

SCAFFOLD_DATASETS = {
    # regression
    'esol': {
        'csv': MOLECULENET_ROOT / 'esol' / 'raw' / 'delaney-processed.csv',
        'split_json': MOLECULENET_ROOT / 'esol' / 'split' / 'scaffold_split.json',
    },
    'freesolv': {
        'csv': MOLECULENET_ROOT / 'freesolv' / 'raw' / 'SAMPL.csv',
        'split_json': MOLECULENET_ROOT / 'freesolv' / 'split' / 'scaffold_split.json',
    },
    'lipo': {
        'csv': MOLECULENET_ROOT / 'lipo' / 'raw' / 'Lipophilicity.csv',
        'split_json': MOLECULENET_ROOT / 'lipo' / 'split' / 'scaffold_split.json',
    },
    # classification
    'bace': {
        'csv': MOLECULENET_ROOT / 'bace' / 'raw' / 'bace.csv',
        'split_json': MOLECULENET_ROOT / 'bace' / 'split' / 'scaffold_split.json',
    },
    'bbbp': {
        'csv': MOLECULENET_ROOT / 'bbbp' / 'raw' / 'BBBP.csv',
        'split_json': MOLECULENET_ROOT / 'bbbp' / 'split' / 'scaffold_split.json',
    },
    'clintox': {
        'csv': MOLECULENET_ROOT / 'clintox' / 'raw' / 'clintox.csv',
        'split_json': MOLECULENET_ROOT / 'clintox' / 'split' / 'scaffold_split.json',
    },
    'sider': {
        'csv': MOLECULENET_ROOT / 'sider' / 'raw' / 'sider.csv',
        'split_json': MOLECULENET_ROOT / 'sider' / 'split' / 'scaffold_split.json',
    },
}
GRAPH_NATIVE_DATASETS = ['aqsol', 'zinc']   # no SMILES/CSV -- own vocab remapped directly
ALL_DATASETS = list(SCAFFOLD_DATASETS.keys()) + GRAPH_NATIVE_DATASETS


def load_gnn(device):
    model = GNN(num_layer=NUM_LAYER, emb_dim=EMB_DIM, JK='last', drop_ratio=0, gnn_type='gin')
    model.load_state_dict(torch.load(CKPT, map_location='cpu'))
    model.to(device)
    model.eval()
    return model


def _pool_batch(model, buffer_data, device):
    """[len(buffer_data), EMB_DIM] mean-pooled graph representations."""
    batch = Batch.from_data_list(buffer_data).to(device)
    with torch.no_grad():
        node_repr = model(batch.x, batch.edge_index, batch.edge_attr)
        graph_repr = torch.zeros(len(buffer_data), node_repr.size(1), device=device)
        graph_repr.index_add_(0, batch.batch, node_repr)
        counts = torch.bincount(batch.batch, minlength=len(buffer_data)).clamp(min=1).unsqueeze(1)
        graph_repr = graph_repr / counts
    return graph_repr.cpu().numpy()


@torch.no_grad()
def encode_smiles(smiles_list, model, device, batch_size, name):
    """Encode smiles_list[i] for every i, DROPPING failures entirely (never a
    zero row). Returns (kept_csv_indices ascending, embeddings[len(kept), 300],
    failed_csv_indices)."""
    kept_idx, kept_chunks, failed = [], [], []
    buffer_data, buffer_idx = [], []

    def flush():
        if not buffer_data:
            return
        kept_chunks.append(_pool_batch(model, buffer_data, device))
        kept_idx.extend(buffer_idx)
        buffer_data.clear()
        buffer_idx.clear()

    t0 = time.time()
    n = len(smiles_list)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed.append(i)
            continue
        try:
            data = mol_to_graph_data_obj_simple(mol)
        except Exception:
            failed.append(i)
            continue
        buffer_data.append(data)
        buffer_idx.append(i)
        if len(buffer_data) >= batch_size:
            flush()
        if (i + 1) % 5000 == 0:
            print(f'  [{name}] {i + 1}/{n} processed, {time.time() - t0:.1f}s elapsed')
    flush()

    embeddings = np.concatenate(kept_chunks, axis=0) if kept_chunks else np.zeros((0, EMB_DIM), dtype=np.float32)
    print(f'[{name}] done in {time.time() - t0:.1f}s. Dropped (unparseable/empty SMILES): {len(failed)}/{n}.')
    return kept_idx, embeddings, failed


# --------------------------------------------------------------------------
# AQSOL / ZINC: no SMILES in their PyG graphs -- remap each dataset's own
# atom/bond vocab directly to Mole-BERT's [atomic_num_idx, chirality_idx] /
# [bond_type_idx, bond_dir_idx] feature scheme (same approach as the
# standalone molebert_encode_aqsol.py / molebert_encode_zinc.py this script
# replaces), then encode via the same GIN forward pass.
# --------------------------------------------------------------------------
def build_aqsol_vocab_maps():
    from torch_geometric.datasets import AQSOL

    probe = AQSOL(root=str(AQSOL_ROOT), split='train')
    atom_words = probe.atoms()
    bond_words = probe.bonds()

    pt = Chem.GetPeriodicTable()
    atom_to_idx = [pt.GetAtomicNumber(sym) - 1 for sym in atom_words]
    bond_to_idx = [MOLEBERT_BOND_TYPE.get(word, 0) for word in bond_words]

    return (torch.tensor(atom_to_idx, dtype=torch.long),
            torch.tensor(bond_to_idx, dtype=torch.long))


class _DictionaryStub:
    """Stand-in for the unpickling class referenced by ZINC's raw
    atom_dict.pickle / bond_dict.pickle (pickled from '__main__.Dictionary'
    by the original benchmarking-gnns data prep script)."""


def build_zinc_vocab_maps():
    import __main__
    __main__.Dictionary = _DictionaryStub

    with open(ZINC_ROOT / 'raw' / 'atom_dict.pickle', 'rb') as f:
        atom_dict = pickle.load(f)
    with open(ZINC_ROOT / 'raw' / 'bond_dict.pickle', 'rb') as f:
        bond_dict = pickle.load(f)

    pt = Chem.GetPeriodicTable()
    atom_to_idx = [pt.GetAtomicNumber(word.split()[0]) - 1 for word in atom_dict.idx2word]
    bond_to_idx = [MOLEBERT_BOND_TYPE.get(word, 0) for word in bond_dict.idx2word]

    return (torch.tensor(atom_to_idx, dtype=torch.long),
            torch.tensor(bond_to_idx, dtype=torch.long))


def convert_graph_native(data, atom_lookup, bond_lookup):
    """Remap a raw AQSOL/ZINC Data object (own vocab indices) to Mole-BERT's
    [atomic_num_idx, chirality_idx] / [bond_type_idx, bond_dir_idx] scheme."""
    atomic_num_idx = atom_lookup[data.x.view(-1)]
    chirality_idx = torch.zeros_like(atomic_num_idx)
    x = torch.stack([atomic_num_idx, chirality_idx], dim=1)

    bond_type_idx = bond_lookup[data.edge_attr]
    bond_dir_idx = torch.zeros_like(bond_type_idx)
    edge_attr = torch.stack([bond_type_idx, bond_dir_idx], dim=1)

    return Data(x=x, edge_index=data.edge_index, edge_attr=edge_attr)


@torch.no_grad()
def encode_graph_native_split(dataset, atom_lookup, bond_lookup, model, device, batch_size, label):
    """Encode dataset[i] for every i, DROPPING failures entirely (never a zero
    row). Returns (kept_local_indices ascending, embeddings[len(kept), 300],
    failed_local_indices) -- indices are LOCAL to this split's own PyG dataset."""
    kept_idx, kept_chunks, failed = [], [], []
    buffer_data, buffer_idx = [], []

    def flush():
        if not buffer_data:
            return
        kept_chunks.append(_pool_batch(model, buffer_data, device))
        kept_idx.extend(buffer_idx)
        buffer_data.clear()
        buffer_idx.clear()

    t0 = time.time()
    n = len(dataset)
    for i, data in enumerate(dataset):
        try:
            mb_data = convert_graph_native(data, atom_lookup, bond_lookup)
        except Exception:
            failed.append(i)
            continue
        buffer_data.append(mb_data)
        buffer_idx.append(i)
        if len(buffer_data) >= batch_size:
            flush()
        if (i + 1) % 5000 == 0:
            print(f'  [{label}] {i + 1}/{n} processed, {time.time() - t0:.1f}s elapsed')
    flush()

    embeddings = np.concatenate(kept_chunks, axis=0) if kept_chunks else np.zeros((0, EMB_DIM), dtype=np.float32)
    print(f'[{label}] done in {time.time() - t0:.1f}s. Dropped (vocab-remap/encode failures): {len(failed)}/{n}.')
    return kept_idx, embeddings, failed


def process_scaffold_dataset(name, cfg, model, device, batch_size, out_root):
    print(f'\n=== {name} ===')
    csv_sha256 = check_split_source(cfg['csv'], cfg['split_json'])
    # Same row parser as the split itself (see split_source.py).
    smiles_list, smiles_col = read_csv_smiles(cfg['csv'], name)
    n_total = len(smiles_list)
    print(f'{name}: {n_total} molecules (smiles column: "{smiles_col}")')

    with open(cfg['split_json']) as f:
        raw_split_idx = json.load(f)

    kept_idx, embeddings, failed = encode_smiles(
        smiles_list, model, device, batch_size, name)

    failed_set = set(failed)
    csv_idx_to_row = {c: r for r, c in enumerate(kept_idx)}

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    new_split_idx = {}
    for split_name, idx_list in raw_split_idx.items():
        filtered = [c for c in idx_list if c not in failed_set]
        split_embeddings = embeddings[[csv_idx_to_row[c] for c in filtered]]
        out_emb = out_dir / f'{name}_embeddings_{split_name}.npy'
        np.save(out_emb, split_embeddings)
        print(f'Saved {out_emb} ({split_embeddings.shape[0]} rows, {len(idx_list) - len(filtered)} dropped)')
        new_split_idx[split_name] = filtered

    unsplit_rows = sorted(set(range(n_total)) - set(sum(raw_split_idx.values(), [])))

    out_meta = out_dir / f'{name}_meta.json'
    with open(out_meta, 'w') as f:
        json.dump({
            'shape': [n_total, EMB_DIM],
            'checkpoint': str(CKPT),
            'model': f'GIN, num_layer={NUM_LAYER}, emb_dim={EMB_DIM}, JK=last (Mole-BERT pretrained encoder)',
            'pooling': 'mean over node representations',
            'source_csv': str(cfg['csv']),
            'source_csv_sha256': csv_sha256,
            'smiles_column': smiles_col,
            'split': 'scaffold',
            'split_idx': new_split_idx,     # indices into source_csv row order; failed rows already excluded
            'split_sizes': {k: len(v) for k, v in new_split_idx.items()},
            'unsplit_rows': unsplit_rows,   # rows never assigned to any split by the split definition itself
            'failed_indices': failed,       # dataset-wide encode failures -- DROPPED, not stored as zero vectors
            'failed_per_split': {k: [] for k in new_split_idx},  # always empty: failures never enter split_idx
            'failed_handling': ('dropped -- unparseable/empty SMILES are excluded entirely from split_idx and '
                                 'the saved embeddings arrays; they are never stored as zero vectors'),
            'embeddings_dir': str(out_dir),
        }, f, indent=2)
    print('Saved metadata to', out_meta)


def process_aqsol(model, device, batch_size, out_root):
    from torch_geometric.datasets import AQSOL

    print('\n=== aqsol ===')
    atom_lookup, bond_lookup = build_aqsol_vocab_maps()

    out_dir = out_root / 'aqsol'
    out_dir.mkdir(parents=True, exist_ok=True)

    split_idx, failed_indices, shape_per_split = {}, {}, {}
    for pyg_split, out_split in (('train', 'train'), ('val', 'valid'), ('test', 'test')):
        dataset = AQSOL(root=str(AQSOL_ROOT), split=pyg_split)
        print(f'aqsol/{out_split}: {len(dataset)} molecules (torch_geometric.datasets.AQSOL(split="{pyg_split}"))')

        kept_idx, embeddings, failed = encode_graph_native_split(
            dataset, atom_lookup, bond_lookup, model, device, batch_size, f'aqsol/{out_split}')

        out_emb = out_dir / f'aqsol_embeddings_{out_split}.npy'
        np.save(out_emb, embeddings)
        print(f'Saved {out_emb} ({embeddings.shape[0]} rows, {len(failed)} dropped)')

        split_idx[out_split] = kept_idx        # LOCAL indices, failures already excluded
        failed_indices[out_split] = failed     # LOCAL indices, per split -- no shared index space to flatten into
        shape_per_split[out_split] = [len(dataset), EMB_DIM]

    out_meta = out_dir / 'aqsol_meta.json'
    with open(out_meta, 'w') as f:
        json.dump({
            'shape_per_split': shape_per_split,   # no single shared 'shape' -- see index_space_note
            'checkpoint': str(CKPT),
            'model': f'GIN, num_layer={NUM_LAYER}, emb_dim={EMB_DIM}, JK=last (Mole-BERT pretrained encoder)',
            'pooling': 'mean over node representations',
            'source': ('torch_geometric.datasets.AQSOL, atom/bond vocab remapped directly to Mole-BERT feature '
                       'scheme -- no SMILES/CSV needed (unlike the MolDeBERTa aqsol path, which must WL-match '
                       'graphs to data_curated.csv for SMILES text); every graph in AQSOL(split=...) is a '
                       'deterministic 1:1 remap, so there is no "unmatched" case here, only rare per-graph '
                       'conversion/encode failures'),
            'split': ('aqsol_official -- exact torch_geometric.datasets.AQSOL benchmark split (7,836 train / '
                      '998 valid / 999 test per PyG docs), read directly, not recovered via CSV matching'),
            'index_space_note': ('Unlike the scaffold-split datasets in this script, split_idx here is LOCAL '
                                  'indices into each split\'s own AQSOL(split=...) dataset (0..len(split)-1). '
                                  'There is no shared raw-row index space across train/valid/test, so '
                                  'failed_indices is keyed by split rather than one flat dataset-wide list, and '
                                  'there is no unsplit_rows concept (every AQSOL graph belongs to exactly one '
                                  'split by construction).'),
            'split_idx': split_idx,                                  # per split, local, failure-filtered
            'split_sizes': {k: len(v) for k, v in split_idx.items()},
            'failed_indices': failed_indices,                        # per split, local -- see index_space_note
            'failed_per_split': {k: [] for k in split_idx},          # always empty: failures never enter split_idx
            'failed_handling': ('dropped -- graphs whose vocab remap or GIN forward pass failed are excluded '
                                 'entirely from split_idx and the saved embeddings arrays; never stored as zero '
                                 'vectors'),
            'embeddings_dir': str(out_dir),
        }, f, indent=2)
    print('Saved metadata to', out_meta)


def process_zinc(model, device, batch_size, out_root):
    from torch_geometric.datasets import ZINC

    print('\n=== zinc ===')
    atom_lookup, bond_lookup = build_zinc_vocab_maps()

    out_dir = out_root / 'zinc'
    out_dir.mkdir(parents=True, exist_ok=True)

    split_idx, failed_indices, shape_per_split = {}, {}, {}
    for pyg_split, out_split in (('train', 'train'), ('val', 'valid'), ('test', 'test')):
        dataset = ZINC(root=str(ZINC_ROOT), subset=True, split=pyg_split)
        print(f'zinc/{out_split}: {len(dataset)} molecules (PyG ZINC(subset=True, split="{pyg_split}"))')

        kept_idx, embeddings, failed = encode_graph_native_split(
            dataset, atom_lookup, bond_lookup, model, device, batch_size, f'zinc/{out_split}')

        out_emb = out_dir / f'zinc_embeddings_{out_split}.npy'
        np.save(out_emb, embeddings)
        print(f'Saved {out_emb} ({embeddings.shape[0]} rows, {len(failed)} dropped)')

        split_idx[out_split] = kept_idx
        failed_indices[out_split] = failed
        shape_per_split[out_split] = [len(dataset), EMB_DIM]

    out_meta = out_dir / 'zinc_meta.json'
    with open(out_meta, 'w') as f:
        json.dump({
            'shape_per_split': shape_per_split,
            'checkpoint': str(CKPT),
            'model': f'GIN, num_layer={NUM_LAYER}, emb_dim={EMB_DIM}, JK=last (Mole-BERT pretrained encoder)',
            'pooling': 'mean over node representations',
            'source': ('torch_geometric.datasets.ZINC(subset=True), atom/bond vocab remapped directly to '
                       'Mole-BERT feature scheme (atom vocab -> element symbol -> atomic number - 1; chirality '
                       'always 0; bond vocab -> Mole-BERT bond-type index; bond direction always 0) -- no '
                       'SMILES/CSV needed'),
            'zinc_root': str(ZINC_ROOT),
            'split': ('zinc_12k_subset_official -- pre-split at the source: train/val/test are separate PyG '
                      'datasets, not index subsets of one shared raw table'),
            'index_space_note': ('split_idx and failed_indices here are LOCAL indices into each split\'s own PyG '
                                  'ZINC(subset=True, split=...) dataset (0..len(split)-1). There is no shared '
                                  'raw-row index space across train/valid/test for ZINC, so failed_indices is '
                                  'keyed by split rather than being one flat dataset-wide list, and there is no '
                                  'unsplit_rows concept (every PyG row belongs to exactly one split by '
                                  'construction).'),
            'split_idx': split_idx,
            'split_sizes': {k: len(v) for k, v in split_idx.items()},
            'failed_indices': failed_indices,
            'failed_per_split': {k: [] for k in split_idx},
            'failed_handling': ('dropped -- graphs whose vocab remap or GIN forward pass failed are excluded '
                                 'entirely from split_idx and the saved embeddings arrays; never stored as zero '
                                 'vectors'),
            'embeddings_dir': str(out_dir),
        }, f, indent=2)
    print('Saved metadata to', out_meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument('--out-dir', default=str(OUT_ROOT))
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f'Using device: {device}')
    print(f'Checkpoint: {CKPT}')
    out_root = Path(args.out_dir)

    model = load_gnn(device)

    for name in args.datasets:
        if name in SCAFFOLD_DATASETS:
            process_scaffold_dataset(name, SCAFFOLD_DATASETS[name], model, device, args.batch_size, out_root)
        elif name == 'aqsol':
            process_aqsol(model, device, args.batch_size, out_root)
        elif name == 'zinc':
            process_zinc(model, device, args.batch_size, out_root)


if __name__ == '__main__':
    main()
