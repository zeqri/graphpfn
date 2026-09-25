"""
Encode the ZINC-12k subset (torch_geometric.datasets.ZINC(subset=True)) into
sentence-level hidden-state representations using a pretrained MolDeBERTa
encoder -- the ZINC analogue of moldeberta_encode_moleculenet.py.

ZINC has no raw SMILES CSV: train/val/test are three separate PyG datasets
(10000/1000/1000 graphs) whose atoms/bonds are indices into ZINC's own
benchmarking-GNNs vocab. Each graph is converted back to a canonical SMILES
string via RDKit (atom/bond token -> RWMol -> Chem.MolToSmiles; stereochemistry
is not retained, matching what the PyG encoding can represent) before being
fed to the MolDeBERTa tokenizer/encoder -- see zinc_to_smiles() below.

Because ZINC's three splits are separate upstream PyG datasets rather than
index subsets of one shared raw table, this dataset's meta.json necessarily
differs in shape from moleculenet_encode's: split_idx/failed_indices are
per-split LOCAL indices (0..len(split)-1) into that split's own PyG dataset,
not indices into a single shared row order. See the 'index_space_note' field
written into the meta file.

Failed rows (a PyG graph whose reconstructed molecule fails RDKit
sanitization, or that the tokenizer/encoder fails on) are DROPPED, never
stored as zero vectors -- exactly the same policy as
moldeberta_encode_moleculenet.py.

Usage:
    python moldeberta_encode_zinc.py
    python moldeberta_encode_zinc.py --model SaeedLab/MolDeBERTa-base-123M-mlc --batch-size 512

Output (under <out-dir>/zinc/ -- default <out-dir> is <graphpfn>/embeddings/MolDeBERTa/,
shared with moldeberta_encode_moleculenet.py's output root):
    zinc_embeddings_{train,valid,test}.npy  -- float32 [n_split, hidden], failed rows already dropped
    zinc_meta.json                          -- run metadata, split sizes/indices, failed rows
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.datasets import ZINC

RDLogger.DisableLog('rdApp.*')

SCRIPT_DIR = Path(__file__).resolve().parent            # .../graphpfn/paper/embedding
REPO_ROOT = SCRIPT_DIR.parent.parent                     # .../graphpfn
ZINC_ROOT = REPO_ROOT / 'datasets' / 'zinc'
OUT_ROOT = REPO_ROOT / 'embeddings' / 'MolDeBERTa'

# Same default as moldeberta_encode_moleculenet.py: local download of SaeedLab/MolDeBERTa-base-123M-mtr.
DEFAULT_MODEL = str(REPO_ROOT / 'embeddings' / 'checkpoints' / 'MolDeBERTa-base-123M-mtr')

# Exact atom encoding used by the PyG/Benchmarking-GNNs ZINC dataset.
ATOM_TYPES = {
    0:  "C",
    1:  "O",
    2:  "N",
    3:  "F",
    4:  "C H1",
    5:  "S",
    6:  "Cl",
    7:  "O -",
    8:  "N H1 +",
    9:  "Br",
    10: "N H3 +",
    11: "N H2 +",
    12: "N +",
    13: "N -",
    14: "S -",
    15: "I",
    16: "P",
    17: "O H1 +",
    18: "N H1 -",
    19: "O +",
    20: "S +",
    21: "P H1",
    22: "P H2",
    23: "C H2 -",
    24: "P +",
    25: "S H1 +",
    26: "C H1 -",
    27: "P H1 +",
}

BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


def _make_atom(token):
    """Convert a ZINC atom token such as 'N H1 +' into an RDKit atom."""
    parts = token.split()
    atom = Chem.Atom(parts[0])

    for part in parts[1:]:
        if part.startswith("H"):
            atom.SetNumExplicitHs(int(part[1:]))
            atom.SetNoImplicit(True)
        elif part == "+":
            atom.SetFormalCharge(1)
        elif part == "-":
            atom.SetFormalCharge(-1)

    return atom


def zinc_to_smiles(data):
    """Convert one PyG ZINC Data object to canonical SMILES, or None if RDKit
    can't sanitize the reconstructed molecule (dropped by the caller, never
    silently zeroed)."""
    mol = Chem.RWMol()

    for atom_type in data.x.view(-1).tolist():
        mol.AddAtom(_make_atom(ATOM_TYPES[int(atom_type)]))

    # PyG stores both i->j and j->i for each bond; add each bond only once.
    for k in range(data.edge_index.size(1)):
        i = int(data.edge_index[0, k])
        j = int(data.edge_index[1, k])
        if i < j:
            bond_type = int(data.edge_attr[k])
            mol.AddBond(i, j, BOND_TYPES[bond_type])

    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    # ZINC's PyG representation does not retain stereochemistry.
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


@torch.no_grad()
def encode_zinc_split(dataset, tokenizer, model, device, batch_size,
                      max_length, pooling, split_name):
    """Encode dataset[i] for every i, DROPPING failures entirely (never a zero
    row). Returns (kept_local_indices ascending, embeddings[len(kept), hidden],
    failed_local_indices) -- indices are LOCAL to this split's own PyG dataset."""
    hidden = model.config.hidden_size
    n = len(dataset)
    kept_idx = []
    kept_chunks = []
    failed = []

    buffer_smiles, buffer_idx = [], []

    def flush():
        if not buffer_smiles:
            return
        enc = tokenizer(
            buffer_smiles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        ).to(device)
        out = model(**enc).last_hidden_state          # [B, T, H]
        mask = enc['attention_mask'].unsqueeze(-1).to(out.dtype)  # [B, T, 1]
        if pooling == 'cls':
            pooled = out[:, 0]
        else:  # masked mean
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        pooled = pooled.float().cpu().numpy()
        kept_chunks.append(pooled)
        kept_idx.extend(buffer_idx)
        buffer_smiles.clear()
        buffer_idx.clear()

    t0 = time.time()
    for i in range(n):
        try:
            smi = zinc_to_smiles(dataset[i])
        except Exception:
            smi = None
        if smi is None:
            failed.append(i)
            continue
        buffer_smiles.append(smi)
        buffer_idx.append(i)
        if len(buffer_smiles) >= batch_size:
            flush()
        if (i + 1) % 2000 == 0:
            print(f'  [zinc/{split_name}] {i + 1}/{n} processed, {time.time() - t0:.1f}s elapsed')
    flush()

    embeddings = np.concatenate(kept_chunks, axis=0) if kept_chunks else np.zeros((0, hidden), dtype=np.float32)
    print(f'[zinc/{split_name}] done in {time.time() - t0:.1f}s. Dropped (reconstruction/encode failures): '
          f'{len(failed)}/{n}.')
    return kept_idx, embeddings, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='HF repo id or local dir of a MolDeBERTa encoder')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-length', type=int, default=None,
                        help='token cap; default = model config max_position_embeddings')
    parser.add_argument('--pooling', choices=['mean', 'cls'], default='mean')
    parser.add_argument('--fp16', action='store_true', help='run the encoder in float16 (cuda only)')
    parser.add_argument('--out-dir', default=str(OUT_ROOT))
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f'Model:   {args.model}')
    print(f'Device:  {device}')
    print(f'Pooling: {args.pooling}')

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()
    if args.fp16 and device.type == 'cuda':
        model.half()

    max_length = args.max_length or getattr(model.config, 'max_position_embeddings', 128)
    print(f'Hidden size: {model.config.hidden_size}   Max length: {max_length}')

    out_dir = Path(args.out_dir) / 'zinc'
    out_dir.mkdir(parents=True, exist_ok=True)

    split_idx = {}
    failed_indices = {}
    shape_per_split = {}
    print('\n=== zinc ===')
    for pyg_split, out_split in (('train', 'train'), ('val', 'valid'), ('test', 'test')):
        dataset = ZINC(root=str(ZINC_ROOT), subset=True, split=pyg_split)
        print(f'zinc/{out_split}: {len(dataset)} molecules (PyG ZINC(subset=True, split="{pyg_split}"))')

        kept_idx, embeddings, failed = encode_zinc_split(
            dataset, tokenizer, model, device, args.batch_size, max_length, args.pooling, out_split)

        out_emb = out_dir / f'zinc_embeddings_{out_split}.npy'
        np.save(out_emb, embeddings)
        print(f'Saved {out_emb} ({embeddings.shape[0]} rows, {len(failed)} dropped)')

        split_idx[out_split] = kept_idx        # LOCAL indices, failures already excluded
        failed_indices[out_split] = failed     # LOCAL indices, per split -- no shared index space to flatten into
        shape_per_split[out_split] = [len(dataset), model.config.hidden_size]

    out_meta = out_dir / 'zinc_meta.json'
    with open(out_meta, 'w') as f:
        json.dump({
            'shape_per_split': shape_per_split,   # no single shared 'shape' -- see index_space_note
            'model': args.model,
            'encoder': 'DeBERTa-v2 (MolDeBERTa pretrained encoder), AutoModel base outputs',
            'hidden_size': model.config.hidden_size,
            'max_length': max_length,
            'pooling': f'{args.pooling} over final hidden states' + (' (masked)' if args.pooling == 'mean' else ''),
            'source': ('torch_geometric.datasets.ZINC(subset=True) reconstructed to canonical SMILES via RDKit '
                       '(atom/bond token -> RWMol -> Chem.MolToSmiles); stereochemistry not retained '
                       '(isomericSmiles=False)'),
            'zinc_root': str(ZINC_ROOT),
            'split': ('zinc_12k_subset_official -- pre-split at the source: train/val/test are separate PyG '
                      'datasets, not index subsets of one shared raw table'),
            'index_space_note': ('Unlike moldeberta_encode_moleculenet.py\'s meta files, split_idx and '
                                  'failed_indices here are LOCAL indices into each split\'s own PyG '
                                  'ZINC(subset=True, split=...) dataset (0..len(split)-1). There is no shared raw-row '
                                  'index space across train/valid/test for ZINC, so unlike moleculenet/aqsol, '
                                  'failed_indices is keyed by split rather than being one flat dataset-wide list, '
                                  'and there is no unsplit_rows concept (every PyG row belongs to exactly one split '
                                  'by construction).'),
            'split_idx': split_idx,                                  # per split, local, failure-filtered
            'split_sizes': {k: len(v) for k, v in split_idx.items()},
            'failed_indices': failed_indices,                        # per split, local -- see index_space_note
            'failed_per_split': {k: [] for k in split_idx},          # always empty: failures never enter split_idx
            'failed_handling': ('dropped -- molecules whose PyG->SMILES reconstruction or tokenizer/encoder step '
                                 'failed are excluded entirely from split_idx and the saved embeddings arrays; '
                                 'never stored as zero vectors'),
            'embeddings_dir': str(out_dir),
        }, f, indent=2)
    print('Saved metadata to', out_meta)


if __name__ == '__main__':
    main()
