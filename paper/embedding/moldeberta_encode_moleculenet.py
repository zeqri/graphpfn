"""
Encode every molecule in the MoleculeNet SMILES datasets (+ AQSOL) into a
sentence-level hidden-state representation using a pretrained MolDeBERTa
encoder (DeBERTa-v2 + byte-level BPE) -- the MolDeBERTa analogue of
molebert_encode_moleculenet.py.

Datasets:
  scaffold-split (raw CSV with a 'smiles' column + dataset/moleculenet/<name>/
  split/scaffold_split.json):
      regression:     esol, freesolv, lipo
      classification: bace, bbbp, clintox, sider
  aqsol (dataset/aqsol/data_curated.csv, 'SMILES' column): has no scaffold
      split file, and its PyG graphs carry no SMILES at all. Every
      torch_geometric.datasets.AQSOL(split)[i] graph, for every split and
      every i, is matched back to its CSV row via a Weisfeiler-Lehman
      canonical signature over AQSOL's own atom/bond vocab (same idea as
      aqsol_smiles_9dim.py's matcher) -- see build_aqsol_matches(). This
      reproduces the OFFICIAL AQSOL benchmark split exactly (9,833 graphs:
      7,836 train / 998 valid / 999 test per torch_geometric's own docs),
      including any duplicate compound the official split itself contains,
      within or across splits -- that split predates AqSolDB's curation into
      one row per compound, so matching it exactly means not deduping or
      enforcing train/valid/test disjointness. Coverage is 9,831/9,833: 2
      graphs in `valid` (local indices 250, 251) have no WL match to any CSV
      row at all and are dropped (see failed_handling in aqsol's meta.json).
      aqsol's embeddings row i is graph i of AQSOL(split) directly (see
      meta["graph_index"]), NOT a data_curated.csv row index -- unlike every
      other dataset in this script.

SMILES are (optionally) RDKit-canonicalised, then fed straight to the
MolDeBERTa tokenizer/encoder. The molecule embedding is a masked mean (or
CLS) pool over the final transformer hidden states. Labels are not touched
-- only the smiles column is read, so the same embeddings serve every task
column in multi-task sets (sider, clintox).

Row order within each dataset's embedding arrays matches its raw CSV row
order (0-indexed, header excluded); split_idx values are indices into that
same row order.

Failed rows are DROPPED, never stored as zero vectors, for EVERY dataset --
the scaffold-split MoleculeNet sets (bace/bbbp/clintox/esol/freesolv/lipo/
sider) exactly like aqsol. A failed row (unparseable / empty SMILES, or --
for aqsol only -- a CSV row that couldn't be matched to any split) is
removed from split_idx before the per-split embeddings are sliced out, so it
never appears in split_idx or in any saved embeddings array; no split file
ever contains a placeholder zero row. meta["failed_indices"] records the
dropped rows for provenance only -- meta["failed_per_split"] is always empty
by construction, since failures are excluded from split_idx up front rather
than left in for a downstream consumer to filter out.

Usage:
    python moldeberta_encode_moleculenet.py
    python moldeberta_encode_moleculenet.py --model SaeedLab/MolDeBERTa-base-123M-mlc
    python moldeberta_encode_moleculenet.py --datasets esol lipo aqsol --batch-size 512

--model defaults to <graphpfn>/embeddings/checkpoints/MolDeBERTa-base-123M-mtr
(see the README for the download command). It also accepts any entry from
https://huggingface.co/collections/SaeedLab/moldeberta or another local
directory. The tokenizer ships inside each model repo, so one path is enough.

Output (per dataset, under <out-dir>/<name>/ -- default <out-dir> is
<graphpfn>/embeddings/MolDeBERTa/, where the evaluation scripts read it):
    <name>_embeddings_{train,valid,test}.npy  -- float32 [n_split, hidden],
                                                  failed rows already dropped
    <name>_meta.json                          -- run metadata, split sizes/indices, failed rows

The output directory alone identifies the encoder (MolDeBERTa) -- file names
carry no encoder tag, so the same layout/schema can be reused verbatim for
other encoders (e.g. Mole-BERT under embeddings/molbert/<name>/).
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from rdkit import Chem
from rdkit import RDLogger

from split_source import check_split_source, read_csv_smiles

RDLogger.DisableLog('rdApp.*')

SCRIPT_DIR = Path(__file__).resolve().parent            # .../graphpfn/paper/embedding
REPO_ROOT = SCRIPT_DIR.parent.parent                     # .../graphpfn
MOLECULENET_ROOT = REPO_ROOT / 'datasets' / 'moleculenet'
AQSOL_ROOT = REPO_ROOT / 'datasets' / 'aqsol'
OUT_ROOT = REPO_ROOT / 'embeddings' / 'MolDeBERTa'

# Local download of SaeedLab/MolDeBERTa-base-123M-mtr (the collection's recommended checkpoint).
DEFAULT_MODEL = str(REPO_ROOT / 'embeddings' / 'checkpoints' / 'MolDeBERTa-base-123M-mtr')

DATASETS = {
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
    # regression, no scaffold_split.json -- split recovered from PyG's AQSOL benchmark split
    'aqsol': {
        'csv': AQSOL_ROOT / 'data_curated.csv',
        'split_json': None,
    },
}


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_smiles_column(df):
    for col in df.columns:
        if col.strip().lower() == 'smiles':
            return col
    raise KeyError(f'no "smiles" column in CSV; columns are {list(df.columns)}')


# --------------------------------------------------------------------------
# AQSOL has no SMILES in its PyG graphs, so its official train/valid/test
# split can't be read off a split_idx json the way scaffold datasets can.
# Instead, match every AQSOL(split)[i] graph back to its data_curated.csv row
# by a 3-round Weisfeiler-Lehman canonical signature over AQSOL's own
# atom/bond vocab encoding (same protocol as
# .../dev_prior_3/evluation/aqsol_smiles_9dim.py's matcher), verify the atom
# mapping, and take the CSV row index. Unmatched/ambiguous rows are left out
# of every split (they end up in unsplit_rows, not silently mis-assigned).
# --------------------------------------------------------------------------
def _wl_adj(n, edges):
    a = [[] for _ in range(n)]
    for u, v, b in edges:
        a[u].append((v, b))
    return a


def _wl_canon(n, node_lab, adj, rounds=3):
    lab = [str(x) for x in node_lab]
    for _ in range(rounds):
        lab = [
            hashlib.md5(
                repr((lab[u], tuple(sorted((b, lab[v]) for v, b in adj[u])))).encode()
            ).hexdigest()[:12]
            for u in range(n)
        ]
    order = sorted(range(n), key=lambda u: (lab[u], node_lab[u], len(adj[u])))
    return order, tuple(lab[u] for u in order)


def _bng_from_smiles(smi, atom_idx, bond_idx):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        x = [atom_idx[a.GetSymbol()] for a in m.GetAtoms()]
    except KeyError:
        return None
    if not x:
        return None
    edges = []
    for b in m.GetBonds():
        bt = str(b.GetBondType())
        bt = bt if bt in bond_idx else 'SINGLE'
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [(u, v, bond_idx[bt]), (v, u, bond_idx[bt])]
    return x, edges


def build_aqsol_matches(df, smiles_col):
    """Match every torch_geometric.datasets.AQSOL(split)[i] graph, for every i
    and every split, to its data_curated.csv row -- reproducing the OFFICIAL
    AQSOL benchmark split exactly as PyG defines it (9,833 graphs total:
    7,836 train / 998 valid / 999 test, per torch_geometric's own AQSOL docs,
    which already filters out no-edge graphs internally). We deliberately do
    NOT dedupe repeated compounds within a split or enforce train/valid/test
    disjointness here: the official split itself doesn't do either (it
    predates AqSolDB's curation into one row per compound), so matching it
    exactly means preserving whatever duplication it contains.

    Returns (matches, unmatched):
      matches[out_split]   = [(graph_idx, csv_row), ...] in AQSOL(split) order,
                              one entry per graph that WL-matched a CSV row
      unmatched[out_split] = [graph_idx, ...] graphs with no WL match at all
    """
    from torch_geometric.datasets import AQSOL

    probe = AQSOL(root=str(AQSOL_ROOT), split='train')
    atom_idx = {s: i for i, s in enumerate(probe.atoms())}
    bond_idx = {s: i for i, s in enumerate(probe.bonds())}

    bucket = defaultdict(list)
    for i, smi in enumerate(df[smiles_col].tolist()):
        res = _bng_from_smiles(smi, atom_idx, bond_idx)
        if res is None:
            continue
        x, edges = res
        n = len(x)
        order, sig = _wl_canon(n, x, _wl_adj(n, edges))
        bucket[(n, len(edges) // 2, tuple(sorted(x)), sig)].append((i, order))

    matches = {}
    unmatched = {}
    for pyg_split, out_split in (('train', 'train'), ('val', 'valid'), ('test', 'test')):
        ds = AQSOL(root=str(AQSOL_ROOT), split=pyg_split)
        m = []
        um = []
        for gi, g in enumerate(ds):
            x = g.x.view(-1).tolist()
            n = len(x)
            e = g.edge_index.t().tolist()
            ea = g.edge_attr.view(-1).tolist()
            edges = [(u, v, b) for (u, v), b in zip(e, ea)]
            order_g, sig_g = _wl_canon(n, x, _wl_adj(n, edges))
            cands = bucket.get((n, len(edges) // 2, tuple(sorted(x)), sig_g), [])
            if len(cands) > 1:
                yv = g.y.item()
                cands = sorted(cands, key=lambda c: abs(float(df.iloc[c[0]]['Solubility']) - yv))
            picked = None
            for csv_i, order_csv in cands:
                perm = [0] * n
                for k in range(n):
                    perm[order_g[k]] = order_csv[k]
                mol = Chem.MolFromSmiles(df.iloc[csv_i][smiles_col])
                csv_x = [atom_idx[a.GetSymbol()] for a in mol.GetAtoms()]
                if all(csv_x[perm[u]] == x[u] for u in range(n)):
                    picked = csv_i
                    break
            if picked is not None:
                m.append((gi, picked))
            else:
                um.append(gi)
        matches[out_split] = m
        unmatched[out_split] = um
        print(f'  [aqsol] {out_split}: matched {len(m)}/{len(ds)} AQSOL graphs to CSV rows '
              f'({len(um)} unmatched)')

    return matches, unmatched


@torch.no_grad()
def encode_smiles(smiles_list, tokenizer, model, device, batch_size,
                  max_length, pooling, canonicalize, name):
    """Encode smiles_list[i] for every i, DROPPING failures entirely (never a
    zero row) -- applies identically whether `name` is a scaffold-split
    MoleculeNet set or aqsol. Returns (kept_csv_indices ascending,
    embeddings[len(kept), hidden], failed_csv_indices); embeddings[r] is the
    encoding of smiles_list[kept_csv_indices[r]]."""
    hidden = model.config.hidden_size
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
    n = len(smiles_list)
    for i, smiles in enumerate(smiles_list):
        if not isinstance(smiles, str) or not smiles:
            failed.append(i)
            continue
        s = smiles
        if canonicalize:
            s = canonical_smiles(smiles)
            if s is None:
                failed.append(i)
                continue
        buffer_smiles.append(s)
        buffer_idx.append(i)
        if len(buffer_smiles) >= batch_size:
            flush()
        if (i + 1) % 5000 == 0:
            print(f'  [{name}] {i + 1}/{n} processed, {time.time() - t0:.1f}s elapsed')
    flush()

    embeddings = np.concatenate(kept_chunks, axis=0) if kept_chunks else np.zeros((0, hidden), dtype=np.float32)
    print(f'[{name}] done in {time.time() - t0:.1f}s. Dropped (unparseable/empty SMILES): {len(failed)}/{n}.')
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
    parser.add_argument('--no-canonicalize', action='store_true',
                        help='feed raw SMILES to the tokenizer without RDKit canonicalisation')
    parser.add_argument('--fp16', action='store_true', help='run the encoder in float16 (cuda only)')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()))
    parser.add_argument('--out-dir', default=str(OUT_ROOT))
    args = parser.parse_args()

    device = torch.device(args.device)
    canonicalize = not args.no_canonicalize
    print(f'Model:   {args.model}')
    print(f'Device:  {device}')
    print(f'Pooling: {args.pooling}   Canonicalize: {canonicalize}')

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()
    if args.fp16 and device.type == 'cuda':
        model.half()

    max_length = args.max_length or getattr(model.config, 'max_position_embeddings', 128)
    print(f'Hidden size: {model.config.hidden_size}   Max length: {max_length}')

    out_root = Path(args.out_dir)

    for name in args.datasets:
        cfg = DATASETS[name]
        print(f'\n=== {name} ===')
        if cfg['split_json'] is None:
            df = pd.read_csv(cfg['csv'])
            smiles_col = get_smiles_column(df)
            n_total = len(df)
        else:
            # Scaffold-split sets: same row parser as the split itself (see split_source.py).
            smiles_list, smiles_col = read_csv_smiles(cfg['csv'], name)
            n_total = len(smiles_list)
        print(f'{name}: {n_total} molecules (smiles column: "{smiles_col}")')

        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if cfg['split_json'] is None:
            # aqsol: dedicated path -- one embedding row per AQSOL(split) graph
            # (index space is per-split PyG graph position, NOT a CSV row; see
            # module docstring). Encode only the unique CSV rows any graph
            # actually matched to, then broadcast to every matching graph
            # position (duplicates preserved, to mirror the official split).
            matches, unmatched = build_aqsol_matches(df, smiles_col)

            unique_csv_rows = sorted({c for m in matches.values() for _, c in m})
            kept_local, unique_embeddings, failed_local = encode_smiles(
                df.iloc[unique_csv_rows][smiles_col].tolist(), tokenizer, model, device,
                args.batch_size, max_length, args.pooling, canonicalize, name)
            row_to_embrow = {unique_csv_rows[li]: r for r, li in enumerate(kept_local)}
            failed_csv_rows = {unique_csv_rows[li] for li in failed_local}

            graph_index, source_csv_row, failed_graph_indices = {}, {}, {}
            for split_name, m in matches.items():
                gi_kept = [gi for gi, c in m if c not in failed_csv_rows]
                csv_kept = [c for _, c in m if c not in failed_csv_rows]
                gi_failed = [gi for gi, c in m if c in failed_csv_rows]
                split_embeddings = unique_embeddings[[row_to_embrow[c] for c in csv_kept]]
                out_emb = out_dir / f'{name}_embeddings_{split_name}.npy'
                np.save(out_emb, split_embeddings)
                print(f'Saved {out_emb} ({split_embeddings.shape[0]} rows, '
                      f'{len(unmatched[split_name])} unmatched + {len(gi_failed)} encode-failed dropped)')
                graph_index[split_name] = gi_kept
                source_csv_row[split_name] = csv_kept
                failed_graph_indices[split_name] = gi_failed

            out_meta = out_dir / f'{name}_meta.json'
            with open(out_meta, 'w') as f:
                json.dump({
                    'shape': [n_total, model.config.hidden_size],   # data_curated.csv shape, for reference only
                    'model': args.model,
                    'encoder': 'DeBERTa-v2 (MolDeBERTa pretrained encoder), AutoModel base outputs',
                    'hidden_size': model.config.hidden_size,
                    'max_length': max_length,
                    'pooling': f'{args.pooling} over final hidden states'
                               + (' (masked)' if args.pooling == 'mean' else ''),
                    'canonicalize_smiles': canonicalize,
                    'source_csv': str(cfg['csv']),
                    'smiles_column': smiles_col,
                    'split': ('aqsol_official -- exact torch_geometric.datasets.AQSOL benchmark split '
                              '(7,836 train / 998 valid / 999 test per PyG docs), WL-signature matched to '
                              'data_curated.csv rows for SMILES; NOT deduped, NOT enforced disjoint, to match '
                              'the official split exactly'),
                    'index_space_note': ('Unlike every other dataset in this script, split membership here is '
                                          'indexed by graph_index (the LOCAL position within '
                                          'torch_geometric.datasets.AQSOL(subset omitted, split=...)), not by a '
                                          'data_curated.csv row -- e.g. embeddings row r of '
                                          'aqsol_embeddings_train.npy is AQSOL(split="train")[graph_index["train"][r]]. '
                                          'source_csv_row is the matched CSV row for that graph, kept only for '
                                          'provenance/reference, and MAY REPEAT within or across splits, since the '
                                          'official benchmark itself contains duplicate compounds.'),
                    'graph_index': graph_index,             # PRIMARY index: local position in AQSOL(split)
                    'source_csv_row': source_csv_row,       # provenance only; may repeat, see index_space_note
                    'split_sizes': {k: len(v) for k, v in graph_index.items()},
                    'unmatched_graph_indices': unmatched,   # AQSOL graphs with no WL match to any CSV row at all
                    'failed_indices': sorted(failed_csv_rows),  # CSV rows whose SMILES failed to encode
                    'failed_graph_indices': failed_graph_indices,  # per split: graphs dropped for that reason
                    'failed_handling': ('dropped -- an AQSOL graph with no WL-matched CSV row, or whose matched '
                                        "row's SMILES failed to encode, is excluded entirely from graph_index and "
                                        'the saved embeddings array; never stored as zero vectors. Coverage: '
                                        f'{sum(len(v) for v in graph_index.values())}/'
                                        f'{sum(len(v) for v in matches.values()) + sum(len(v) for v in unmatched.values())} '
                                        'AQSOL graphs (official benchmark total: 9,833).'),
                    'embeddings_dir': str(out_dir),
                }, f, indent=2)
            print('Saved metadata to', out_meta)
            continue

        csv_sha256 = check_split_source(cfg['csv'], cfg['split_json'])
        with open(cfg['split_json']) as f:
            raw_split_idx = json.load(f)

        # Same drop-on-failure encoding path for every scaffold-split dataset.
        # No zero rows are ever produced here.
        kept_idx, embeddings, failed = encode_smiles(
            smiles_list, tokenizer, model, device,
            args.batch_size, max_length, args.pooling, canonicalize, name)

        failed_set = set(failed)
        csv_idx_to_row = {c: r for r, c in enumerate(kept_idx)}

        # Failures are filtered OUT of split_idx here, before ever touching disk --
        # so no split file ever contains a placeholder zero row for a failed molecule.
        new_split_idx = {}
        for split_name, idx_list in raw_split_idx.items():
            filtered = [c for c in idx_list if c not in failed_set]
            split_embeddings = embeddings[[csv_idx_to_row[c] for c in filtered]]
            out_emb = out_dir / f'{name}_embeddings_{split_name}.npy'
            np.save(out_emb, split_embeddings)
            print(f'Saved {out_emb} ({split_embeddings.shape[0]} rows, '
                  f'{len(idx_list) - len(filtered)} dropped)')
            new_split_idx[split_name] = filtered

        unsplit_rows = sorted(set(range(n_total)) - set(sum(raw_split_idx.values(), [])))

        out_meta = out_dir / f'{name}_meta.json'
        with open(out_meta, 'w') as f:
            json.dump({
                'shape': [n_total, model.config.hidden_size],
                'model': args.model,
                'encoder': 'DeBERTa-v2 (MolDeBERTa pretrained encoder), AutoModel base outputs',
                'hidden_size': model.config.hidden_size,
                'max_length': max_length,
                'pooling': f'{args.pooling} over final hidden states'
                           + (' (masked)' if args.pooling == 'mean' else ''),
                'canonicalize_smiles': canonicalize,
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


if __name__ == '__main__':
    main()
