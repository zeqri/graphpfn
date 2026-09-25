"""SMILES repair for FS-Mol: robust_mol() builds an RDKit mol from a possibly-malformed SMILES.

FS-Mol / ChEMBL exports contain a systematic malformation: phosphonic / phosphinic acids written as
[PH](=O)(=O)O (P valence 7 -> RDKit rejects the whole molecule). The real group is P(=O)(O)O
(valence 5). Rewriting it recovers every affected molecule. Other rejects (bad N valence,
unkekulizable rings) are handled by the permissive parse in robust_mol().

Verbatim from FS-Mol's datasets/encode_fsmol_test_embeddings.py (fix_smiles / robust_mol), without
that script's embedding-model code.
"""

from __future__ import annotations

import re

_PHOS_SUBS = [
    (re.compile(r"\[PH\d?\]\(=O\)\(=O\)O"), "P(=O)(O)O"),
    (re.compile(r"\[PH\d?\]\(=O\)=O"), "P(=O)O"),
    (re.compile(r"\[PH\d?\]\(([^)]+)\)\(=O\)=O"), r"P(\1)(=O)O"),
    (re.compile(r"\[PH\d?\]\(=O\)\(([^)]+)\)=O"), r"P(=O)(\1)O"),
    (re.compile(r"\[PH\d?\]"), "P"),  # last-ditch: drop the bogus explicit H
]


def fix_smiles(smi):
    """Return (fixed_smiles, was_changed) applying the known systematic fixes."""
    out = smi
    for pat, repl in _PHOS_SUBS:
        out = pat.sub(repl, out)
    return out, (out != smi)


def robust_mol(Chem, smi):
    """Best-effort RDKit mol from a possibly-malformed SMILES. Never gives up
    unless the string is not parseable at all. Returns (mol, status) where
    status is 'ok' | 'fixed' | 'permissive' | 'permissive_fixed' | 'unparseable'."""
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        return m, "ok"

    fixed, changed = fix_smiles(smi)
    if changed:
        m = Chem.MolFromSmiles(fixed)
        if m is not None:
            return m, "fixed"

    # permissive: parse without sanitizing, then sanitize everything that will
    # not raise (skip PROPERTIES = valence check, and KEKULIZE). The graph
    # featuriser only reads atomic number, chirality, bond type, bond dir --
    # all present after a bare parse -- so this is enough.
    skip = (
        Chem.SanitizeFlags.SANITIZE_PROPERTIES
        | Chem.SanitizeFlags.SANITIZE_KEKULIZE
    )
    for cand, tag in ((smi, "permissive"), (fixed if changed else None, "permissive_fixed")):
        if cand is None:
            continue
        m = Chem.MolFromSmiles(cand, sanitize=False)
        if m is None:
            continue
        try:
            m.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ skip,
                         catchErrors=True)
        return m, tag

    return None, "unparseable"
