"""Deterministic hash-based train/val/test assignment.

Hashing the grouping key (not random shuffling) means adding new source data
never reshuffles existing records across splits — a frozen test set stays frozen.
For CUAD we group by contract title so all chunks of a contract land in the same
split, preventing context leakage from train into test.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator

from src.data.schema import ExtractionRecord, Split


def _group_key(record: ExtractionRecord) -> str:
    """Key used for bucket assignment — all records with the same key share a split."""
    if record.source_dataset == "cuad":
        # Strip the per-chunk suffix from CUAD ids so all chunks of one contract
        # end up in the same split. CUAD ids look like 'contract_name__CHUNK_N'.
        base = re.sub(r"__.*$", "", record.id)
        return f"cuad::{base}"
    return f"{record.source_dataset}::{record.id}"


def _bucket(key: str, seed: int) -> float:
    """Map a key + seed to a stable float in [0, 1)."""
    digest = hashlib.sha256(f"{seed}::{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(record: ExtractionRecord, seed: int, val_frac: float, test_frac: float) -> Split:
    assert 0 <= val_frac < 1
    assert 0 <= test_frac < 1
    assert val_frac + test_frac < 1

    b = _bucket(_group_key(record), seed)
    if b < test_frac:
        return "test"
    if b < test_frac + val_frac:
        return "val"
    return "train"


def assign_splits(
    records: Iterable[ExtractionRecord], seed: int, val_frac: float, test_frac: float
) -> Iterator[ExtractionRecord]:
    for r in records:
        r.split = assign_split(r, seed, val_frac, test_frac)
        yield r
