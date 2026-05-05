from __future__ import annotations

from src.data.schema import ExtractionRecord
from src.data.splits import assign_split, assign_splits


def _make(id_: str, source: str = "mychen76") -> ExtractionRecord:
    return ExtractionRecord(
        id=id_,
        doc_type="invoice",
        source_dataset=source,
        fields={"invoice_number": "string"},
        input_text="x",
        output_json={"invoice_number": "1"},
    )


def test_split_assignment_is_deterministic():
    r = _make("inv-1")
    first = assign_split(r, seed=42, val_frac=0.1, test_frac=0.1)
    second = assign_split(r, seed=42, val_frac=0.1, test_frac=0.1)
    assert first == second


def test_different_seeds_produce_different_buckets_on_average():
    records = [_make(f"inv-{i}") for i in range(200)]
    s1 = [assign_split(r, 1, 0.1, 0.1) for r in records]
    s2 = [assign_split(r, 2, 0.1, 0.1) for r in records]
    assert s1 != s2


def test_approximate_split_fractions():
    records = [_make(f"inv-{i}") for i in range(5000)]
    assigned = list(assign_splits(records, seed=42, val_frac=0.05, test_frac=0.10))
    counts = {"train": 0, "val": 0, "test": 0}
    for r in assigned:
        counts[r.split] += 1  # type: ignore[index]
    assert 0.08 < counts["test"] / 5000 < 0.12
    assert 0.03 < counts["val"] / 5000 < 0.07
    assert 0.80 < counts["train"] / 5000 < 0.90


def test_cuad_contract_chunks_share_a_split():
    """All chunks of one CUAD contract must land in the same split to prevent leakage."""
    chunks = [
        ExtractionRecord(
            id=f"cuad-Contract_A__{i}",
            doc_type="contract",
            source_dataset="cuad",
            fields={"governing_law": "string"},
            input_text="...",
            output_json={"governing_law": None},
        )
        for i in range(20)
    ]
    assigned = list(assign_splits(chunks, seed=42, val_frac=0.1, test_frac=0.1))
    splits = {r.split for r in assigned}
    assert len(splits) == 1
