"""Build unified train/val/test JSONL from source HF datasets.

Usage:
    python scripts/prepare_data.py --config configs/data.yaml
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
import yaml
from datasets import load_dataset

from src.data.loaders import LOADERS
from src.data.schema import ExtractionRecord
from src.data.splits import assign_splits

app = typer.Typer(add_completion=False)


def _load_source(spec: dict) -> list[ExtractionRecord]:
    loader_name = spec["loader"]
    if loader_name not in LOADERS:
        raise ValueError(f"Unknown loader {loader_name!r}. Available: {list(LOADERS)}")

    ds = load_dataset(
        spec["hf_id"],
        split=spec.get("split", "train"),
        trust_remote_code=spec.get("trust_remote_code", False),
    )
    max_chars = spec.get("max_context_chars")

    records: list[ExtractionRecord] = []
    for rec in LOADERS[loader_name](ds):
        if max_chars and len(rec.input_text) > max_chars:
            rec.input_text = rec.input_text[:max_chars]
        records.append(rec)
    return records


def _write_jsonl(path: Path, records: list[ExtractionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


@app.command()
def main(
    config: Path = typer.Option(..., exists=True, readable=True),
) -> None:
    cfg = yaml.safe_load(config.read_text())

    all_records: list[ExtractionRecord] = []
    for source in cfg["sources"]:
        typer.echo(f"Loading {source['name']} from {source['hf_id']} ...")
        src_records = _load_source(source)
        typer.echo(f"  -> {len(src_records)} records")
        all_records.extend(src_records)

    assigned = list(
        assign_splits(
            all_records,
            seed=cfg["seed"],
            val_frac=cfg["splits"]["val_fraction"],
            test_frac=cfg["splits"]["test_fraction"],
        )
    )

    out_dir = Path(cfg["output"]["dir"])
    by_split: dict[str, list[ExtractionRecord]] = {"train": [], "val": [], "test": []}
    for r in assigned:
        assert r.split is not None
        by_split[r.split].append(r)

    _write_jsonl(out_dir / cfg["output"]["train_file"], by_split["train"])
    _write_jsonl(out_dir / cfg["output"]["val_file"], by_split["val"])
    _write_jsonl(out_dir / cfg["output"]["test_file"], by_split["test"])

    typer.echo("\nSplit sizes:")
    for split_name, recs in by_split.items():
        by_source = Counter(r.source_dataset for r in recs)
        typer.echo(f"  {split_name:5s}: {len(recs):>6d}   {dict(by_source)}")

    manifest = {
        "seed": cfg["seed"],
        "val_fraction": cfg["splits"]["val_fraction"],
        "test_fraction": cfg["splits"]["test_fraction"],
        "sources": [s["hf_id"] for s in cfg["sources"]],
        "split_sizes": {k: len(v) for k, v in by_split.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    app()
