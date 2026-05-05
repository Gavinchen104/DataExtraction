"""Loaders that normalize source datasets into `ExtractionRecord` objects.

Each loader is a generator over records. Loaders do no IO beyond `datasets.load_dataset`
and do not assign splits — that is the splitter's job.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

from src.data.schema import ExtractionRecord

# CUAD questions look like:
#   'Highlight the parts (if any) of this contract related to "Governing Law" ...'
# The inner quoted phrase is the clause category name.
_CUAD_CATEGORY_RE = re.compile(r'related to "([^"]+)"')


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown"


def load_cuad(rows: Iterable[dict[str, Any]]) -> Iterator[ExtractionRecord]:
    """Convert CUAD (SQuAD-style) rows into per-clause extraction records.

    Each CUAD row is (contract_chunk, question_about_one_clause, answers). The question
    names the clause category; we turn that into a single-field extraction task.
    Empty `answers.text` means the clause is absent — modeled as null.
    """
    for row in rows:
        question = row.get("question") or ""
        match = _CUAD_CATEGORY_RE.search(question)
        if not match:
            continue
        category = match.group(1)
        field_name = _slugify(category)

        answers = row.get("answers") or {}
        answer_texts = answers.get("text") or []
        value: str | None = answer_texts[0].strip() if answer_texts else None

        context = row.get("context") or ""
        contract_id = row.get("title") or row.get("id") or "unknown"

        yield ExtractionRecord(
            id=f"cuad-{row.get('id', contract_id + '-' + field_name)}",
            doc_type="contract",
            source_dataset="cuad",
            fields={field_name: "string"},
            input_text=context,
            output_json={field_name: value},
        )


_INVOICE_FIELDS: dict[str, str] = {
    "invoice_number": "string",
    "invoice_date": "date",
    "due_date": "date",
    "seller_name": "string",
    "seller_address": "string",
    "seller_tax_id": "string",
    "client_name": "string",
    "client_address": "string",
    "client_tax_id": "string",
    "iban": "string",
    "subtotal": "currency",
    "tax": "currency",
    "total": "currency",
    "items": "list_of_line_items",
}


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _extract_ocr_text(raw_data: Any) -> str:
    parsed = _coerce_json(raw_data)
    if isinstance(parsed, dict) and "ocr_words" in parsed:
        words = parsed["ocr_words"]
        if isinstance(words, list):
            return "\n".join(str(w) for w in words)
    if isinstance(raw_data, str):
        return raw_data
    return ""


def _normalize_invoice_output(parsed: dict[str, Any]) -> dict[str, Any]:
    """Flatten the mychen76 nested JSON into the canonical invoice schema."""
    header = parsed.get("header") or {}
    subtotal_block = parsed.get("subtotal") or {}
    payment = parsed.get("payment_instructions") or {}
    items = parsed.get("items") or []

    out: dict[str, Any] = {
        "invoice_number": header.get("invoice_no") or header.get("invoice_number"),
        "invoice_date": header.get("invoice_date"),
        "due_date": header.get("due_date") or payment.get("due_date"),
        "seller_name": header.get("seller"),
        "seller_address": None,
        "seller_tax_id": header.get("seller_tax_id"),
        "client_name": header.get("client"),
        "client_address": None,
        "client_tax_id": header.get("client_tax_id"),
        "iban": header.get("iban") or payment.get("account_number"),
        "subtotal": subtotal_block.get("subtotal") or subtotal_block.get("gross_worth"),
        "tax": subtotal_block.get("tax"),
        "total": subtotal_block.get("total"),
        "items": items if isinstance(items, list) else [],
    }
    return out


def load_mychen76(rows: Iterable[dict[str, Any]]) -> Iterator[ExtractionRecord]:
    """Convert mychen76/invoices-and-receipts_ocr_v1 rows into invoice records."""
    for row in rows:
        input_text = _extract_ocr_text(row.get("raw_data"))
        if not input_text:
            continue

        parsed = _coerce_json(row.get("parsed_data"))
        if not isinstance(parsed, dict):
            continue

        output = _normalize_invoice_output(parsed)
        row_id = row.get("id") or parsed.get("header", {}).get("invoice_no") or "unknown"

        yield ExtractionRecord(
            id=f"mychen76-{row_id}",
            doc_type="invoice",
            source_dataset="mychen76",
            fields=dict(_INVOICE_FIELDS),
            input_text=input_text,
            output_json=output,
        )


LOADERS: dict[str, Any] = {
    "cuad": load_cuad,
    "mychen76": load_mychen76,
}
