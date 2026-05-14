from __future__ import annotations

import json

from src.data.loaders import load_cuad, load_mychen76


def test_cuad_parses_category_and_answer():
    rows = [
        {
            "id": "Contract_X__0",
            "title": "Contract_X",
            "context": "This Agreement shall be governed by the laws of New York.",
            "question": 'Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed.',
            "answers": {"text": ["the laws of New York"], "answer_start": [40]},
        }
    ]
    records = list(load_cuad(rows))
    assert len(records) == 1
    r = records[0]
    assert r.doc_type == "contract"
    assert r.source_dataset == "cuad"
    assert r.fields == {"governing_law": "string"}
    assert r.output_json == {"governing_law": "the laws of New York"}


def test_cuad_empty_answer_becomes_null():
    rows = [
        {
            "id": "Contract_Y__0",
            "title": "Contract_Y",
            "context": "...",
            "question": 'Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed.',
            "answers": {"text": [], "answer_start": []},
        }
    ]
    records = list(load_cuad(rows))
    assert records[0].output_json == {"non_compete": None}


def test_cuad_skips_rows_without_category():
    rows = [{"id": "x", "title": "x", "context": "x", "question": "malformed", "answers": {}}]
    assert list(load_cuad(rows)) == []


def test_mychen76_flattens_nested_json():
    """mychen76 wraps invoice data as {xml, json: '<repr-string>', kie}."""
    inner_invoice = {
        "header": {
            "invoice_no": "40378170",
            "invoice_date": "10/15/2012",
            "seller": "Acme Corp",
            "client": "Jackson Inc",
            "iban": "GB77WRBQ",
        },
        "items": [
            {
                "item_desc": "Widget",
                "item_qty": "2,00",
                "item_net_price": "5,00",
                "item_gross_worth": "10,00",
            }
        ],
        "summary": {
            "total_net_worth": "$10,00",
            "total_vat": "$1,00",
            "total_gross_worth": "$11,00",
        },
    }
    envelope = {"xml": "", "json": repr(inner_invoice), "kie": ""}
    raw = {"ocr_words": repr(["Invoice no: 40378170", "Date: 10/15/2012"])}
    rows = [{"id": "abc", "parsed_data": json.dumps(envelope), "raw_data": json.dumps(raw)}]

    records = list(load_mychen76(rows))
    assert len(records) == 1
    r = records[0]
    assert r.doc_type == "invoice"
    assert r.output_json["invoice_number"] == "40378170"
    assert r.output_json["total"] == "$11,00"
    assert r.output_json["items"][0]["description"] == "Widget"
    assert r.output_json["items"][0]["total_price"] == "10,00"
    assert "Invoice no: 40378170" in r.input_text


def test_mychen76_skips_rows_with_unparseable_envelope():
    rows = [{"id": "x", "parsed_data": "{{{not parseable", "raw_data": "some text"}]
    assert list(load_mychen76(rows)) == []
