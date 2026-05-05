from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DocType = Literal["invoice", "contract"]
Split = Literal["train", "val", "test"]


class ExtractionRecord(BaseModel):
    """One (input_text, output_json) training example in a dataset-neutral shape.

    The record is intentionally decoupled from any chat template — formatting for
    Llama-3 SFT happens at training time from (fields, input_text, output_json).
    """

    id: str
    doc_type: DocType
    source_dataset: str
    fields: dict[str, str] = Field(
        description="Map of field_name -> type descriptor (e.g. 'string', 'date', 'currency')."
    )
    input_text: str
    output_json: dict[str, Any]
    split: Split | None = None
