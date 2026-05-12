"""Verify Hugging Face access to the base model before any training run.

Usage:
    python scripts/check_model_access.py                  # tokenizer only (fast)
    python scripts/check_model_access.py --full-model     # also downloads weights (~16 GB)
"""

from __future__ import annotations

import sys

import typer

app = typer.Typer(add_completion=False)

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


@app.command()
def main(
    model_id: str = MODEL_ID,
    full_model: bool = typer.Option(False, help="Also download and load the full model weights."),
    prompt: str = typer.Option(
        "Extract the invoice number from: Invoice no: 12345 dated 2024-01-15.",
        help="Prompt used for the smoke-test generation when --full-model is set.",
    ),
) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        typer.echo("transformers not installed. Run: pip install transformers")
        sys.exit(1)

    typer.echo(f"Loading tokenizer for {model_id} ...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        typer.echo(f"\nFAILED: {type(e).__name__}: {e}")
        typer.echo(
            "\nLikely causes:"
            "\n  1. You haven't accepted the license at https://huggingface.co/" + model_id
            + "\n  2. You haven't run `huggingface-cli login` with a token that has access"
            + "\n  3. Your access request is still pending approval"
        )
        sys.exit(2)

    typer.echo(f"  vocab size: {tok.vocab_size}")
    typer.echo(f"  chat template present: {tok.chat_template is not None}")

    sample = tok.apply_chat_template(
        [
            {"role": "system", "content": "Extract fields as JSON."},
            {"role": "user", "content": "Invoice no: 12345"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    typer.echo("\nChat template renders to:")
    typer.echo("-" * 60)
    typer.echo(sample)
    typer.echo("-" * 60)

    if not full_model:
        typer.echo("\nTokenizer-only check passed. Re-run with --full-model to load weights.")
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        typer.echo("torch not installed. Run: pip install torch")
        sys.exit(1)

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32
    typer.echo(f"\nLoading full model onto {device} (dtype={dtype}) — this may take a while ...")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    typer.echo("\nGeneration:")
    typer.echo(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    app()
