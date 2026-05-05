# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project mission

Fine-tune **Llama-3-8B** with **LoRA (r=16)** on ~20k labeled documents (invoices, contracts) to extract structured fields (names, dates, amounts, parties, line items) as JSON from unstructured text.

The goal is a portfolio-grade ML project strong enough to open doors at FAANG-level AI/ML teams. That means: reproducible training, honest evaluation against strong baselines, a working demo, and a clear write-up.

## What a FAANG reviewer will look for

Build every component with these in mind — they are the acceptance criteria:

1. **Clean problem framing** — a crisp schema, documented label taxonomy, and a held-out test set frozen from day one.
2. **Strong baselines** — zero-shot Llama-3-8B, GPT-4o / Claude few-shot, and a regex/heuristic baseline. The fine-tuned model must beat all three on the same test set, or the project explains why not.
3. **Proper evaluation** — field-level precision/recall/F1, exact-match vs normalized-match (dates, currency), per-document-type breakdowns, and confidence calibration.
4. **Ablations** — LoRA rank sweep (r=4, 8, 16, 32), target-module ablation, data-size scaling curve, base-model comparison (Llama-3-8B vs Mistral-7B vs Phi-3).
5. **Production thinking** — inference latency (p50/p95), tokens/sec, cost per 1k docs, failure-mode analysis, and a small FastAPI/vLLM serving layer.
6. **Reproducibility** — pinned deps, seeds, configs checked in, one-command training + eval.
7. **Write-up** — README with method, results, and honest limitations; ideally a short blog post or technical report PDF in `docs/`.

## Tech stack

- **Model**: meta-llama/Meta-Llama-3-8B-Instruct (base for instruction-tuned field extraction)
- **Fine-tuning**: PEFT (LoRA r=16, alpha=32, dropout=0.05), QLoRA 4-bit NF4 for single-GPU training
- **Target modules**: q_proj, k_proj, v_proj, o_proj (all attention projections; ablate adding gate/up/down_proj)
- **Trainer**: TRL `SFTTrainer` with packing, bf16, gradient checkpointing
- **Serving**: vLLM for batched inference; FastAPI wrapper for the demo
- **Tracking**: Weights & Biases (runs, sweeps, artifacts)
- **Eval**: custom field-level F1 + `evaluate` library; per-field confusion analysis
- **Env**: Python 3.11, PyTorch 2.3+, CUDA 12.1, single A100-40GB or 2x A10G minimum

## Data strategy

20k high-quality labeled documents is the hard part. Do not hand-label from scratch — combine these sources:

- **Invoices**: SROIE, CORD, RVL-CDIP invoice subset, DocILE (public, labeled)
- **Contracts**: CUAD (510 contracts, 41 clause types), LEDGAR, EDGAR filings
- **Synthetic augmentation**: use GPT-4o / Claude to generate varied invoice/contract templates with known ground-truth fields — this is a legitimate technique and should be disclosed
- **Weak supervision**: template-matching + regex to bootstrap labels, then human-verify a stratified sample

Freeze a **2k-document test set** on day one. Never train on it, never tune on it, never peek. Track a separate **1k-document validation set** for checkpoint selection.

## Repository layout (target)

```
.
├── CLAUDE.md              # this file
├── README.md              # public-facing write-up (method, results, demo)
├── pyproject.toml         # pinned deps, ruff/black config
├── configs/               # YAML training configs (one per experiment)
├── data/
│   ├── raw/               # untouched source datasets
│   ├── processed/         # tokenized, formatted train/val/test splits
│   └── schema.json        # canonical field taxonomy
├── src/
│   ├── data/              # loaders, formatters, synthetic data gen
│   ├── train/             # SFTTrainer wrapper, LoRA setup
│   ├── eval/              # field-level metrics, baselines
│   ├── serve/             # vLLM + FastAPI inference server
│   └── prompts/           # instruction templates, few-shot examples
├── scripts/               # train.py, eval.py, infer.py, prepare_data.py
├── notebooks/             # EDA, error analysis (not for core logic)
├── tests/                 # pytest: data contracts, metric correctness, schema validation
└── docs/                  # technical report, architecture diagrams, result tables
```

## Training config defaults

```yaml
model: meta-llama/Meta-Llama-3-8B-Instruct
quantization: nf4
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
training:
  epochs: 3
  per_device_batch_size: 4
  gradient_accumulation_steps: 4   # effective batch = 16
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  max_seq_length: 4096             # contracts are long — validate distribution
  packing: true
  bf16: true
  gradient_checkpointing: true
seed: 42
```

## Instruction format

Train the model to output strict JSON matching `data/schema.json`. Use a consistent template:

```
<|system|>Extract the following fields as JSON: {field_list}. Use null for missing fields.
<|user|>{document_text}
<|assistant|>{json_output}
```

During eval, parse JSON strictly — a malformed output counts as a full miss for that document. Track JSON-validity rate as a first-class metric.

## Evaluation protocol

- **Primary metric**: macro-averaged field-level F1 on the frozen test set
- **Secondary**: JSON-validity rate, exact-match accuracy, normalized-match (dates parsed, currency normalized), per-document-type F1
- **Baselines to always report alongside the fine-tuned model**:
  1. Llama-3-8B-Instruct zero-shot (same prompt)
  2. GPT-4o / Claude Sonnet few-shot (5 in-context examples)
  3. Regex + heuristic extractor
- **Error analysis**: per-field confusion matrix, top-20 failure documents with annotated reasons, hallucination rate (fields invented that aren't in the document)

## Roadmap (suggested milestones)

1. **Week 1** — schema design, dataset sourcing, test/val freeze, EDA notebook
2. **Week 2** — baselines (zero-shot, GPT-4o few-shot, regex) with full eval pipeline
3. **Week 3** — first fine-tune run end-to-end, wandb tracking, basic serving
4. **Week 4** — LoRA ablations (r, target modules), data-scaling curve
5. **Week 5** — error analysis, prompt iteration, synthetic data augmentation
6. **Week 6** — inference optimization (vLLM, batching, latency numbers), demo UI
7. **Week 7** — write-up, blog post, recorded demo, polish README

## Development commands (fill in as implemented)

```bash
# Setup
uv sync                                  # install pinned deps

# Data
python scripts/prepare_data.py --config configs/data.yaml

# Train
python scripts/train.py --config configs/lora_r16.yaml

# Evaluate (runs all baselines + fine-tuned model on frozen test set)
python scripts/eval.py --checkpoint runs/lora_r16/final --report docs/results.md

# Serve
python -m src.serve.api --checkpoint runs/lora_r16/final --port 8000
```

## Guidance for Claude when working in this repo

- **Never touch the frozen test set.** If a script references `data/processed/test.jsonl`, it is for evaluation only.
- **Every experiment gets a config file in `configs/` and a wandb run.** Do not run ad-hoc training from the CLI without a config.
- **Do not report numbers without the baseline numbers alongside them.** A fine-tuned F1 without zero-shot and GPT-4o comparison is not a result.
- **Prefer reading existing configs and extending them** over writing new ones from scratch.
- **When adding a metric, add a test for it in `tests/`.** Metric bugs silently invalidate entire projects.
- **Keep the README honest.** If the fine-tuned model loses to GPT-4o few-shot on some field, say so and discuss why. Reviewers respect honesty more than cherry-picked wins.
