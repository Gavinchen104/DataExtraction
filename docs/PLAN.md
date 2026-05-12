# Project Plan

Living roadmap for the Llama-3-8B + LoRA structured extraction project. Update as phases complete.

## Status snapshot

**Last updated:** 2026-05-04

| Phase | Status |
|---|---|
| 0. Framing & scaffolding | ✅ Done |
| 1. Data materialization | 🟡 Code written, not yet run |
| 2. Eval harness | ⬜ Not started |
| 3. Baselines | ⬜ Not started |
| 4. Fine-tune (single run) | ⬜ Not started |
| 5. Ablations | ⬜ Not started |
| 6. Error analysis + iteration | ⬜ Not started |
| 7. Serving + demo | ⬜ Not started |
| 8. Write-up | ⬜ Not started |

## Compute environment

**Target cluster:** Duke CS SLURM (`compsci-gpu` partition)
**Job submission:** `sbatch`; interactive via `srun -p compsci-gpu --gres=gpu:1 --pty bash -i`
**Open cluster questions to resolve before Phase 4:**
- Which GPU types are available? (Need 24 GB+ VRAM — A100, A6000, A5000, L40, 3090/4090. The 2080rtx and k80 mentioned in docs are unusable for Llama-3-8B.)
- Do compute nodes have internet access? (Determines whether we pre-download weights from the login node.)
- Confirmed scratch / project storage path for HF cache (`HF_HOME` must NOT be on home dir — Llama-3-8B is 16 GB).

## Phase 0 — Framing & scaffolding ✅

**Done:**
- [CLAUDE.md](../CLAUDE.md) — project north star.
- Repo skeleton ([pyproject.toml](../pyproject.toml), [.gitignore](../.gitignore), [configs/](../configs/)).
- Canonical schema ([data/schema.json](../data/schema.json), [src/data/schema.py](../src/data/schema.py)).
- Data loaders ([src/data/loaders.py](../src/data/loaders.py)) for CUAD and mychen76.
- Deterministic hash-based splitter ([src/data/splits.py](../src/data/splits.py)).
- Data prep CLI ([scripts/prepare_data.py](../scripts/prepare_data.py)).
- Model access check ([scripts/check_model_access.py](../scripts/check_model_access.py)).
- 9 passing unit tests ([tests/](../tests/)).

## Phase 1 — Data materialization 🟡

**Goal:** Produce frozen `data/processed/{train,val,test}.jsonl`. Test set never touched again.

**Steps:**
1. `pip install datasets typer pyyaml transformers` (local laptop).
2. Run `python scripts/check_model_access.py` to verify HF auth for Llama-3-8B (tokenizer-only).
3. Run `python scripts/prepare_data.py --config configs/data.yaml`.
4. Inspect 5 sample rows per split; verify schema correctness.
5. Confirm split sizes match expectation (~13k CUAD + ~2.2k invoices ≈ 15k total).

**Acceptance:**
- [ ] `data/processed/train.jsonl` exists, non-empty.
- [ ] `data/processed/val.jsonl` ~5% of total.
- [ ] `data/processed/test.jsonl` ~10% of total — **frozen from this point forward**.
- [ ] `manifest.json` records seed, fractions, source dataset ids.
- [ ] At least one sample of each (doc_type, source_dataset) eyeballed for correctness.

**Risks:**
- CUAD or mychen76 column shapes may differ from assumptions — loader fixes likely on first run.
- mychen76 license is unspecified — flag in write-up; can be dropped if needed.

## Phase 2 — Eval harness ⬜

**Goal:** Single source of truth for "how good is the extractor." Built once, run on every model forever.

**Deliverables:**
- `src/eval/metrics.py` — `field_f1`, `normalize(value, field_type)`, `json_validity_rate`, per-doc-type breakdown.
- `src/eval/runner.py` — given a model's predictions JSONL, compute the full metrics table.
- `tests/test_metrics.py` — round-trip tests on known examples, edge cases (empty, null, malformed JSON).

**Acceptance:**
- [ ] Field-level macro F1 computed end-to-end on a toy dataset matches hand-calculated truth.
- [ ] Normalization handles: date formats (`10/15/2012` ≡ `2012-10-15`), currency (`$1,250.00` ≡ `1250.00`), whitespace, case for strings.
- [ ] Malformed JSON counted as full-miss for that document (not silently skipped).
- [ ] Per-document-type and per-field breakdowns rendered as markdown tables.

## Phase 3 — Baselines ⬜

**Goal:** Lock in the numbers the fine-tuned model must beat. **No fine-tuning until these exist.**

**Deliverables (in cost order):**
1. **Regex / heuristic** — `src/eval/baselines/regex.py`. Local, free. Sets the floor.
2. **GPT-4o or Claude Sonnet few-shot** — `src/eval/baselines/api_few_shot.py`. 5 in-context examples. ~$5–20 in API spend on the full test set.
3. **Llama-3-8B-Instruct zero-shot** — `src/eval/baselines/llama_zero_shot.py`. Runs on the cluster; same chat template as the eventual fine-tune.

**Acceptance:**
- [ ] All three baselines produce predictions in the same JSONL format.
- [ ] All three evaluated by `src/eval/runner.py`.
- [ ] Results recorded in `docs/results.md` with cost-per-1k-docs and p95 latency where applicable.

## Phase 4 — Fine-tune (single run) ⬜

**Blocked on:** Phases 1–3 complete; cluster questions resolved.

**Deliverables:**
- `src/train/format.py` — `ExtractionRecord` → Llama-3 chat-template SFT text.
- `scripts/train.py` — TRL `SFTTrainer` + PEFT LoRA r=16, QLoRA 4-bit NF4, bf16, packing, gradient checkpointing.
- `configs/lora_r16.yaml` — full training config (mirrors CLAUDE.md defaults).
- `slurm/train.sbatch` — cluster submission script with `HF_HOME` on scratch.
- Smoke-test: 50-row subset, 1 epoch, end-to-end success before submitting the real job.

**Acceptance:**
- [ ] Smoke-test produces a valid LoRA adapter and one parseable JSON generation.
- [ ] Full run completes in <12 hr on a single A100/A6000.
- [ ] Adapter checkpoint saved to `runs/lora_r16/`.
- [ ] wandb run recorded with loss curves and eval-on-val metrics.
- [ ] Fine-tuned model evaluated on the same frozen test set with `src/eval/runner.py`.

## Phase 5 — Ablations ⬜

**Goal:** Move from "I trained a model" to "I understand what makes this model work." This is where FAANG portfolio differentiation lives.

**Required ablations:**
- LoRA rank sweep: r ∈ {4, 8, 16, 32}.
- Target-module ablation: attention-only vs +MLP (gate/up/down_proj).
- Data-scaling curve: train on {25%, 50%, 75%, 100%} of training data.
- Base-model comparison: Llama-3-8B vs Mistral-7B vs Phi-3-mini.

**Acceptance:**
- [ ] Each ablation has its own config in `configs/` and its own wandb run.
- [ ] Results summarized as a single comparison table in `docs/results.md`.
- [ ] At least one finding stated as a takeaway, e.g. "r=8 matches r=16 at half the parameter count."

## Phase 6 — Error analysis + iteration ⬜

**Deliverables:**
- `notebooks/error_analysis.ipynb` — per-field confusion, top-20 failure documents with annotated causes, hallucination rate (fields invented not in document).
- Targeted improvements driven by findings: prompt tweaks, additional synthetic data, schema refinement.

**Acceptance:**
- [ ] At least one identified failure mode has a measurable fix that improves macro F1 by ≥1 point.
- [ ] Hallucination rate measured and reported.

## Phase 7 — Serving + demo ⬜

**Deliverables:**
- `src/serve/api.py` — FastAPI wrapper around vLLM with the merged or PEFT-loaded model.
- Inference latency benchmark: p50/p95, tokens/sec, cost/1k docs (using a known cloud GPU rate as reference).
- Minimal web demo (Gradio or Streamlit) for the recorded portfolio demo.

**Acceptance:**
- [ ] API returns valid JSON for the schema in <3s p95 on a single A10G or equivalent.
- [ ] Demo video recorded showing extraction from a fresh invoice or contract.

## Phase 8 — Write-up ⬜

**Deliverables:**
- Polished public README with method, results table, honest limitations, demo link.
- Technical report PDF in `docs/report.pdf` — ~6–10 pages: motivation, method, ablations, error analysis, deployment, limitations.
- (Stretch) Blog post on Medium / personal site.

**Acceptance:**
- [ ] Results table reports fine-tune *alongside* all three baselines on the same test set.
- [ ] At least one place in the write-up admits where the fine-tune lost to a baseline, and why.
- [ ] README runnable end-to-end by a stranger with the listed commands.

## Open decisions

- **Add RAG (long-context contract retrieval)?** Deferred until after Phase 5. Only add if ablations show context-window truncation is hurting contract F1.
- **Synthetic data augmentation?** Deferred until Phase 6. Only add if data-scaling curve shows we're data-bound, not capacity-bound.
- **Drop mychen76 if license blocks publication?** Decide before Phase 8. Backup invoice source: Voxel51 ODbL dataset (1,489 annotated).

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Compute nodes lack internet → can't pull Llama-3 weights | Pre-download from login node; point `HF_HOME` to scratch |
| Only 2080 / k80 GPUs available → can't train 8B | Confirm A100/A6000 access; if not, fall back to Phi-3-mini or Llama-3.2-3B |
| Fine-tune loses to GPT-4o few-shot on every field | Reframe story around cost/latency/on-prem — still a valid project, but lead with that |
| Test-set contamination via CUAD chunks | Already mitigated: splitter groups all chunks of a contract |
| HF gated-model approval delayed | Run access check today, not on training day |
