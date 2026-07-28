# Evidence

Raw logs behind every number in the top-level README and DESIGN.md. Nothing here
is regenerated or rounded for presentation — the ledgers are the append-only
files the pipeline wrote call-by-call while running.

## Current OpenAI evidence

`openai/` contains the current, independently rerun GPT-4o baseline and
GPT-4o-mini-router/GPT-4o-extractor comparison over all 47 input files:
**$0.010309/doc → $0.008449/doc**, field accuracy **59/91 → 66/91**, and
refusal accuracy **5/8 → 8/8**. It includes the raw cost/token ledgers and the
full scored report. The artefacts in the rest of this folder are the earlier
Anthropic benchmark and ablations.

## Ledgers (one JSONL row per API call)

| file | what |
|---|---|
| `ledger_baseline_opus1x.jsonl` | the naive 1x: one full-res Opus call per image |
| `ledger_optimized_accurate.jsonl` | shipped default profile |
| `ledger_optimized_balanced.jsonl` | mid operating point |
| `ledger_optimized_cheap.jsonl` | maximum-savings operating point |

Each row carries `doc_id`, `stage`, `model`, `input_tokens`, `output_tokens`,
`cache_read_tokens`, `cost_usd`, `latency_s` and the image resolution used.
Costs are computed from API-reported token counts against `optera/pricing.json`.

Reproduce any headline figure:

```bash
python3 -c "import json;rows=[json.loads(l) for l in open('results/ledger_optimized_accurate.jsonl')];\
print(round(sum(r['cost_usd'] for r in rows),6), 'over', len({r['doc_id'] for r in rows})-1, 'docs')"
```

## Reports

| file | what |
|---|---|
| `report_baseline_and_balanced.json` | baseline + balanced, with full per-document scoring |
| `report_accurate.json` | shipped default |
| `report_cheap.json` | cheapest profile |

Each contains `cost` (token/USD rollups by stage and model) and `accuracy`
(routing, refusal, field, hallucinations, and a `per_doc` breakdown naming every
individual field miss).

## Experiments

| file | what |
|---|---|
| `sweep.json` | model x resolution grid per class — the evidence behind `CLASS_POLICY` |
| `variance.json` | same config run 3x; establishes the noise floor (1.1 pts overall) |

## Extractions

`extractions_optimized_accurate.json` and `extractions_baseline_opus1x.json`
hold the actual structured output for all 47 images, including refusals and
per-document provenance.
