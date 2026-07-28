# Optera document pipeline

Turns mixed fleet-operations photos into structured JSON: handwritten mechanic
logs, vendor bills and dashboard readings are extracted; batteries, tyres and
other non-documents are explicitly refused.

## Measured result

The committed OpenAI benchmark ran on all 47 supplied files. The naive baseline
is one full-resolution GPT-4o call per file. The optimized path uses free input
screening, a 384px GPT-4o-mini router, class-specific schemas and compact row
encoding before GPT-4o extraction.

| | GPT-4o baseline | Optimized pipeline |
|---|---:|---:|
| Cost per document | $0.010309 | **$0.008449** |
| Total cost | $0.484545 | **$0.397088** |
| API-reported tokens | 111,774 in / 20,255 out | 456,731 in / 16,828 out |
| Field accuracy | 59/91 (64.8%) | **66/91 (72.5%)** |
| Routing accuracy | 39/39 | 39/39 |
| Refusal accuracy | 5/8 | **8/8** |
| Hallucinated records | 2 | **0** |

That is **18.04% lower cost (1.22x cheaper)** with higher scored accuracy.
The raw extraction outputs, call-by-call token ledgers and scored report live in
[`results/openai/`](results/openai/).

## Run

```bash
pip install -r requirements.txt
cp openai.env.example .env
# Add OPENAI_API_KEY to .env
make run
```

`make run` is one command for the required comparison: it runs the naive GPT-4o
baseline and the optimized path, prints cost/accuracy, and writes timestamped
JSON, JSONL ledgers and a report to `out/`.

The optimized path also keeps a conservative local `.optera-cache/`: it reuses
only an exact-byte match whose prior output passed validation at high confidence,
and invalidates automatically when a model, prompt or schema changes. Use
`make run` on a fresh clone or `python3 run.py --no-cache` for a fresh benchmark.

Useful variants:

```bash
python3 run.py --mode optimized
python3 run.py --mode baseline
python3 run.py --input /path/to/images
python3 run.py --limit 5 --no-score
make verify  # validates committed results without images, API key or network
make test
```

### Asynchronous Batch API

The repository includes a resumable, two-phase OpenAI Batch workflow for a
nightly queue. It sends one document per JSONL request—batching receives the
API discount without mixing unrelated document images in one model context.

```bash
# 1. Submit baseline and optimized-router jobs independently.
python3 scripts/batch_requests.py build --stage baseline --input images --out out/batch/baseline.jsonl
python3 scripts/batch_requests.py submit --requests out/batch/baseline.jsonl --state out/batch/baseline-state.json --label baseline
python3 scripts/batch_requests.py build --stage router --input images --out out/batch/router.jsonl
python3 scripts/batch_requests.py submit --requests out/batch/router.jsonl --state out/batch/router-state.json --label router

# 2. After the router completes, collect routes and submit extraction.
python3 scripts/batch_requests.py collect --state out/batch/router-state.json --out out/batch/router-output.jsonl
python3 scripts/batch_requests.py routes --router-output out/batch/router-output.jsonl --out out/batch/routes.json
python3 scripts/batch_requests.py build --stage extract --input images --routes out/batch/routes.json --out out/batch/extract.jsonl
python3 scripts/batch_requests.py submit --requests out/batch/extract.jsonl --state out/batch/extract-state.json --label extract
python3 scripts/batch_requests.py collect --state out/batch/extract-state.json --out out/batch/extract-output.jsonl
```

The published headline remains the synchronous benchmark until this staged run
finishes and is scored against the same gold labels.

## Pipeline

```
files
  ├─ free preflight: content sniffing, EXIF rotation, truncation recovery,
  │                  duplicate detection and quality flags
  ├─ 384px GPT-4o-mini router: document class/subtype or refusal
  └─ GPT-4o extractor: only the schema for that class, compact row output
                         followed by free arithmetic/date/shape validation
```

Preflight rejected three of the 47 supplied files without a model call: an HTML
error page disguised as an image and two near-duplicate forwards. The router
refused all eight non-records in the optimized run, so a battery photo never
reaches an invoice or mechanic-log schema.

`--targeted-reread` is an opt-in recovery path for a failed GSTIN or meter
arithmetic check. It crops only the relevant portion and applies its result only
when the deterministic validator improves; it is deliberately excluded from
the headline until it has a labelled accuracy result.

## Evidence and scope

`make verify` recomputes the committed numbers from raw JSONL usage records,
extractions and hand-written ground truth. It does not make model calls.

The real images are intentionally excluded from Git because they are client
operational data. See `images/README.md`. The ground-truth labels and evidence
artifacts are committed so the scoring and cost arithmetic are reviewable.

See [DESIGN.md](DESIGN.md) for limits and next steps.
