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

Useful variants:

```bash
python3 run.py --mode optimized
python3 run.py --mode baseline
python3 run.py --input /path/to/images
python3 run.py --limit 5 --no-score
make verify  # validates committed results without images, API key or network
make test
```

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

## Evidence and scope

`make verify` recomputes the committed numbers from raw JSONL usage records,
extractions and hand-written ground truth. It does not make model calls.

The real images are intentionally excluded from Git because they are client
operational data. See `images/README.md`. The ground-truth labels and evidence
artifacts are committed so the scoring and cost arithmetic are reviewable.

See [DESIGN.md](DESIGN.md) for limits and next steps.
