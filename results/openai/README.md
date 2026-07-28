# OpenAI re-run — 2026-07-28

Fresh full-corpus run using the OpenAI model-role configuration:

| | GPT-4o baseline | optimized |
|---|---:|---:|
| model policy | GPT-4o for every image | GPT-4o-mini route; GPT-4o extraction |
| cost / document | $0.010309 | **$0.008449** |
| total cost | $0.484545 | **$0.397088** |
| API calls | 46 | 83 |
| input / output tokens | 111,774 / 20,255 | 456,731 / 16,828 |
| field accuracy | 59/91 (64.8%) | **66/91 (72.5%)** |
| routing accuracy | 39/39 | 39/39 |
| refusal accuracy | 5/8 | **8/8** |
| hallucinated records | 2 | **0** |

`ledger_*.jsonl` is the raw provider-token and computed-cost evidence.
`report_*.json` contains the full per-document scoring breakdown, and
`extractions_*.json` makes the field-accuracy computation independently
reproducible. Source images remain excluded from the repository.
