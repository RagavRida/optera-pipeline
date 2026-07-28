# Optera document pipeline

Turns a heterogeneous WhatsApp inbox of Indian bus-fleet paperwork — handwritten
mechanic logs, printed vendor bills, dashboard photos, and photos of objects
that aren't documents at all — into clean structured JSON, as cheaply as
possible without losing accuracy.

**1.52x cheaper than a naive one-call-per-image baseline, at identical field
accuracy, with hallucinated records reduced from 2 to 0.**

| | naive baseline | this pipeline |
|---|---|---|
| cost / document | $0.0300 | **$0.0197** |
| cost / 1,000 documents | $30.03 | **$19.71** |
| field accuracy | 80.2% | **80.2%** |
| routing accuracy | 100% | 100% |
| refusal accuracy | 62.5% | **100%** |
| hallucinated records | 2 | **0** |

Measured on all 47 starter images. The immutable evidence is committed under
[`results/`](results/): every number is summed from its call-by-call JSONL
ledger using API-reported token counts. New runs write the same artifacts to
`out/`; see [DESIGN.md](DESIGN.md) for methodology, error bars and known
failure modes.

---

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY
make run                    # baseline vs optimised on images/, with scoring
```

`make run` is `python3 run.py --mode both`. It prints the full cost and accuracy
comparison and writes ledgers, per-document JSON and a report to `out/`.

Only dependency is Pillow. API calls use the standard library.

### No API key?

You can still verify every published number without client images or any model
credential:

```bash
make verify
```

This recomputes the cost and accuracy summary from the committed extraction
outputs, JSONL token/cost ledgers, and hand-written ground truth. It makes no
network calls. It is evidence verification, **not** offline inference on new
images; live `make run` requires either `ANTHROPIC_API_KEY` (the measured
configuration) or a separately evaluated provider/model configuration.

### Other commands

```bash
python3 run.py                          # optimised only (default profile)
python3 run.py --mode baseline          # the naive 1x only
python3 run.py --profile cheap          # a cheaper, less accurate operating point
python3 run.py --input /path/to/images  # any directory
python3 run.py --limit 5                # smoke test on 5 files
python3 run.py --batch                  # opt-in multi-image experiment; compare score first

make sweep                              # model x resolution grid → out/sweep.json
make variance                           # how much accuracy moves when nothing changes
```

### Operating points

There is no single right cost/accuracy tradeoff, so the frontier is measured
and selectable with `--profile`:

| profile | $/doc | vs baseline | field accuracy |
|---|---|---|---|
| `cheap` | $0.0165 | 1.81x | 60.4% |
| `balanced` | $0.0179 | 1.68x | 70.3% |
| **`accurate`** (default) | **$0.0197** | **1.52x** | **80.2%** |

`accurate` is the default because it *dominates*: only 19% more than `cheap`
while scoring 20 points higher. On this corpus the spend is concentrated in
work-report output tokens, which no profile changes — so downgrading models
buys very little and costs a lot. That finding is discussed in DESIGN.md §4.2.

### Batching is deliberately not in the headline

The code includes an opt-in multi-image extractor for bounded meter and invoice
outputs. It records one shared API call and allocates its actual cost across the
participating documents. It is **off by default** because the committed 47-image
accuracy run predates this optimization; run `make run` with and without
`--batch` on a labelled corpus before promoting it. The published 1.52x result
is therefore conservative rather than a claimed-but-unverified batch saving.

---

## How it works

```
0. PRE-FLIGHT   $0, no tokens   content sniffing, truncation repair, EXIF rotation,
                                perceptual-hash dedup, blur/exposure scoring
1. ROUTER       ~$0.0006/doc    384px thumbnail → cheapest model → class + subtype;
                                object photos refused before any schema is shown
2. EXTRACT      ~$0.023/doc     per-class model, resolution and schema;
                                compact positional row encoding
3. VALIDATE     $0, no tokens   arithmetic, date sanity, cross-field contamination;
                                escalate only what fails
```

On the starter set, stage 0 removes 3 of 47 files for nothing — an HTML error
page saved with a `.jpg` extension, and two WhatsApp re-sends — and stage 1
refuses 5 object photos before they ever see an extraction prompt.

### Document classes

| class | what it is |
|---|---|
| `work_report` | Mechanic's daily log; ruled form or notebook page, English/Hindi/Gujarati |
| `vendor_bill` | Supplier invoice, cash memo or bill of supply |
| `meter_reading` | Instrument cluster or DEF/fuel dispenser display |
| `not_a_document` | An object or scene carrying no record — **refused, never extracted** |

### Output

Every image produces the same envelope regardless of outcome:

```json
{
  "doc_id": "optera_doc_41",
  "doc_class": "meter_reading",
  "status": "extracted",
  "confidence": 0.92,
  "data": {
    "reading_type": "def_dispenser",
    "amount_rs": 3199.76, "litres": 43.24,
    "rate_per_litre": 74.00, "urea_concentration_pct": 32.50
  },
  "refusal": null,
  "quality_flags": [],
  "validation": { "passed": true, "issues": [], "severity": 0.0 },
  "provenance": { "cost_usd": 0.0104, "stages": ["route:...", "extract:..."] }
}
```

A refused image carries the same shape with `data: null` and a populated
`refusal` — a structured "there is no record here", which is the point of the
routing requirement.

---

## Layout

```
run.py                      one-command entrypoint
optera/
  config.py                 pricing, per-class policy, profiles  ← the tradeoffs live here
  pricing.json              per-MTok rates; nothing else hardcodes a price
  providers.py              provider-agnostic vision client (Anthropic + OpenAI)
  preflight.py              stage 0, the free gate
  router.py                 stage 1, cheap classification
  extract.py                stage 2, per-class extraction + escalation
  validate.py               stage 3, deterministic checks
  schemas.py                canonical schemas; the prompt is generated from these
  prompts.py                rulebook + per-class task text
  ledger.py                 append-only token/cost log
  score.py                  accuracy scoring against ground truth
groundtruth/
  routing.json              hand-labelled class for all 47 files
  fields.json               hand-transcribed gold values for 15 files
scripts/
  sweep_resolution.py       model x resolution grid
  variance.py               run-to-run noise floor
out/                        ledgers, results, reports (generated)
```

## Configuration

Model roles are environment-overridable, so the whole comparison can be re-run
against different vendors without touching code:

```bash
OPTERA_BASELINE_MODEL=claude-opus-4-5-20251101
OPTERA_ROUTER_MODEL=claude-haiku-4-5-20251001
OPTERA_CHEAP_MODEL=gpt-4o-mini        # OpenAI models work too
OPTERA_ESCALATION_THRESHOLD=0.55
```

Anything listed in `optera/pricing.json` can fill any role. Adding a model means
adding its rates there, so no cost can enter the ledger unaccounted for.

## Ground truth

Hand-labelled by reading every image at full resolution — deliberately not
produced by any model in this pipeline, since grading a pipeline with its own
output would make the accuracy number meaningless.

Routing labels cover all 47 files. Field-level gold covers 15: all 6 meter
readings, 6 of 10 vendor bills, and 3 work reports (structural only — depot,
date, row count and vehicle codes). That coverage gap is the evaluation's main
limitation and is discussed honestly in DESIGN.md §5.
