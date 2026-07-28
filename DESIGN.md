# DESIGN.md

What I built, what the numbers actually are, where it breaks, and what I'd do next.

---

## 1. The headline

Measured on all 47 starter images. Every figure below is a sum over
`out/ledger_*.jsonl`, written call-by-call from API-reported token counts.
Nothing is estimated.

| | naive baseline | optimised (default) |
|---|---|---|
| **cost / document** | **$0.0300** | **$0.0197** |
| cost / 1,000 documents | $30.03 | $19.71 |
| **field accuracy** (91 gold fields) | **80.2%** | **80.2%** |
| routing accuracy (39 real docs) | 100% | 100% |
| **refusal accuracy** (8 non-records) | **62.5%** | **100%** |
| **hallucinated records** | **2** | **0** |
| API calls | 46 | 83 |
| input / output tokens | 150,606 / 26,187 | 119,090 / 17,213 |

**1.52x cheaper (34% saved) at identical field accuracy, with the refusal
behaviour fixed.** The baseline invented invoice content from two images that
carry no record; the optimised pipeline invents none.

The accuracy number is deliberately *equal*, not better. I could have reported
a bigger cost win by shipping the `cheap` profile at $0.0165/doc, but that
scores 60.4% and the brief is explicit that cheaper-but-wrong doesn't count.

---

## 2. Architecture

Four stages. Each is a cost gate, and the earlier a document exits, the less it
costs.

```
        47 files
           │
  ┌────────▼─────────┐
  │ 0. PRE-FLIGHT    │   magic-byte sniff, truncation repair, EXIF rotate,
  │    $0 — no tokens│   perceptual-hash dedup, blur/exposure scoring
  └────────┬─────────┘   → 3 files rejected here for nothing
           │ 44
  ┌────────▼─────────┐
  │ 1. ROUTER        │   384px thumbnail → Haiku → class + subtype
  │    ~$0.0006/doc  │   → 5 object photos refused before any schema is shown
  └────────┬─────────┘
           │ 39
  ┌────────▼─────────┐
  │ 2. EXTRACT       │   per-class model, resolution and schema
  │    ~$0.023/doc   │   compact positional row encoding
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │ 3. VALIDATE      │   arithmetic, date sanity, cross-field contamination
  │    $0 — no tokens│   → escalate only what fails
  └──────────────────┘
```

### Where the 34% actually comes from

| lever | effect | mechanism |
|---|---|---|
| Free rejects + dedup | 3 of 47 files never reach a model | Content sniffing and perceptual hashing, zero tokens |
| Early refusal at router | 5 Opus calls → 5 Haiku thumbnail calls | Classifying is a much easier problem than reading, so buy it cheaper |
| Narrowed per-class schema | input tokens −20.9% | Baseline ships all three schemas on every image; we ship one |
| Compact row encoding | output tokens −34.3% | Positional arrays instead of repeated JSON keys |

Notably, **model downgrading contributed nothing to the shipped configuration.**
That is the most surprising result in this project and section 4 explains why.

---

## 3. Schema and routing

Four classes, chosen from the data rather than from an invoice ontology:
`work_report`, `vendor_bill`, `meter_reading`, `not_a_document`.

Every image returns the same envelope, whether it succeeded, was refused, or
failed — so a consumer never branches on success before it can branch on type:

```json
{
  "doc_id": "optera_doc_41",
  "doc_class": "meter_reading",
  "status": "extracted",          // extracted | refused | error
  "confidence": 0.92,
  "data": { "...": "..." },
  "refusal": null,                // populated instead of data when refused
  "quality_flags": ["underexposed"],
  "validation": { "passed": true, "issues": [], "severity": 0.0 },
  "provenance": { "calls": [...], "cost_usd": 0.0104, "stages": [...] }
}
```

**A refusal is a first-class result, not an error.** That distinction is the
whole point of the routing requirement: a battery photo must produce a
confident, structured "there is no record here", not an empty invoice.

Field naming follows the documents' own vocabulary (`bus_no`, `GSTIN`, DEF
litres) because the people who read this output speak that vocabulary.

The schema is the single source of truth: the extraction prompt is *generated*
from the field spec, so a schema change cannot leave a stale prompt behind.

---

## 4. Findings that changed the design

These are the things I did not expect, each of which came from a measurement
that contradicted an assumption.

### 4.1 JPEG quality is billed at zero — so compressing harder is a pure loss

Same image, same 1568px long edge:

| JPEG quality | payload | input tokens |
|---|---|---|
| 70 | 120 KB | **1585** |
| 82 | 160 KB | **1585** |
| 90 | 222 KB | **1585** |
| 95 | 326 KB | **1585** |

Vision billing is a function of pixel dimensions, not bytes. "Trim the image"
is one of the first things you reach for on a cost brief, and in its most
obvious form — compress harder — it saves *nothing* while measurably destroying
small printed text like GSTINs. I had shipped q=82 for invoices and it was
costing accuracy for no benefit. Only **downscaling** is a real lever.

### 4.2 Cost per class is unrelated to how hard the class is

| class | docs | share of spend | Opus vs cheap tier |
|---|---|---|---|
| work_report | 25 | **~82%** | +6.7 pts |
| vendor_bill | 10 | ~13% | **+21.4 pts** |
| meter_reading | 6 | ~3% | **+15.8 pts** |

The best model helps *most* on the two classes that are cheapest to run it on —
a dashboard photo is one small image and ~110 output tokens — and helps *least*
on the one class that dominates the bill. The intuitive policy (cheap models on
"easy" documents, expensive models on "hard" ones) is close to exactly wrong.

The shipped policy is therefore: **best model where it's cheap and errors are
financial; economise only where the tokens actually are.**

This is also why the `cheap` profile is a bad deal: it saves 19% and loses 20
accuracy points, because it cannot touch the work-report output tokens that
make up most of the bill.

### 4.3 My own sweep had a blind spot

`scripts/sweep_resolution.py` compared Haiku against Sonnet and concluded Haiku
was fine for meters and bills. It never tested Opus. Scoring the Opus baseline
per class showed Opus far ahead on exactly those classes. **A sweep can only
tell you about the axis you swept**; I nearly shipped a policy built on a
comparison that excluded the winning option.

### 4.4 Unused schema fields are not free — they are distractors

Asked for all ten meter fields at once, the model put the dashboard's
`AFE 3.4 km/l` (fuel *efficiency*) into `rate_per_litre`, and fabricated
`captured_at: 2024-08-21` from a clock reading `08:21 AM`.

Having the router emit a subtype (`odometer` / `dispenser`) and sending only the
fields that device can have removed the failure mode **and** cut prompt tokens.
`doc_38` went from `odometer_km: 320654` with a hallucinated date to a clean
`32065.4 / 1458.1 / null`. Cost and accuracy moved the same direction.

### 4.5 Escalation is only worth it if it is rare

My first escalation policy fired on **46%** of work reports and cost **+92%** of
total spend to buy **2.2 points**. The cause was `effective_confidence`
multiplying penalties, so one blank row in twenty-three pushed a document under
the threshold and bought a full second pass on the largest model.

Fixes: proportional weighting, quiet below 25% of a page, capped subtraction
instead of multiplication, threshold 0.70 → 0.55. Escalations fell 18 → 6.

The deeper point: **validation catches arithmetic and structure, never a
misread token.** An invoice whose lines sum to its total is checkable for free;
an odometer has no internal redundancy at all, which is precisely why the
meter class gets the best model rather than the cheapest.

### 4.6 Prompt caching matters less than expected for vision

Caching works (verified: 10,007 tokens written, then read back at 0.1x). But the
*image* dominates input tokens and every image is unique, so only the ~400-token
text prefix is cacheable. There is also a floor: Anthropic will not cache a
prefix under 2048 tokens (Haiku), so "trim the prompt" and "cache the prompt"
actively work against each other. My first caching test silently did nothing
because the prompt sat just under the floor.

---

## 5. Honest accounting

**Real numbers.** Both paths ran on the same 47 files, scored by the same scorer
against the same hand-written ground truth. Costs are summed from
API-reported token counts.

**Ground truth is mine, not the model's.** I read every image myself and
transcribed the gold values from zoomed crops. Grading a pipeline with its own
output would make the accuracy number meaningless. It also caught a case where
my own contact-sheet reading was wrong (`320265.4`) and the careful re-read
(`32065.4`) matched what the model said.

**Gold coverage is partial and unevenly distributed.**

| class | field gold | note |
|---|---|---|
| meter_reading | 6 of 6 | full |
| vendor_bill | 6 of 10 | full |
| work_report | 3 of 25 | **structural only** — depot, date, row count, vehicle codes |
| not_a_document | 5 of 5 | refusal only |

Verbatim transcription of 25 pages of trilingual handwriting was not achievable
in the time. So the single largest class — 82% of spend — is validated against
3 documents, and only structurally. **This is the weakest part of the
evaluation** and the first thing I'd fix.

**Measured noise.** `scripts/variance.py` runs the production policy over the
gold set three times: overall spread 1.1 pts (sd 0.6), work_report 3.3 pts. So
differences under ~1 point overall are not resolved by this gold set, and
work-report differences under ~3 points are not either. Two "regressions" I
initially attributed to policy turned out to be same-config variance.

**Batching not measured.** The gateway I ran against does not forward
`/v1/messages/batches`, so the standard 50% batch discount is **not** in any
number here. For a nightly WhatsApp backlog it applies cleanly and would take
$0.0197 → roughly $0.010/doc, but I am not going to claim a discount I could
not measure.

**Cost of building this.** Roughly $12 of inference across development,
sweeps, ablations and final runs — about 600 API calls.

---

## 6. Where it breaks

Ordered by how much they'd worry me in production.

1. **Handwritten vehicle codes.** The weakest real output. On `doc_03` the codes
   read `TAM17 → TCM17`, `MAC31 → MPC 31`. A format hint (3-letter prefix + digits,
   with the T/M and A/C/M/P confusions called out) helped substantially but did
   not solve it. These codes are the join key to the fleet database, so an error
   here silently attaches work to the wrong bus — worse than a missing row.

2. **Partly pre-printed dates.** Forms where the operator writes `08-06-` and the
   year `2026` is pre-printed get misread (`2020-06-22`). The prompt now
   addresses it; it still fails sometimes.

3. **Two GSTINs on one invoice.** On `doc_26` the model returned the *buyer's*
   GSTIN as `vendor_gstin`. Both are valid GSTINs on the same page, so no format
   check can catch it — it needs spatial reasoning about which block is the
   letterhead.

4. **Multi-page days.** `doc_20` has SR numbers running 12–16: it is page 2 of a
   day whose page 1 is a different image. Nothing stitches them. Downstream this
   looks like a day with five jobs starting at serial 12.

5. **Dedup is a policy question, not just a hash.** I collapse near-duplicates at
   Hamming ≤ 5. Two photos of the *same page* are a duplicate; two photos of
   *consecutive pages* of the same ruled form are not, and they look similar.
   Threshold 5 gets it right on all 47 here, but this will misfire eventually,
   and silently dropping a real document is the worst failure mode in the system.

6. **`not_a_document` is a grab-bag.** A battery, a blank page and an unreadable
   blur are all refused for different reasons but land in one class. A blank
   form probably deserves "document, no content" rather than "not a document".

7. **The 47 images are two clients.** Client A writes ruled forms with a Paldi/
   Vadaj/Sarangpur header; client B sends bills and dashboards. A third client
   with a different form will exercise paths nothing here has tested. The router
   fails *open* to the generalist path rather than dropping such a document, but
   that is damage control, not coverage.

---

## 7. What I'd do with another week

**Day 1–2 — fix the evaluation before touching the pipeline.**
Verbatim gold for 15 work reports, not 3. Right now the class carrying 82% of
spend is measured by 30 fields, and I cannot honestly distinguish a 5-point
change on it. Every optimisation below is unfalsifiable until this exists.

**Day 2–3 — targeted column re-read instead of whole-page escalation.**
Escalation currently re-runs an entire page on a bigger model to fix one bad
column. The `BUS NO` column is a narrow vertical strip: crop it, upscale it, and
re-read just that strip. A tall thin crop is a few hundred tokens rather than
~2,000, so it costs perhaps $0.003 against $0.03 for a full Opus pass, and it
attacks failure #1 directly. This is the single highest-value item.

**Day 3–4 — fleet-code resolution against a real vocabulary.**
The codes are a closed set in the customer's fleet database. Fuzzy-match every
extracted code against it, accept within edit distance 1, flag the rest for
review. Turns failure #1 from a silent wrong join into a bounded review queue,
and gives a genuine confidence signal to escalate on.

**Day 4 — spatial disambiguation for invoices.**
Ask for the letterhead block's GSTIN and the "Bill To" block's GSTIN as separate
fields, then assign by role. Fixes failure #3 without more pixels.

**Day 5 — batching and a real cache strategy.**
Route the nightly backlog through the Batch API (≈50% off, applies to the whole
bill) and expand the rulebook past the 2048-token cache floor with genuinely
useful few-shot examples. Caching inverts the usual tradeoff: above the floor,
a *richer* prompt is nearly free after the first call, so it buys accuracy at
about a tenth of list price.

**Ongoing — an actual review loop.**
The pipeline emits calibrated confidence and structured validation issues but
nothing consumes them. Route low-confidence documents to a human, capture the
corrections, and feed them back as few-shot examples. On a stream this
repetitive — the same depots, the same vendors, the same 40 buses every day —
that loop will beat any amount of prompt engineering.

---

## 8. Things I deliberately did not do

- **No fine-tuning.** With 47 images it would memorise the starter set, and the
  brief says evaluation is on unseen images.
- **No local OCR pre-pass.** Tesseract on Gujarati handwriting at this quality
  adds a dependency and a failure mode without removing the vision call.
- **No overfitting to the 47.** The bus-code hint describes the *shape* of a code,
  not a whitelist of the codes present here, so an unseen prefix still parses. The
  same discipline applies to the depot names, which are never enumerated in a prompt.
- **No made-up batch discount** in the headline, for the reason in section 5.
