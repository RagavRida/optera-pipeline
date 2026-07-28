# Design notes

## What I built

This is a small OpenAI-only document pipeline for a mixed image inbox. It emits
a stable JSON envelope for every input and treats refusal as a valid outcome:
an object photo returns `status: refused` and `data: null`, never invented rows.

The baseline deliberately makes one full-resolution GPT-4o call per image with
all schemas available. The optimized path is:

1. Free preflight: detect real image bytes, repair tolerable truncation, rotate
   EXIF images, identify near-duplicates and flag poor lighting/blur.
2. GPT-4o-mini router on a 384px thumbnail: choose `work_report`,
   `vendor_bill`, `meter_reading` or `not_a_document`; identify meter subtype.
3. GPT-4o extractor only for genuine documents, with the one relevant schema
   and compact positional rows for repeated work-log data.
4. Free validation of dates, numeric fields and bill arithmetic.

## Measured result

All values below come from API-reported token usage in the committed JSONL
ledgers, over the same 47 files on 2026-07-28.

| | Baseline | Optimized |
|---|---:|---:|
| Cost/document | $0.010309 | **$0.008449** |
| Total cost | $0.484545 | **$0.397088** |
| API calls | 46 | 83 |
| Input / output tokens | 111,774 / 20,255 | 456,731 / 16,828 |
| Field accuracy (91 gold fields) | 59/91 | **66/91** |
| Routing accuracy (39 real documents) | 39/39 | 39/39 |
| Refusal accuracy (8 non-records) | 5/8 | **8/8** |
| Hallucinated records | 2 | **0** |

The optimized result is 18.04% cheaper (1.22x) and scores better. `make verify`
recomputes this summary from `results/openai/` without an API key or network.

## What reduced cost without reducing accuracy

- Preflight avoided model calls for an HTML error file and two near-duplicates.
- Routing was done at 384px with GPT-4o-mini, then the five object photos were
  refused before extraction.
- The extractor sees only the schema for its routed class; it does not receive
  irrelevant invoice, meter and work-report fields on every call.
- Repeated mechanic-log rows use compact positional arrays in the response and
  are expanded into normal JSON locally.
- Prompt text is kept stable and concise. The OpenAI API reported cached prompt
  tokens in both benchmark ledgers; the ledger prices those separately rather
  than charging them as full-price input.

I removed unmeasured batch extraction and cross-provider experiments from the
submission. They were not necessary for the verified OpenAI claim, and a
cheaper request shape is not an optimization until it is re-scored for accuracy.

## Where it breaks

- The 91-field gold set is small and uneven: all six meter readings and six
  bills are covered, but only three of 25 handwritten work reports have
  structural field labels. This is the biggest limitation.
- Handwritten vehicle identifiers remain error-prone, especially visually
  similar characters. A wrong identifier can be worse than a null value.
- Bills with buyer and seller GSTIN values need spatial role awareness; a
  format validator cannot tell which valid GSTIN belongs to the vendor.
- Near-duplicate detection is deliberately conservative but could still confuse
  two visually similar consecutive pages in a larger corpus.
- The result is one 47-image evaluation, not a claim of generalization to a
  new fleet, handwriting style or document layout.

## With another week

1. Transcribe full field-level gold for more mechanic logs before optimizing
   anything else.
2. Add targeted crop re-reads for uncertain vehicle-code columns instead of
   repeating a full page.
3. Resolve vehicle identifiers against a customer fleet vocabulary and send
   unmatched values to review.
4. Split invoice seller and buyer blocks spatially before assigning GSTIN.
5. Measure OpenAI Batch API and multi-image requests on labelled data; ship
   them only if cost falls with the same or better accuracy.
