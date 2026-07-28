# Evidence

The committed, independently verified OpenAI benchmark is in
[`openai/`](openai/). It contains baseline and optimized structured outputs,
call-by-call JSONL token/cost ledgers, and the full scored report for all 47
input files.

Run `make verify` from the repository root to recompute its cost and accuracy
summary without an API key, images or network access.
