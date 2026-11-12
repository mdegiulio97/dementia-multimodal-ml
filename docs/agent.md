# The local analysis agent

The agent answers questions about this project's data and results without inventing numbers.
A local model, served by [Ollama](https://ollama.com), chooses which tool to call. The tools do
the computing and return a value together with its uncertainty and provenance. The model only
words the answer.

## Why tools and not a fine-tuned model

A language model trained on participant data can memorise it, which is both a privacy problem
and a breach of the ADNI Data Use Agreement. A model that recalls numbers from its weights also
gets them wrong. Here the model is never trained on the data: everything it "knows" about the
data reaches it through the tool results.

## Tools

| Tool | Returns | Guards |
|---|---|---|
| `describe_data` | sources, cohort sizes, variables, available results | none needed |
| `cohort_summary(filters, variable)` | n, mean and SD (one decimal) of a baseline variable in the MCI cohort; quartiles only from n ≥ 20 | categorical filters only (APOE4, converted, age band); refuses N < 10; refuses a cell whose sibling (one filter dropped) holds 1-9 people, so two answers cannot be subtracted down to a small group; refuses participant IDs |
| `predict_conversion_risk(age, mmse, apoe4, ..., horizon_months)` | conversion probability from the full Cox model, with a 95% interval from 50 bootstrap refits | warns when inputs fall outside the training 1st-99th percentiles (never min/max, which are individual values), caps the horizon at the end of follow-up, lists imputed features |
| `explain_prediction(...)` | per-feature contributions to the log hazard (exact linear SHAP) | none needed |
| `get_result(name)` | stored results: `conversion_metrics`, `trajectories`, `mr_candidates`, `eeg_metrics` | unknown names refused |

## Abstention

If no tool returned a value, because nothing was called, the request was refused or the tool
was unknown, the agent abstains and replaces whatever the model wrote with a fixed message.
If the answer contains a number that matches no tool value, it is withheld too, and the raw
tool results are shown instead. A number counts as matched when it equals a tool value (or
that value as a percentage) at the precision written, or when it appears in the question.
The check works on numbers, so a wrong claim written in words would not be caught.

## Evaluation

`src/dementia_ml/agent/golden.yaml` holds 12 questions with the tool that must answer each one,
including a request for a single participant that must be refused.

- **In CI:** a scripted backend checks routing, refusals, provenance and abstention.
- **Locally:** `uv run pytest -m ollama` checks that the live model calls the expected tool in
  at least 80% of the questions. The measured rate is in [RESULTS.md](RESULTS.md).

## MCP

`uv run --extra mcp dementia-mcp` serves the same tools over stdio, so any Model Context
Protocol client gets the same refusals and provenance. MCP clients are often driven by cloud
models, so ADNI mode is refused unless started with `--kind adni --allow-adni-to-client`,
which is meant only for a client backed by a local model.
