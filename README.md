# Caliper

A generic, metadata-driven LLM evaluation framework.

Caliper runs the Cartesian product of `(prompt × model × test_case × iteration)`,
sends each output to a configurable judge, and writes structured traces +
per-dimension scores to [Langfuse](https://langfuse.com) — so the
"which prompt/model combo wins?" question has an out-of-the-box answer.

## Status

Early development. Not yet ready for general use.

## Philosophy

- **Test cases own the rubric.** The judge is a generic executor; what
  it's judging *against* lives on the test case's metadata. One framework,
  N eval types (code review, agent tool-call verification, agent outcome
  correctness, …) — selected by a single `eval_type` discriminator.
- **Configs in folders, framework in code.** Adding a new eval type or a
  new test case means dropping files into a folder — not editing the
  framework. The framework is the runner; the configs are the work.
- **OSS dependencies all the way down.** [Langfuse](https://langfuse.com)
  for trace + score storage. [LiteLLM](https://github.com/BerriAI/litellm)
  for multi-vendor LLM calls. Everything runs in your own Docker.

## Quickstart

### 1. Bring up the local stack

```bash
cp .env.example .env
./scripts/generate-secrets.sh   # prints values to paste into .env
# Fill in upstream LLM provider keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...)
docker compose up -d
docker compose ps               # all services should report "healthy"
```

The stack includes:

| Service       | URL / Port                    | Purpose                                  |
| ------------- | ----------------------------- | ---------------------------------------- |
| Langfuse UI   | http://localhost:3000         | Traces, datasets, scores, experiments    |
| LiteLLM Proxy | http://localhost:4000         | Unified gateway to OpenAI / Anthropic /… |
| Postgres      | localhost:5432                | Langfuse operational data                |
| ClickHouse    | localhost:8123                | Langfuse trace storage                   |
| MinIO console | http://localhost:9091         | S3-compatible blob store                 |
| Redis         | localhost:6379                | Langfuse ingestion queue                 |

### 2. First-time Langfuse setup

Open http://localhost:3000, sign up (local-only account), create a project,
copy the public + secret API keys into `.env` as `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY`.

### 3. Install Caliper

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 4. Run an evaluation

```bash
python -m caliper.eval_runner examples/code_review/eval_config.yaml
```

(Sample eval lands in Milestone 4 — for now, point at any eval_config.yaml
following the [schema](caliper/schemas.py).)

## How it works

For each `(prompt, model, test_case, iteration)` Caliper:

1. Opens a Langfuse trace bound to a **Dataset Run** named after the
   `(prompt, model)` pair, so the Experiments view aggregates per combo.
2. Calls the model under test via the LiteLLM proxy, capturing the LLM call
   as a child **Generation** observation with token usage attached.
3. Sends the output to the **rubric judge** — a generic LLM-as-judge that
   reads the rubric from `metadata.json` and emits per-dimension scores
   plus reasoning.
4. Attaches one Langfuse **Score** per rubric dimension, one boolean
   `__pass` score per dimension, and an aggregate `overall` + `overall__pass`
   on the parent trace.

Result: the Langfuse Datasets -> Runs comparison view shows mean score and
pass rate per `(prompt, model)` combo, with drilldowns into individual traces.
That's the "which combo wins?" answer, materialized.

## License

[MIT](LICENSE) — use it however you like.
