# Caliper

A generic, metadata-driven LLM evaluation framework.

Caliper runs the Cartesian product of `(prompt × model × test_case × iteration)`,
sends each output to a configurable LLM judge, optionally samples a slice for
human calibration via [Langfuse](https://langfuse.com) Annotation Queues, and
reports human-vs-LLM judge agreement so you know how much to trust the LLM
judge before you run it at scale.

**Status:** early development. Not yet ready for general use.

---

## What's in the box

```
caliper/                            ← framework (importable library)
├── schemas.py                      Pydantic data contracts
├── runner.py                       run_eval()  — the Cartesian loop
├── calibration.py                  agreement metrics (MAE, r, ρ, κ)
├── diagnostics.py                  stack readiness probes
├── dataset_bootstrap.py            folder → Langfuse Dataset
├── litellm_client.py               proxy client with retry/backoff
├── idempotency.py                  hash-based cell skip logic
├── human_review.py                 annotation queue + score config HTTP client
├── judges/
│   ├── base.py                     JudgeAdapter protocol
│   └── rubric_judge.py             generic LLM-as-judge driven by metadata
└── cli/                            ← CLI shims (thin entry points, no logic)
    ├── eval.py                     caliper-eval [--force]
    ├── calibrate.py                caliper-calibrate
    └── check.py                    caliper-check

docs/
└── human-review-setup.md           full setup guide + manual UI fallback

examples/
└── code_review/                    sample folder layout (M4)
    ├── code_snippets/
    ├── prompts/
    ├── judge_prompts/
    └── eval_config.yaml

scripts/
└── generate-secrets.sh             one-shot generator for Langfuse secrets

docker-compose.yml                  Langfuse v3 self-hosted + LiteLLM proxy
litellm_config.yaml                 model routing config (edit to add vendors)
.env.example                        every env var, documented
```

---

## The four phases

Caliper splits evaluation work into four discrete phases. **Phase 1 is one-time
setup per environment. Phases 2-4 are the recurring eval cycle.**

```
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 1: Setup           (once per env, framework-assisted)        │
│  Phase 2: Eval pass       (per cycle, framework)                    │
│  Phase 3: Human review    (per cycle, humans in Langfuse UI)        │
│  Phase 4: Calibration     (per cycle, framework)                    │
└──────────────────────────────────────────────────────────────────────┘
```

### Phase 1 — Setup (one-time per environment)

Bring up the stack, create accounts/keys, install Caliper, verify it can
reach everything.

```bash
# 1a. Generate secrets and fill in .env
cp .env.example .env
./scripts/generate-secrets.sh        # prints values — copy into .env
# Edit .env, add upstream LLM API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...)

# 1b. Bring up the stack
docker compose up -d
docker compose ps                    # all services should report "healthy"

# 1c. Open http://localhost:3000 in a browser
#     - Sign up (local-only account)
#     - Create a project
#     - Settings -> API Keys -> Create new key
#     - Paste public + secret keys into .env as LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY

# 1d. Install Caliper
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 1e. Sanity-check the whole stack
caliper-check examples/code_review/eval_config.yaml
```

Expect to see a clean readiness report:

```
[PASS]  Langfuse reachable     http://localhost:3000 responded 200
[PASS]  Langfuse credentials   Basic auth accepted on REST API
[PASS]  LiteLLM proxy reachable http://localhost:4000 responded 200
[PASS]  LiteLLM master key     Present and well-formed
[PASS]  Eval config parse      examples/code_review/eval_config.yaml
[PASS]  test_cases_dir exists  /path/to/code_snippets
... (etc)
All probes passed cleanly. You're ready to run an eval pass.
```

If anything FAILs, fix it before continuing — see [Troubleshooting](#troubleshooting).

### Phase 2 — Eval pass (per evaluation cycle)

Run the Cartesian, write traces + LLM judge scores to Langfuse, enqueue
sampled traces for humans.

```bash
caliper-eval examples/code_review/eval_config.yaml
```

What happens:

1. **Dataset bootstrap.** Folder of test cases becomes a Langfuse Dataset
   (idempotent on item id — re-runs upsert, never duplicate).
2. **Cartesian loop.** For each `(prompt × model × snippet × iteration)`, the
   framework opens a Langfuse trace linked to a Dataset Run named
   `<campaign>__<prompt_id>__<model>__<timestamp>`.
3. **LLM call.** Routed through the LiteLLM proxy. Token usage attached to
   the generation observation; Langfuse computes cost server-side.
4. **Judge call.** The generic `RubricJudge` templates the judge prompt
   with the rubric metadata, calls the judge model, parses structured JSON
   back into per-dimension scores.
5. **Scores attached.** One Score per dimension, one boolean `__pass`
   score per dimension, plus `overall` and `overall__pass`, all on the
   parent trace.
6. **Sample for humans.** If `human_review.enabled=true` in the config,
   the sampler picks N traces per Run and enqueues them via the Langfuse
   Annotation Queue REST API.

After this completes, the Langfuse UI shows the Experiments comparison view
populated with aggregated scores per `(prompt, model)` combo. That's the
"which combo wins?" answer.

### Phase 3 — Human review (per cycle, async)

Humans annotate sampled traces in the Langfuse UI. **No Caliper command for
this phase** — it's all UI work.

1. Open Langfuse: **http://localhost:3000**
2. Sidebar → **Annotation Queues** → click the queue named in `eval_config.yaml`
3. Click **Annotate Next Item** to walk through queued traces
4. For each trace, score every rubric dimension (slider for NUMERIC, toggle
   for BOOLEAN `__pass`) and submit
5. Repeat until the queue is empty (or you've done enough for calibration —
   ~12-20 samples is plenty for a useful agreement estimate)

Human scores attach to the trace with `source=ANNOTATION`. They sit alongside
the LLM scores (`source=API`) on the same trace — calibration will pair them
in phase 4.

### Phase 4 — Calibration report (per cycle, after some humans annotate)

Read paired LLM + human scores, compute agreement metrics, print + CSV-write
a report.

```bash
caliper-calibrate examples/code_review/eval_config.yaml
```

Output:

```
========================================================================
CALIPER CALIBRATION REPORT
========================================================================

Dimension                       N      MAE        r      rho    kappa
------------------------------------------------------------------------
actionability                  12    0.087    0.872    0.853     -
actionability__pass            12      -        -        -    0.667
finds_bug                      12    0.052    0.921    0.934     -
finds_bug__pass                12      -        -        -    0.833
overall                        12    0.071    0.889    0.901     -
overall__pass                  12      -        -        -    0.750

  Confusion for finds_bug__pass:  TP=8  FP=1  TN=2  FN=1

Guidance: r/rho > 0.85 and kappa > 0.70 = trust the LLM judge.
          Anything materially lower means the judge prompt or rubric
          needs revisiting before treating LLM scores as authoritative.

[caliper] CSV written to results/calibration-code-review-eval-20260608-141522.csv
```

If agreement is good (r/ρ > 0.85, κ > 0.70), you can trust the LLM judge for
larger matrices without humans in the loop. If it's bad, edit the judge
prompt template (or rubric thresholds), re-run phases 2-4. That's the loop.

Full calibration setup guide (with manual UI fallback for queue creation
when the REST auto-create can't run): [docs/human-review-setup.md](docs/human-review-setup.md).

---

## Configuration reference

All eval configs are YAML matching the `EvalConfig` Pydantic model in
[caliper/schemas.py](caliper/schemas.py). Minimal example:

```yaml
name: "code-review-eval"
dataset_name: "code-eval-snippets-v1"
test_cases_dir: "code_snippets"
prompts_dir: "prompts"
judge_prompts_dir: "judge_prompts"
judge_prompt: "rubric_judge_v1"      # folder name under judge_prompts/
judge_model: "gpt-4o-mini"           # alias from litellm_config.yaml
models:                              # aliases from litellm_config.yaml
  - "gpt-4o-mini"
  - "claude-haiku"
iterations: 3                        # per (prompt, model, test_case) cell
concurrency: 10                      # parallel cells in flight (override via --concurrency N)

human_review:
  enabled: true
  queue_name: "code-review-calibration"
  sample_strategy: stratified        # or "random" | "all"
  samples_per_run: 2
  auto_create: true                  # try REST-API to create queue + configs

# Retry / backoff for every LLM call (model under test AND judge).
# Defaults are tuned for typical hosted-LLM rate limits.
retry:
  max_attempts: 5
  initial_wait_seconds: 2.0
  max_wait_seconds: 60.0
  exponential_base: 2.0
  jitter_seconds: 1.0
  retry_on_statuses: [408, 425, 429, 500, 502, 503, 504]

# Hash-based opt-in idempotency. When true, Caliper computes a SHA-256 hash
# per Cartesian cell over (campaign, prompt text, judge prompt text, model,
# judge model, snippet content, expected, rubric, iteration) and skips any
# cell whose hash already appears on a trace in this campaign.
#
# Edit a prompt -> hash changes -> cell re-runs. No version field to bump.
# Override for one run with: caliper-eval --force <config.yaml>
idempotent: false

# Optional. Map LiteLLM-returned model names to Langfuse-recognized names
# so Langfuse's built-in pricing map can compute cost. Needed when models
# are hosted behind providers Langfuse doesn't know natively (Databricks,
# internal fine-tunes, custom Bedrock endpoints, etc.).
langfuse_model_mapping:
  "databricks-llama-3-70b-instruct": "llama-3-70b-instruct"
  "internal-fine-tuned-judge": "gpt-4o"
```

`extra_run_metadata` (optional) lets you stamp every Dataset Run with
arbitrary key/values for richer filtering in the Experiments view.

---

## Philosophy

- **Test cases own the rubric.** The judge is a generic executor; what
  it's judging *against* lives on the test case's `metadata.json`. One
  framework, N eval types (code review, agent tool-call verification, agent
  outcome correctness, …) — selected by a single `eval_type` discriminator.
- **Configs in folders, framework in code.** Adding a new eval type or
  test case means dropping files into a folder — not editing the framework.
- **OSS dependencies all the way down.** Langfuse for trace + score storage.
  LiteLLM proxy for multi-vendor LLM calls. Everything runs in your own
  Docker.

---

## Troubleshooting

### `caliper-check` says **Langfuse reachable: FAIL**

The Langfuse stack isn't responding on the URL in `LANGFUSE_HOST`. Verify:

```bash
docker compose ps                    # langfuse-web should be "healthy"
docker compose logs langfuse-web     # check for startup errors
curl http://localhost:3000/api/public/health
```

Common causes: stack not started, port 3000 in use, browser-side caching of
the URL (use `curl` to confirm independently).

### `caliper-check` says **Langfuse credentials: FAIL**

`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are missing or wrong. Go to
the Langfuse UI → your project → Settings → API Keys → create a new key,
copy both values into `.env`, source `.env` (or restart your shell).

### `caliper-check` says **LiteLLM proxy reachable: FAIL**

The LiteLLM container isn't responding. Verify:

```bash
docker compose ps                    # litellm should be "healthy"
docker compose logs litellm
curl http://localhost:4000/health/liveliness
```

Common causes: missing `LITELLM_MASTER_KEY` in `.env`, malformed
`litellm_config.yaml`, all upstream provider API keys empty (proxy starts
but can't route).

### `caliper-eval` runs but says `WARN: failed to enqueue trace ... for human review`

The Langfuse Annotation Queue REST API returned an error. The eval pass
will still complete — LLM scores are all written. Only the human-review
sampling is skipped.

Most common cause: the queue or score-config endpoints differ between
Langfuse versions. Look at the WARN output — it includes the full URL
and HTTP status from the failed request. If the URL is wrong for your
Langfuse version, edit `caliper/human_review.py` and adjust the
`/api/public/...` paths to match. Or: set `human_review.auto_create: false`
in your config, create the queue manually in the Langfuse UI under
Annotation Queues, and Caliper will look it up by name.

See [docs/human-review-setup.md](docs/human-review-setup.md) for the
manual UI path.

### `caliper-calibrate` reports `no traces found tagged campaign:...`

Either no eval pass has run yet for this campaign name, or the campaign
name in your config doesn't match the campaign tag on the traces. Verify
in Langfuse UI: Traces → filter by tag → look for `campaign:<your-name>`.

### Judge returns non-JSON, calibration shows zero pairs

The judge model isn't producing parseable JSON. Two likely fixes:

1. **Switch to a more-instruction-following judge model.** Cheaper judge
   models sometimes wander. Try `gpt-4o-mini` or `claude-sonnet` instead.
2. **Tighten the judge prompt.** The default template asks for JSON;
   include an explicit `"You MUST return ONLY a JSON object matching this
   schema: {...}"` line.

### Eval pass is hitting rate limits (HTTP 429) repeatedly

Caliper retries every LLM call automatically with exponential backoff + jitter
per the `retry:` config block. If you're seeing sustained 429s in the
retry-progress lines on stderr, you're hitting a real quota wall — bump
`retry.max_attempts` and `retry.max_wait_seconds`, or use a cheaper / lower-
tier model for the matrix's first cycle, or batch your eval over multiple
shorter passes. Single 429s with successful retries are healthy and expected.

### Eval pass skipped lots of cells — am I missing data?

The end-of-pass summary line is `ran=X skipped=Y failed=Z`. If `skipped` is
non-zero, you have `idempotent: true` in your config and Caliper found
existing traces with matching cell hashes. That's intentional — you're not
missing data, you're re-using prior runs.

To force a fresh pass without editing the config: `caliper-eval --force <config>`.
To verify what the hash protects: check `metadata.cell_hash` on existing
traces in Langfuse UI.

### I edited prompt.txt but Caliper still skipped some cells

That's a real bug — hash-based idempotency should detect any change in
`prompt.txt`. Verify:

1. The file you edited is actually under `prompts/<prompt_id>/prompt.txt`
   in the folder referenced by `eval_config.yaml: prompts_dir`.
2. The change is saved (no editor buffer / encoding issue).
3. The campaign name in your config matches the existing traces' campaign tag.

If all three check out and you're still getting unexpected skips, run with
`--force` to confirm fresh runs work, then file an issue with the prompt
file's SHA-256 + the trace's `metadata.cell_hash` so the hash compute can
be debugged.

### A single cell failed (FAIL in the matrix) — what happens next?

The matrix continues. Per-cell failure isolation: one cell exhausting retries
or hitting an unrecoverable error gets logged and skipped. The end-of-pass
summary lists every failed cell with the error message. The other cells
complete normally — you don't waste 80 cells' worth of API calls because
cell #41 had a transient issue.

To re-run only the failed cells later: set `idempotent: true`, leave the
successful traces in place, re-run. Caliper will skip the successful cells
(hashes match) and re-attempt only the missing ones.

### Langfuse traces show cost as $0.00 (or no cost at all)

Langfuse computes server-side cost by looking up the generation's `model`
field against its built-in pricing table (plus any custom models you've
added under Settings → Models). If the model name LiteLLM returned isn't
in either table, cost stays at $0.

Two fixes:

**(A) Map the name in `eval_config.yaml`** — preferred for portability:

```yaml
langfuse_model_mapping:
  "databricks-llama-3-70b-instruct": "llama-3-70b-instruct"
  "internal-fine-tuned-judge": "gpt-4o"
```

The LEFT side is what LiteLLM returns (visible on the generation's `model`
field in the Langfuse UI). The RIGHT side is a name Langfuse already knows.

**(B) Register the model in Langfuse UI** — preferred when no equivalent
public model exists for pricing:

Settings → Models → Add custom model. Use the LiteLLM-returned name
exactly. Set input + output prices per 1M tokens.

Don't do both for the same model — pick one. Option A is portable across
Langfuse instances (e.g., self-hosted at home vs. self-hosted at work);
option B is more accurate when your custom model has unique pricing.

### Eval pass burns a lot of API budget unexpectedly

Sanity-check your Cartesian: `prompts × models × test_cases × iterations`
calls to the model under test + the same count of calls to the judge
(judge is called once per trace). For 2 × 3 × 5 × 3 = 90 traces, that's
180 LLM calls per pass. Use cheap models for the judge (`gpt-4o-mini`,
`claude-haiku`) — see the [insight on the judge cost knob](docs/human-review-setup.md).

---

## License

[MIT](LICENSE) — use it however you like.
