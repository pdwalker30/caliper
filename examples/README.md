# Caliper Examples

Reference folder layouts for Caliper evals. **Public-safe samples only** —
don't put proprietary content in here; it lives in the public OSS repo.

## Available samples

### `code_review/`

Five classic security-vulnerability snippets (SQL injection, path traversal,
hardcoded API key, reflected XSS, missing access control) reviewed by two
prompts (a terse baseline + a structured variant) against two judge-friendly
models. Four rubric dimensions: `finds_bug`, `severity_accuracy`,
`actionability`, `no_false_alarms`.

Total per pass: 2 × 2 × 5 × 2 = 40 cells = ~80 LLM calls. Under $1 at
cheap-model pricing.

```bash
caliper-check examples/code_review/eval_config.yaml   # readiness probe
caliper-eval  examples/code_review/eval_config.yaml   # run the eval
# (optional) flip human_review.enabled: true in the YAML, then:
caliper-calibrate examples/code_review/eval_config.yaml
```

---

## Folder convention

This is the convention for any Caliper eval — public-safe sample or
private (your-team's) data. The framework validates these shapes on load
via Pydantic; deviation fails fast with a clear error.

```
<your_eval>/
├── eval_config.yaml                ← the top-level driver
├── code_snippets/                  ← test_cases_dir (configurable name)
│   ├── <test_case_id_1>/
│   │   ├── <content_file>          ← any filename EXCEPT metadata.json
│   │   └── metadata.json           ← TestCaseMetadata
│   ├── <test_case_id_2>/
│   │   ├── ...
│   │   └── ...
│   └── ...
├── prompts/                        ← prompts_dir (configurable name)
│   ├── <prompt_id_1>/
│   │   ├── prompt.txt              ← fixed filename
│   │   └── metadata.json           ← PromptMetadata
│   └── ...
├── judge_prompts/                  ← judge_prompts_dir (configurable name)
│   └── <judge_id>/
│       ├── judge_prompt.txt        ← fixed filename
│       └── metadata.json           ← JudgePromptMetadata
└── rubrics/                        ← rubrics_dir (configurable name)
    └── <rubric_id>/
        └── rubric.json             ← Rubric (dimensions, weights, thresholds)
```

The folder names (`code_snippets/`, `prompts/`, etc.) are conventional —
all four are configurable via `*_dir` fields in `eval_config.yaml`. If your
domain isn't code review, name them whatever fits: `inputs/`, `queries/`,
`agent_scenarios/`, etc. The framework only cares about the field references
in the config.

The folder name IS the asset id. `<snippet_id>` becomes the Langfuse
DatasetItem id (idempotent across re-runs). `<prompt_id>` becomes the
`prompt:` tag and `prompt_id` in run_metadata. `<judge_id>` is referenced
by name from `eval_config.yaml: judge_prompt`.

### Fixed filenames

- **`metadata.json`** (in every sub-folder) — the structured metadata.
- **`prompt.txt`** (in each `prompts/<prompt_id>/`) — the prompt template.
  Use the literal token `{test_case}` where the snippet should be inserted.
- **`judge_prompt.txt`** (in each `judge_prompts/<judge_id>/`) — the judge
  template. Supports placeholders: `{test_case_input}`, `{test_case_description}`,
  `{expected_json}`, `{rubric_dimensions}`, `{llm_output}`, `{response_format_example}`.

### Content file naming

In `code_snippets/<id>/`, the content file can be named anything except
`metadata.json`. Use a name that reflects the content — `snippet.py`,
`snippet.go`, `vulnerable_query.sql`, etc. The loader picks up the single
non-`metadata.json` file as the content.

## Schemas in code

Canonical type definitions live in
[`caliper/schemas.py`](../caliper/schemas.py). When in doubt about a field,
read the Pydantic models — they're the source of truth, and they validate at
load time so typos surface immediately.

Key models:

- `TestCaseMetadata` — what goes in `code_snippets/<id>/metadata.json`
- `PromptMetadata` — what goes in `prompts/<id>/metadata.json`
- `JudgePromptMetadata` — what goes in `judge_prompts/<id>/metadata.json`
- `Rubric` + `RubricDimension` — the rubric block inside `TestCaseMetadata`
- `EvalConfig` — the top-level `eval_config.yaml` shape (with `RetryConfig`,
  `HumanReviewConfig`)

## Forking this sample for your own evals

The expected pattern when you have private data (work / client / proprietary):

```bash
# 1. Clone Caliper (the public OSS framework) — once per environment
git clone https://github.com/pdwalker30/caliper.git ~/dev/caliper
cd ~/dev/caliper && pip install -e ".[dev]"

# 2. Create a SEPARATE folder (or private git repo) for your evals
mkdir -p ~/dev/my-evals/code_review
cd ~/dev/my-evals/code_review

# 3. Copy the sample folder layout as a structural starting point
cp -r ~/dev/caliper/examples/code_review/* .

# 4. Replace contents with your own data:
#    - rm -rf code_snippets/*; drop in your snippets
#    - edit prompts/ to your prompt variants
#    - edit judge_prompts/rubric_judge_v1/judge_prompt.txt if needed
#    - edit eval_config.yaml (change `name`, `dataset_name`, `models`, etc.)

# 5. Run (paths are resolved relative to the config file)
caliper-eval ~/dev/my-evals/code_review/eval_config.yaml
```

The framework reads from wherever the `eval_config.yaml` lives. Your private
data never touches the public Caliper repo, but you still get framework
updates via `git pull` in `~/dev/caliper`.

## Rubric design tips

**Mix objective and subjective dimensions.** Objective dimensions (`finds_bug`,
`no_false_alarms`) are where the LLM judge is most reliable. Subjective ones
(`actionability`, `clarity`, `severity_accuracy`) are where human calibration
earns its keep — those are the dimensions where you'll see disagreement
between LLM and human scores, and those disagreements are the signal you
want for tuning the judge.

**Write dimension descriptions as instructions to the judge.** The judge
reads these verbatim. Phrasing like "Score 1.0 if X, 0.5 if Y, 0.0 if Z"
gives the judge clear anchors. Vague descriptions like "How good is the
explanation?" produce noisy scores.

**Put ground truth in `expected`.** Whatever the judge needs to score
fairly against the test case (the real vulnerability, the correct severity,
the canonical remediation steps) goes in `expected`. The judge prompt
templates it in via `{expected_json}` automatically.

**Share rubrics across snippets when possible.** Caliper's annotation queue
derives its score configs from the FIRST test case's rubric (current POC
assumption). If all snippets in an eval pass use the same rubric dimensions,
the human-review UI works cleanly across all of them.
