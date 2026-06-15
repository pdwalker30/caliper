# Human Review Setup

Caliper supports human-in-the-loop calibration of the LLM judge via Langfuse
Annotation Queues. This document covers both the **automatic** setup path
(Caliper creates everything via the REST API) and the **manual fallback** path
(create score configs + queue in the Langfuse UI when the API doesn't cooperate).

## The flow

```
1. First eval pass with human_review.enabled=true
   * Caliper creates score configs + queue (auto) OR uses pre-existing ones
   * Eval runs to completion, writing LLM scores to every trace
   * Sampled traces (default: 2 per Run, stratified) added to the queue
                              v
2. Humans annotate via Langfuse UI -> Annotation Queues
   * Each queue item shows the trace + a 1-5 score per rubric dimension
   * Human scores attach to the trace with source=ANNOTATION
                              v
3. Calibration report
   $ caliper-calibrate path/to/eval_config.yaml
   * Pairs LLM scores (source=API) with human scores (source=ANNOTATION)
   * Derives pass flags + overall (both judge and human) from the numeric scores
   * Reports MAE, Pearson r, Spearman rho per numeric dimension
   * Reports Cohen's kappa + confusion matrix for the derived pass scores
```

## Configuration in `eval_config.yaml`

```yaml
human_review:
  enabled: true
  queue_name: "code-review-calibration"
  sample_strategy: stratified    # or "random" | "all"
  samples_per_run: 2             # for stratified
  sample_rate: 0.15              # for random (ignored when stratified)
  auto_create: true              # try REST API to create queue + configs
```

## Automatic path (try this first)

With `auto_create: true`, Caliper will:

1. List existing Score Configs in the Langfuse project.
2. For each rubric dimension in your test cases, find-or-create:
   * `<dim_name>` (NUMERIC, range 1-5)
3. Find-or-create the queue named in `queue_name`, binding those per-dimension
   configs. **No `__pass` or `overall` configs** — the human enters only the
   dimension scores; pass/fail and overall are derived at calibration time.
4. Sample N traces per Dataset Run and add each to the queue.

Run `python -m caliper.check_stack examples/code_review/eval_config.yaml`
BEFORE the first eval to verify the API is reachable with your credentials.

## Manual fallback path

If the auto-create REST calls fail (auth scope, Langfuse version drift, etc.),
set `auto_create: false` in the config and create the queue + configs in the
UI by hand. One-time setup:

### 1. Create Score Configs (Settings -> Score Configs -> New)

For each rubric dimension in your test cases, create ONE config whose name
EXACTLY MATCHES the dimension name emitted by the LLM judge (Caliper uses
`dim.name` as the score name). Do **not** create `__pass` or `overall`
configs — those are derived, not hand-entered:

| Name              | Data type  | Range / categories |
| ----------------- | ---------- | ------------------ |
| `finds_bug`       | NUMERIC    | min 1, max 5       |
| `actionability`   | NUMERIC    | min 1, max 5       |
| ... one per dim   | NUMERIC    | min 1, max 5       |

### 2. Create the Annotation Queue (Annotation Queues -> New)

* Name: matches `human_review.queue_name` in your `eval_config.yaml`
* Score configs: bind ALL of the configs you just created
* Description: optional, e.g. "Caliper human review for X campaign"

### 3. Run Caliper with `auto_create: false`

Caliper will look up the existing queue by name and start enqueueing sampled
traces.

## What humans see in the UI

Going to **Annotation Queues -> \[your queue\] -> Next item**:

* The trace (with the LLM's output and the judge's reasoning visible)
* The judge's rendered prompt on the `judge` generation, so you can see exactly
  what the judge was asked before you score
* One 1-5 score per rubric dimension (that's all you enter — pass/fail and the
  overall score are computed for you at calibration time)
* Optional comment field per score

Submit. Repeat. After ~15 minutes of annotation work for a small calibration
sample, run `python -m caliper.calibration <config.yaml>` and read the report.

## Reading the calibration report

The header table looks like:

```
Dimension                       N      MAE        r      rho    kappa
------------------------------------------------------------------------
actionability                  12    0.417    0.872    0.853     -
actionability__pass            12      -        -        -    0.667
finds_bug                      12    0.333    0.921    0.934     -
finds_bug__pass                12      -        -        -    0.833
no_false_alarms                12    0.667    0.654    0.601     -    <-- weak
no_false_alarms__pass          12      -        -        -    0.412   <-- weak
overall                        12    0.288    0.889    0.901     -
overall__pass                  12      -        -        -    0.750
```

MAE is on the 1-5 score scale (~0.4 = within half a point on average). The
`__pass` rows (and the human `overall`) are **derived** at calibration time
from the per-dimension numeric scores — only numeric scores are written to
Langfuse, for both the judge and the human.

Then per-dimension confusion matrices for the `__pass` boolean scores:

```
  Confusion for finds_bug__pass:  TP=8  FP=1  TN=2  FN=1
```

### Interpretation guide

| Metric     | "Trust the LLM judge" | "Judge prompt needs work" |
| ---------- | --------------------- | ------------------------- |
| MAE (1-5)  | < 0.5                 | > 0.75                    |
| Pearson r  | > 0.85                | < 0.70                    |
| Spearman rho | > 0.85              | < 0.70                    |
| Cohen's k  | > 0.70                | < 0.50                    |
| Confusion  | TP/TN dominate        | One-sided (lots of FP or FN) |

If one dimension consistently underperforms (e.g., `no_false_alarms` in the
example above), the fix is usually one of:

1. **Sharpen the rubric description** for that dimension — humans and the
   judge may be interpreting it differently.
2. **Add a worked example** of a borderline case to the judge prompt.
3. **Tighten the pass threshold** if FPs dominate (judge is too lenient) or
   loosen it if FNs dominate (judge is too strict).

Re-run the eval + a small human pass; if metrics improve, you've improved
the judge. That's the calibration loop.
