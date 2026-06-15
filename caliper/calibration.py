"""Judge calibration — compute human-vs-LLM agreement metrics.

Library entry points. Read scores by REST, pair LLM-source with
ANNOTATION-source, compute MAE / Pearson r / Spearman rho / Cohen's kappa /
confusion matrices, print + CSV-write a report.

The CLI shim that wires these together against a YAML config lives in
`caliper.cli.calibrate`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any

from caliper.human_review import LangfuseAnnotationClient
from caliper.schemas import Rubric, aggregate_scores

# ---------------------------------------------------------------------------
# Score fetching
# ---------------------------------------------------------------------------


@dataclass
class ScorePair:
    trace_id: str
    name: str
    llm_value: float | None = None
    human_value: float | None = None
    llm_comment: str = ""
    human_comment: str = ""

    @property
    def paired(self) -> bool:
        return self.llm_value is not None and self.human_value is not None


def fetch_all_scores_for_campaign(
    client: LangfuseAnnotationClient,
    campaign: str,
    limit_per_page: int = 100,
) -> list[dict[str, Any]]:
    """Fetch every score in the project, filtered by campaign tag at the
    score-trace level. Walks all pages."""
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        # NOTE: the /scores endpoint doesn't directly filter on tags. We pull
        # all and filter client-side. For very large projects we'd add date
        # filters; for a POC this is fine.
        resp = client._client.get(
            "/api/public/scores",
            params={"page": page, "limit": limit_per_page},
        )
        client._raise_with_context(resp, "list_scores")
        data = resp.json()
        out.extend(data.get("data", []))
        meta = data.get("meta", {})
        if page >= meta.get("totalPages", 1):
            break
        page += 1
    return out


def fetch_traces_for_campaign(
    client: LangfuseAnnotationClient,
    campaign: str,
) -> dict[str, dict[str, Any]]:
    """Returns {trace_id: trace} for every trace tagged campaign:<campaign>."""
    out: dict[str, dict[str, Any]] = {}
    page = 1
    target_tag = f"campaign:{campaign}"
    while True:
        resp = client._client.get(
            "/api/public/traces",
            params={"page": page, "limit": 100, "tags": target_tag},
        )
        client._raise_with_context(resp, "list_traces")
        data = resp.json()
        for trace in data.get("data", []):
            out[trace["id"]] = trace
        meta = data.get("meta", {})
        if page >= meta.get("totalPages", 1):
            break
        page += 1
    return out


def build_score_pairs(
    scores: list[dict[str, Any]],
    trace_ids: set[str],
) -> dict[str, list[ScorePair]]:
    """Group scores by name -> list of ScorePair, only for traces in this campaign.

    Source taxonomy in Langfuse:
        API        — from Caliper's LLM judge via the SDK
        ANNOTATION — from a human via the UI annotation queue
        EVAL       — from an LLM-as-a-judge eval triggered inside Langfuse
    """
    # Index by (trace_id, name) so we can merge LLM + human pairs
    pairs: dict[tuple[str, str], ScorePair] = {}
    for s in scores:
        trace_id = s.get("traceId")
        if trace_id not in trace_ids:
            continue
        name = s["name"]
        key = (trace_id, name)
        if key not in pairs:
            pairs[key] = ScorePair(trace_id=trace_id, name=name)
        pair = pairs[key]
        source = s.get("source", "")
        value = _coerce_score_value(s.get("value"), s.get("dataType"))
        comment = s.get("comment") or ""
        if source == "ANNOTATION":
            pair.human_value = value
            pair.human_comment = comment
        else:
            # Treat API + EVAL as "machine" — we mostly care about humans vs
            # not-humans for calibration purposes.
            pair.llm_value = value
            pair.llm_comment = comment

    by_name: dict[str, list[ScorePair]] = defaultdict(list)
    for pair in pairs.values():
        by_name[pair.name].append(pair)
    return by_name


def derive_pass_and_overall(
    by_name: dict[str, list[ScorePair]],
    rubric: Rubric,
) -> None:
    """Reconstruct pass flags + overall from the per-dimension numeric scores.

    Caliper writes NUMERIC scores only — no boolean `__pass` columns and no
    human-entered `overall`. This rebuilds, for BOTH the judge (llm) and human
    sides, the data calibration reports on:

      - `<dim>__pass`   : score >= dim.pass_threshold
      - `overall`       : aggregate of the dimension scores (the judge's is also
                          emitted; only filled here if absent / for the human)
      - `overall__pass` : overall >= rubric.overall_pass_threshold

    Both sides derive identically, so they stay directly comparable. Mutates
    `by_name` in place; derived values land on the same ScorePair per
    (name, trace) so the two sides pair up.
    """
    dim_names = [d.name for d in rubric.dimensions]
    thresholds = {d.name: d.pass_threshold for d in rubric.dimensions}

    llm_dims: dict[str, dict[str, float]] = defaultdict(dict)
    human_dims: dict[str, dict[str, float]] = defaultdict(dict)
    for dim_name in dim_names:
        for p in by_name.get(dim_name, []):
            if p.llm_value is not None:
                llm_dims[p.trace_id][dim_name] = p.llm_value
            if p.human_value is not None:
                human_dims[p.trace_id][dim_name] = p.human_value

    def _pair(name: str, trace_id: str) -> ScorePair:
        for p in by_name.setdefault(name, []):
            if p.trace_id == trace_id:
                return p
        pair = ScorePair(trace_id=trace_id, name=name)
        by_name[name].append(pair)
        return pair

    thr = rubric.overall_pass_threshold
    for trace_id in set(llm_dims) | set(human_dims):
        ld = llm_dims.get(trace_id, {})
        hd = human_dims.get(trace_id, {})

        for dim_name in dim_names:
            pp = _pair(f"{dim_name}__pass", trace_id)
            if dim_name in ld:
                pp.llm_value = float(ld[dim_name] >= thresholds[dim_name])
            if dim_name in hd:
                pp.human_value = float(hd[dim_name] >= thresholds[dim_name])

        overall_pair = _pair("overall", trace_id)
        overall_pass_pair = _pair("overall__pass", trace_id)

        # Judge overall numeric is emitted; prefer it, else aggregate when the
        # judge scored every dimension. Derive the pass flag from whichever.
        llm_overall = overall_pair.llm_value
        if llm_overall is None and all(d in ld for d in dim_names):
            llm_overall = aggregate_scores(rubric, ld)
            overall_pair.llm_value = llm_overall
        if llm_overall is not None:
            overall_pass_pair.llm_value = float(llm_overall >= thr)

        # Human overall is always derived (humans enter only dimension scores),
        # and only when they scored EVERY dimension so it's comparable.
        if all(d in hd for d in dim_names):
            human_overall = aggregate_scores(rubric, hd)
            overall_pair.human_value = human_overall
            overall_pass_pair.human_value = float(human_overall >= thr)


def _coerce_score_value(value: Any, data_type: str | None) -> float | None:
    """Boolean scores come back as 0/1 numerics in Langfuse — coerce uniformly."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Statistics — hand-rolled to keep deps minimal (no numpy/scipy)
# ---------------------------------------------------------------------------


def mean_absolute_error(pairs: list[ScorePair]) -> float | None:
    paired = [p for p in pairs if p.paired]
    if not paired:
        return None
    return sum(abs(p.llm_value - p.human_value) for p in paired) / len(paired)  # type: ignore[operator]


def pearson_r(pairs: list[ScorePair]) -> float | None:
    paired = [p for p in pairs if p.paired]
    if len(paired) < 2:
        return None
    xs = [p.llm_value for p in paired]
    ys = [p.human_value for p in paired]
    mx = sum(xs) / len(xs)  # type: ignore[arg-type]
    my = sum(ys) / len(ys)  # type: ignore[arg-type]
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))  # type: ignore[operator]
    denx = sqrt(sum((x - mx) ** 2 for x in xs))  # type: ignore[operator]
    deny = sqrt(sum((y - my) ** 2 for y in ys))  # type: ignore[operator]
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def spearman_rho(pairs: list[ScorePair]) -> float | None:
    paired = [p for p in pairs if p.paired]
    if len(paired) < 2:
        return None
    llm_ranks = _ranks([p.llm_value for p in paired])  # type: ignore[arg-type]
    human_ranks = _ranks([p.human_value for p in paired])  # type: ignore[arg-type]
    rebuilt = [
        ScorePair(trace_id=p.trace_id, name=p.name, llm_value=lr, human_value=hr)
        for p, lr, hr in zip(paired, llm_ranks, human_ranks)
    ]
    return pearson_r(rebuilt)


def _ranks(xs: list[float]) -> list[float]:
    """Rank values from smallest = 1. Ties get averaged ranks."""
    indexed = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def cohen_kappa(pairs: list[ScorePair]) -> tuple[float, dict[str, int]] | None:
    """For BOOLEAN scores (pass flags). Returns (kappa, confusion_counts)."""
    paired = [p for p in pairs if p.paired]
    if not paired:
        return None
    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for p in paired:
        llm = bool(round(p.llm_value or 0))
        human = bool(round(p.human_value or 0))
        if llm and human:
            confusion["tp"] += 1
        elif llm and not human:
            confusion["fp"] += 1
        elif not llm and not human:
            confusion["tn"] += 1
        else:
            confusion["fn"] += 1
    n = len(paired)
    po = (confusion["tp"] + confusion["tn"]) / n
    p_llm_yes = (confusion["tp"] + confusion["fp"]) / n
    p_hum_yes = (confusion["tp"] + confusion["fn"]) / n
    pe = p_llm_yes * p_hum_yes + (1 - p_llm_yes) * (1 - p_hum_yes)
    if pe == 1:
        return 1.0, confusion
    return (po - pe) / (1 - pe), confusion


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class DimensionReport:
    name: str
    paired_count: int
    mae: float | None = None
    pearson: float | None = None
    spearman: float | None = None
    kappa: float | None = None
    confusion: dict[str, int] = field(default_factory=dict)


def report(by_name: dict[str, list[ScorePair]]) -> list[DimensionReport]:
    out = []
    for name in sorted(by_name.keys()):
        pairs = by_name[name]
        paired = sum(1 for p in pairs if p.paired)
        rpt = DimensionReport(name=name, paired_count=paired)
        if name.endswith("__pass"):
            kappa_result = cohen_kappa(pairs)
            if kappa_result:
                rpt.kappa, rpt.confusion = kappa_result
        else:
            rpt.mae = mean_absolute_error(pairs)
            rpt.pearson = pearson_r(pairs)
            rpt.spearman = spearman_rho(pairs)
        out.append(rpt)
    return out


def print_report(reports: list[DimensionReport]) -> None:
    print()
    print("=" * 72)
    print("CALIPER CALIBRATION REPORT")
    print("=" * 72)
    print()
    print(f"{'Dimension':<28} {'N':>5} {'MAE':>8} {'r':>8} {'rho':>8} {'kappa':>8}")
    print("-" * 72)
    for r in reports:
        mae = f"{r.mae:.3f}" if r.mae is not None else "    -   "
        pr = f"{r.pearson:.3f}" if r.pearson is not None else "    -   "
        sp = f"{r.spearman:.3f}" if r.spearman is not None else "    -   "
        ka = f"{r.kappa:.3f}" if r.kappa is not None else "    -   "
        print(f"{r.name:<28} {r.paired_count:>5} {mae:>8} {pr:>8} {sp:>8} {ka:>8}")
    print()
    for r in reports:
        if r.confusion:
            c = r.confusion
            print(
                f"  Confusion for {r.name}:  "
                f"TP={c['tp']}  FP={c['fp']}  TN={c['tn']}  FN={c['fn']}"
            )
    print()
    print("Guidance: r/rho > 0.85 and kappa > 0.70 = trust the LLM judge.")
    print("          Anything materially lower means the judge prompt or rubric")
    print("          needs revisiting before treating LLM scores as authoritative.")
    print()


def write_csv(reports: list[DimensionReport], pairs_by_name: dict[str, list[ScorePair]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["dimension", "trace_id", "llm_value", "human_value", "llm_comment", "human_comment"]
        )
        for name, pairs in sorted(pairs_by_name.items()):
            for p in pairs:
                if p.paired:
                    w.writerow(
                        [name, p.trace_id, p.llm_value, p.human_value, p.llm_comment, p.human_comment]
                    )


