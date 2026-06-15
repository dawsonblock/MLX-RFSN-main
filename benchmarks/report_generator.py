"""RFSN benchmark report generator.

Converts a list of CandidateResults + Verdicts into:
  - A structured JSON artifact (machine-readable)
  - A Markdown report (human-readable)

Usage
-----
    from benchmarks.report_generator import ReportGenerator
    from benchmarks.schemas import CandidateResult
    from benchmarks.judge import Judge, Verdict

    generator = ReportGenerator(
        out_dir=Path("benchmarks/results"),
        report_dir=Path("benchmarks/reports"),
    )
    generator.write(
        candidates=results,
        baseline=baseline_result,
        verdicts=verdicts,
        run_tag="a1_run",
    )

Outputs
-------
    {out_dir}/{run_tag}_{timestamp}.json
    {out_dir}/{run_tag}_latest.json
    {report_dir}/{run_tag}_{timestamp}.md
    {report_dir}/{run_tag}_latest.md
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rfsn_v11.candidates.json_utils import dumps_json_strict  # noqa: E402

from .judge import Verdict, VerdictLabel
from .schemas import CandidateResult

# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Write JSON and Markdown reports for a benchmark run.

    Parameters
    ----------
    out_dir : Path
        Where to write .json artifacts.
    report_dir : Path
        Where to write .md reports.
    """

    def __init__(
        self,
        out_dir: Path | str = Path("benchmarks/results"),
        report_dir: Path | str = Path("benchmarks/reports"),
    ) -> None:
        self.out_dir = Path(out_dir)
        self.report_dir = Path(report_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        candidates: Sequence[CandidateResult],
        baseline: CandidateResult,
        verdicts: Sequence[Verdict],
        run_tag: str = "benchmark",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """Write JSON + Markdown.  Returns (json_path, md_path) for the latest files."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload = _build_json_payload(candidates, baseline, verdicts, timestamp, metadata or {})

        json_ts = self.out_dir / f"{run_tag}_{timestamp}.json"
        json_latest = self.out_dir / f"{run_tag}_latest.json"
        for p in (json_ts, json_latest):
            p.write_text(dumps_json_strict(payload, indent=2, default=str))

        md = _build_markdown(candidates, baseline, verdicts, run_tag, timestamp)
        md_ts = self.report_dir / f"{run_tag}_{timestamp}.md"
        md_latest = self.report_dir / f"{run_tag}_latest.md"
        for p in (md_ts, md_latest):
            p.write_text(md)

        return json_latest, md_latest

    def write_json_only(
        self,
        candidates: Sequence[CandidateResult],
        baseline: CandidateResult,
        verdicts: Sequence[Verdict],
        run_tag: str = "benchmark",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload = _build_json_payload(candidates, baseline, verdicts, timestamp, metadata or {})
        p = self.out_dir / f"{run_tag}_latest.json"
        p.write_text(dumps_json_strict(payload, indent=2, default=str))
        return p


# ---------------------------------------------------------------------------
# JSON payload
# ---------------------------------------------------------------------------

def _build_json_payload(
    candidates: Sequence[CandidateResult],
    baseline: CandidateResult,
    verdicts: Sequence[Verdict],
    timestamp: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    verdict_map: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        key = f"{v.candidate_name}_{v.prompt_id}"
        verdict_map[key] = {
            "label": v.label.value,
            "reason": v.reason,
            "quality_failures": v.quality_failures,
            "missing_required": v.missing_required,
            "improvement_met": v.improvement_met,
            "improvement_notes": v.improvement_notes,
        }

    return {
        "schema_version": "1.0",
        "timestamp": timestamp,
        "metadata": metadata,
        "baseline": baseline.to_dict(),
        "candidates": [c.to_dict() for c in candidates],
        "verdicts": verdict_map,
        "summary": _build_summary(candidates, verdicts),
    }


def _build_summary(
    candidates: Sequence[CandidateResult],
    verdicts: Sequence[Verdict],
) -> dict[str, Any]:
    counts: dict[str, int] = {label.value: 0 for label in VerdictLabel}
    for v in verdicts:
        counts[v.label.value] += 1
    return {
        "total_candidates": len(candidates),
        "total_verdicts": len(verdicts),
        "verdict_counts": counts,
        "promoted": [v.candidate_name for v in verdicts if v.label == VerdictLabel.PROMOTE],
        "rejected": [v.candidate_name for v in verdicts if v.label == VerdictLabel.REJECT],
        "regressions": [v.candidate_name for v in verdicts if v.label == VerdictLabel.REGRESSION],
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _build_markdown(
    candidates: Sequence[CandidateResult],
    baseline: CandidateResult,
    verdicts: Sequence[Verdict],
    run_tag: str,
    timestamp: str,
) -> str:
    model_ids = sorted({c.model_id for c in candidates} | {baseline.model_id})
    verdict_map = {(v.candidate_name, v.prompt_id): v for v in verdicts}

    def _f(v: float | None, fmt: str = ".3f") -> str:
        return format(v, fmt) if v is not None else "—"

    lines: list[str] = [
        f"# RFSN Benchmark Report: `{run_tag}`",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Models:** {', '.join(f'`{m}`' for m in model_ids)}  ",
        "",
        "## Summary",
        "",
    ]

    summary = _build_summary(candidates, verdicts)
    lines += [
        "| Verdict | Count |",
        "|---|---|",
    ]
    for label, count in summary["verdict_counts"].items():
        lines.append(f"| {label} | {count} |")

    if summary["promoted"]:
        lines += ["", f"**Promoted:** {', '.join(f'`{n}`' for n in summary['promoted'])}"]
    if summary["rejected"]:
        lines += [f"**Rejected:** {', '.join(f'`{n}`' for n in summary['rejected'])}"]
    if summary["regressions"]:
        lines += [f"**Regressions:** {', '.join(f'`{n}`' for n in summary['regressions'])}"]

    # Per-candidate section
    lines += ["", "## Candidate Results", ""]
    for c in candidates:
        v = verdict_map.get((c.candidate_name, c.prompt_id))
        verdict_str = v.label.value if v else "—"
        lines += [
            f"### `{c.candidate_name}` — prompt `{c.prompt_id}`",
            "",
            f"**Verdict:** `{verdict_str}`  ",
            f"**Model:** `{c.model_id}`  ",
            "",
            "#### Quality",
            "",
            "| metric | candidate | baseline |",
            "|---|---|---|",
            f"| logit_cosine | {_f(c.logit_cosine, '.5f')} | 1.00000 |",
            f"| top5_overlap | {_f(c.top5_overlap)} | 1.000 |",
            f"| top10_overlap | {_f(c.top10_overlap)} | 1.000 |",
            f"| attention_score_cosine | {_f(c.attention_score_cosine, '.5f')} | 1.00000 |",
            f"| attention_top5_overlap | {_f(c.attention_top5_overlap)} | 1.000 |",
            f"| perplexity_delta | {_f(c.perplexity_delta, '+.4f')} | +0.0000 |",
            f"| visible_output_drift | {_f(c.visible_output_drift_score)} | 0.000 |",
            "",
            "#### Memory",
            "",
            "| metric | value |",
            "|---|---|",
            f"| peak_memory_mb | {_f(c.peak_memory_mb, '.1f')} |",
            f"| kv_cache_memory_mb (dense est.) | {_f(c.kv_cache_memory_mb, '.1f')} |",
            f"| compressed_kv_memory_mb | {_f(c.compressed_kv_memory_mb, '.1f')} |",
            f"| metadata_memory_mb | {_f(c.metadata_memory_mb, '.1f')} |",
            f"| compression_factor | {_f(c.compression_factor, '.2f')}x |",
            f"| effective_bits/elem | {_f(c.effective_bits_per_kv_element, '.2f')} |",
            "",
            "#### Runtime",
            "",
            "| metric | value |",
            "|---|---|",
            f"| prefill_tps | {_f(c.prefill_tps, '.1f')} |",
            f"| decode_tps | {_f(c.decode_tps, '.1f')} |",
            f"| first_token_latency_ms | {_f(c.first_token_latency_ms, '.1f')} |",
            f"| total_latency_ms | {_f(c.total_latency_ms, '.1f')} |",
            f"| compression_time_ms | {_f(c.compression_time_ms, '.2f')} |",
            f"| decompression_time_ms | {_f(c.decompression_time_ms, '.2f')} |",
            "",
        ]
        if v and v.reason:
            lines += [f"**Reason:** {v.reason}", ""]
        if c.error:
            lines += [f"**Error:** `{c.error}`", ""]

    # Baseline section
    lines += [
        "## Dense Baseline Reference",
        "",
        "| metric | value |",
        "|---|---|",
        f"| model_id | `{baseline.model_id}` |",
        f"| decode_tps | {_f(baseline.decode_tps, '.1f')} |",
        f"| peak_memory_mb | {_f(baseline.peak_memory_mb, '.1f')} |",
        f"| kv_cache_memory_mb | {_f(baseline.kv_cache_memory_mb, '.1f')} |",
        "",
        "---",
        "*Generated by RFSN benchmark harness*",
        "",
    ]

    return "\n".join(lines)
