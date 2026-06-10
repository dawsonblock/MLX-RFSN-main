"""Artifact helpers for benchmark output generation.

These utilities build honest markdown tables and export winner metadata.
They live in rfsn_v11 (not benchmarks/) so tests can import them without
package-shadowing issues from tests/benchmarks/__init__.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARTIFACTS_ROOT = Path("artifacts/bench/shootout")
WINNER_DIR = Path("artifacts/winner")


def _build_honest_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Build the honest benchmark table required by Plan B."""
    lines: list[str] = ["# KV Shootout Results\n"]
    if not rows:
        lines.append("No results.\n")
        return "\n".join(lines)

    # Header
    lines.append("## Honest Benchmark Table\n")
    lines.append(
        "| Candidate | Status | Speed (tps) | Memory (ratio) | "
        "Logit gate | Real cache used | Promotion |"
    )
    lines.append(
        "|-----------|--------|-------------|----------------|"
        "------------|-----------------|-----------|"
    )

    for row in rows:
        # Skipped artifact rows (e.g. SKIPPED_NO_MLX_LM)
        if row.get("status", "").startswith("SKIPPED"):
            status_label = row.get("status", "SKIPPED")
            reason = row.get("reason", "")
            lines.append(
                f"| **{status_label}** | — | — | — | "
                f"— | no | no |"
            )
            if reason:
                lines.append(f"| *Reason:* {reason} | | | | | | |")
            continue
        if "note" in row:
            lines.append(f"| {row['note']} | | | | | | |")
            continue
        name = row.get("name", "")
        status = row.get("candidate_status", "—")
        speed = (
            f"{row.get('tokens_per_sec', 0):.2f}"
            if row.get("tokens_per_sec") is not None
            else "—"
        )
        mem = (
            f"{row.get('size_ratio', 0):.3f}"
            if row.get("size_ratio") is not None
            else "baseline"
        )
        gate = row.get("gate_status", "—")
        real_cache = "yes" if row.get("real_cache_used") else "no"
        promo = "yes" if row.get("promotion_eligible") else "no"
        lines.append(
            f"| {name} | {status} | {speed} | {mem} | "
            f"{gate} | {real_cache} | {promo} |"
        )

    lines.append("")
    return "\n".join(lines)


def _export_winner(
    rows: list[dict[str, Any]], models_tested: list[str]
) -> None:
    """Export winner artifacts when a candidate is promotion eligible."""
    eligible = [r for r in rows if r.get("promotion_eligible")]
    WINNER_DIR.mkdir(parents=True, exist_ok=True)

    if not eligible:
        winner_data = {
            "winner": None,
            "status": "NO_PROMOTION_ELIGIBLE_CANDIDATE",
            "reason": (
                "No candidate has full logit, real cache, and memory proof."
            ),
        }
    else:
        # Pick the one with best tokens/sec among eligible
        best = max(eligible, key=lambda r: r.get("tokens_per_sec") or 0)
        winner_data = {
            "winner": best.get("name"),
            "status": "PROMOTED",
            "reason": (
                "Passed full logit gate and reduced KV memory with "
                "equal or better decode speed."
            ),
            "models_tested": models_tested,
            "artifacts": {
                "full_logit": str(
                    ARTIFACTS_ROOT / "full_logit" / "results.json"
                ),
                "memory": str(
                    ARTIFACTS_ROOT / "memory" / "results.json"
                ),
                "promotion": str(
                    ARTIFACTS_ROOT / "promotion" / "results.json"
                ),
            },
        }

    json_path = WINNER_DIR / "winner.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(winner_data, fh, indent=2)

    md_path = WINNER_DIR / "winner.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Winner Report\n\n")
        if winner_data["winner"] is None:
            fh.write("## No winner\n\n")
            fh.write(f"**Status:** {winner_data['status']}\n\n")
            fh.write(f"**Reason:** {winner_data['reason']}\n")
        else:
            fh.write(f"## Winner: {winner_data['winner']}\n\n")
            fh.write(f"**Status:** {winner_data['status']}\n\n")
            fh.write(f"**Reason:** {winner_data['reason']}\n\n")
            fh.write(
                "**Models tested:** "
                f"{', '.join(winner_data['models_tested'])}\n"
            )

    notes_path = WINNER_DIR / "integration_notes.md"
    with notes_path.open("w", encoding="utf-8") as fh:
        fh.write("# Integration Notes\n\n")
        if winner_data["winner"] is None:
            fh.write("No candidate promoted. Integration notes pending.\n")
        else:
            fh.write(
                "The promoted candidate "
                f"`{winner_data['winner']}` can be integrated via:\n\n"
            )
            fh.write("```python\n")
            fh.write(
                "from rfsn_v11.integrations.cache_policy "
                "import create_cache_policy\n"
            )
            fh.write(
                f'policy = create_cache_policy("{winner_data["winner"]}")\n'
            )
            fh.write("# model.generate(prompt, cache_policy=policy)\n")
            fh.write("```\n")

    print(f"  Wrote {json_path}, {md_path}, {notes_path}")
