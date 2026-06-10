"""Markdown artifacts must never contradict JSON artifacts.

Humans read markdown first. If markdown says "Promotion: yes" while JSON
says ``promotion_allowed: false``, the release is misleading.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rfsn_v11.candidates.artifact_utils import _build_honest_markdown_table


def _read_payload(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        return raw
    return {"metadata": {}, "results": raw}


def test_markdown_does_not_claim_promotion_when_json_disallows_it() -> None:
    """If JSON says promotion_allowed=false, markdown must say Promotion no."""
    for mode in ("quick", "full_logit", "memory", "promotion"):
        json_path = Path(f"artifacts/bench/shootout/{mode}/results.json")
        md_path = Path(f"artifacts/bench/shootout/{mode}/results.md")
        if not json_path.exists() or not md_path.exists():
            continue
        payload = _read_payload(json_path)
        meta = payload.get("metadata", {})
        if meta.get("promotion_allowed") is False:
            md_text = md_path.read_text(encoding="utf-8")
            # Check that no row ends with "| yes |" (Promotion column)
            for line in md_text.splitlines():
                if line.startswith("| "):
                    if line.startswith("| Candidate"):
                        continue
                    parts = [p.strip() for p in line.split("|")]
                    # Promotion is last column before trailing empty
                    if len(parts) >= 3 and parts[-2] == "yes":
                        raise AssertionError(
                            f"{md_path} claims promotion=yes but "
                            f"{json_path} says promotion_allowed=false"
                        )


def test_markdown_winner_matches_winner_json() -> None:
    """winner.md must not name a candidate when winner.json says null."""
    winner_json = Path("artifacts/winner/winner.json")
    winner_md = Path("artifacts/winner/winner.md")
    assert winner_json.exists()
    assert winner_md.exists()
    data = json.loads(winner_json.read_text(encoding="utf-8"))
    md_text = winner_md.read_text(encoding="utf-8")
    if data.get("winner") is None:
        assert "## Winner:" not in md_text, (
            "winner.md has a 'Winner:' heading but winner.json says null"
        )
    else:
        assert f"## Winner: {data['winner']}" in md_text, (
            f"winner.md missing Winner: {data['winner']}"
        )


def _promotion_column_values(md: str) -> list[str]:
    """Extract the Promotion column value from every data row.

    Skips header rows, separator rows, note rows, and summary rows.
    """
    vals: list[str] = []
    for line in md.splitlines():
        if not line.startswith("| "):
            continue
        if any(k in line for k in ("Candidate", "---", "note", "Summary")):
            continue
        parts = [p.strip() for p in line.split("|")]
        # A real data row has at least: name, status, speed, mem,
        # gate, real_cache, promo + 2 empty ends = 9 parts
        if len(parts) >= 9:
            vals.append(parts[-2])
    return vals


def test_pass_no_promote_renders_as_promotion_no() -> None:
    """Rows with PASS_NO_PROMOTE must show Promotion no in markdown."""
    rows = [
        {
            "name": "mlx_lm_baseline",
            "candidate_status": "CONTROL",
            "tokens_per_sec": 50.0,
            "size_ratio": 1.0,
            "gate_status": "PASS_NO_PROMOTE",
            "real_cache_used": True,
            "promotion_eligible": False,
        }
    ]
    md = _build_honest_markdown_table(rows, promotion_allowed=False)
    promo_vals = _promotion_column_values(md)
    assert all(v == "no" for v in promo_vals), (
        f"Expected all Promotion=no, got {promo_vals}"
    )


def test_promotion_allowed_false_blocks_promoted_text() -> None:
    """Even if a row claims promotion_eligible, markdown must say no."""
    rows = [
        {
            "name": "fake_winner",
            "candidate_status": "BASELINE",
            "tokens_per_sec": 100.0,
            "size_ratio": 0.5,
            "gate_status": "PASS",
            "real_cache_used": True,
            "promotion_eligible": True,
        }
    ]
    md = _build_honest_markdown_table(rows, promotion_allowed=False)
    promo_vals = _promotion_column_values(md)
    assert all(v == "no" for v in promo_vals)
    assert "No candidate is promotion eligible" in md


def test_promotion_allowed_true_respects_row_flag() -> None:
    """When promotion_allowed=True, eligible rows show yes."""
    rows = [
        {
            "name": "real_winner",
            "candidate_status": "BASELINE",
            "tokens_per_sec": 100.0,
            "size_ratio": 0.5,
            "gate_status": "PASS",
            "real_cache_used": True,
            "promotion_eligible": True,
        }
    ]
    md = _build_honest_markdown_table(rows, promotion_allowed=True)
    assert "| yes |" in md
    assert "No candidate is promotion eligible" not in md
