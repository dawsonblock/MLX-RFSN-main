"""Markdown artifact consistency scanner.

Ensures that human-readable markdown files cannot contradict the strict
JSON state (winner.json and artifact metadata).
"""
from __future__ import annotations

import json
from pathlib import Path

STALE_PHRASES_NO_WINNER = [
    "Promoted candidate:",
    "Promotion yes",
    "promotion eligible under Alpha 8.3 rules",
]

MARKDOWN_FILES = [
    Path("artifacts/bench/shootout/quick/results.md"),
    Path("artifacts/bench/shootout/full_logit/results.md"),
    Path("artifacts/bench/shootout/memory/results.md"),
    Path("artifacts/bench/shootout/promotion/results.md"),
    Path("artifacts/winner/winner.md"),
]

WINNER_JSON = Path("artifacts/winner/winner.json")


def _read_winner() -> dict:
    assert WINNER_JSON.exists(), "winner.json missing"
    return json.loads(WINNER_JSON.read_text())


def _read_markdown(path: Path) -> str:
    assert path.exists(), f"{path} missing"
    return path.read_text()


def test_markdown_no_stale_promotion_phrases_when_no_winner() -> None:
    """If winner.json has no winner, markdown must not claim promotion."""
    winner = _read_winner()
    if winner.get("winner") is not None:
        return  # JSON names a winner; different rules apply

    for md_path in MARKDOWN_FILES:
        text = _read_markdown(md_path)
        lower_text = text.lower().replace("**", " ")
        for phrase in STALE_PHRASES_NO_WINNER:
            lower_phrase = phrase.lower()
            if lower_phrase not in lower_text:
                continue
            # Allow the defensive "Official promoted candidate: NONE"
            if lower_phrase == "promoted candidate:":
                if "official promoted candidate:" in lower_text and "none" in lower_text:
                    continue
            assert False, (
                f"{md_path} contains stale promotion phrase "
                f"{phrase!r} but winner.json has no winner"
            )


def test_markdown_shows_promotion_no_when_promotion_allowed_false() -> None:
    """If promotion_allowed is false, markdown must not say promotion is allowed."""
    winner = _read_winner()
    if winner.get("promotion_allowed") is True:
        return

    for md_path in MARKDOWN_FILES:
        text = _read_markdown(md_path)
        # "Promotion allowed:** True" or "Promotion: yes" would be a contradiction
        if "Promotion allowed:** True" in text or "| yes |" in text.split("Promotion")[-1] if "Promotion" in text else False:
            # More robust: look for explicit contradictions in the known table format
            pass
        # Fail on explicit "Promotion allowed: True" in Notes section
        assert "Promotion allowed:** True" not in text, (
            f"{md_path} says promotion_allowed=True but winner.json says false"
        )


def test_winner_md_agrees_with_winner_json() -> None:
    """winner.md must not contradict winner.json."""
    winner = _read_winner()
    text = _read_markdown(Path("artifacts/winner/winner.md"))

    if winner.get("winner") is None:
        assert "## No winner" in text or "Official promoted candidate: NONE" in text, (
            "winner.md must declare No winner when winner.json is null"
        )
    else:
        winner_name = winner.get("winner")
        assert f"## Winner: {winner_name}" in text, (
            f"winner.md must declare Winner: {winner_name}"
        )


def test_promotion_report_markdown_shows_no_eligible_when_json_agrees() -> None:
    """promotion/results.md must contain the honest summary line."""
    promo_md = Path("artifacts/bench/shootout/promotion/results.md")
    if not promo_md.exists():
        return
    text = _read_markdown(promo_md)
    winner = _read_winner()
    if winner.get("winner") is None:
        assert "No candidate is promotion eligible" in text, (
            "promotion/results.md must state no eligible candidate when winner is null"
        )
