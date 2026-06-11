"""Test teacher-forced logit capture token alignment.

The teacher-forced loop must feed each generated token to predict the NEXT
token in the known sequence.  If the loop is off-by-one (e.g. feeds
gen_ids[1:] instead of gen_ids[:-1]), the captured logits will correspond
to the wrong tokens and the comparison is invalid.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rfsn_v11.candidates.teacher_forcing import (
    expected_logprob_count,
    forced_input_tokens_for_generated,
)


class FakeTokenizer:
    """Minimal tokenizer for testing."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}
        for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
            self.vocab[ch] = i
            self.inv[i] = ch
        self.vocab[" "] = 26
        self.inv[26] = " "

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv.get(i, "?") for i in ids)


class FakeCache:
    """Placeholder cache object."""

    def __init__(self) -> None:
        self.state = []


class FakeModel:
    """Fake model that records every input token it receives."""

    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.layers: list[Any] = [object()]  # at least one layer
        self.received_tokens: list[int] = []
        self.call_count = 0

    def __call__(self, x: Any, cache: Any = None) -> np.ndarray:
        # x is expected to be an array-like with shape (1, seq_len)
        self.call_count += 1
        token_ids = list(x.flatten())
        self.received_tokens.extend(token_ids)
        # Return fake logits: vocab_size = 27, return uniform logits
        vocab = len(self.tokenizer.vocab)
        seq_len = len(token_ids)
        return np.zeros((1, seq_len, vocab))

    def make_cache(self) -> list[FakeCache]:
        return [FakeCache()]


def _simulate_teacher_forced_loop(
    prompt: str,
    target_text: str,
    tokenizer: FakeTokenizer,
    model: FakeModel,
) -> list[int]:
    """Simulate the CORRECT teacher-forced logic."""
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target_text)

    if (
        len(target_ids) >= len(prompt_ids)
        and target_ids[: len(prompt_ids)] == prompt_ids
    ):
        gen_ids = target_ids[len(prompt_ids):]
    else:
        gen_ids = target_ids

    # Prefill
    model(np.array(prompt_ids)[None])
    # Prefill final chunk produces first prediction
    model(np.array(prompt_ids)[None])
    # logprob_list gets first entry from prefill final call

    # Teacher-forced decode: feed all but the last generated token
    for forced_token_id in forced_input_tokens_for_generated(gen_ids):
        model(np.array([forced_token_id])[None])

    return model.received_tokens


def _simulate_broken_teacher_forced_loop(
    prompt: str,
    target_text: str,
    tokenizer: FakeTokenizer,
    model: FakeModel,
) -> list[int]:
    """Simulate the BROKEN teacher-forced logic (gen_ids[1:])."""
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target_text)

    if (
        len(target_ids) >= len(prompt_ids)
        and target_ids[: len(prompt_ids)] == prompt_ids
    ):
        gen_ids = target_ids[len(prompt_ids):]
    else:
        gen_ids = target_ids

    model(np.array(prompt_ids)[None])
    model(np.array(prompt_ids)[None])

    for next_token_id in gen_ids[1:]:
        model(np.array([next_token_id])[None])

    return model.received_tokens


def test_correct_loop_feeds_all_but_last_gen_id() -> None:
    """Correct loop feeds gen_ids[:-1] to predict every next token."""
    tok = FakeTokenizer()
    model = FakeModel(tok)

    prompt = "ab"
    # target text = prompt + generated tokens "cde"
    target = "abcde"

    received = _simulate_teacher_forced_loop(prompt, target, tok, model)

    prompt_ids = tok.encode(prompt)       # [0, 1]
    gen_ids = tok.encode("cde")           # [2, 3, 4]

    # Expected calls:
    # 1. prefill prompt
    # 2. prefill final chunk (produces prediction for g1)
    # 3. feed g1 (0->2), predict g2
    # 4. feed g2 (1->3), predict g3
    expected_received = prompt_ids + prompt_ids + [gen_ids[0], gen_ids[1]]

    assert received == expected_received, (
        f"Correct loop received {received}, expected {expected_received}"
    )


def test_broken_loop_feeds_wrong_tokens() -> None:
    """Broken loop (gen_ids[1:]) feeds wrong tokens and skips g1."""
    tok = FakeTokenizer()
    model = FakeModel(tok)

    prompt = "ab"
    target = "abcde"

    received = _simulate_broken_teacher_forced_loop(prompt, target, tok, model)

    prompt_ids = tok.encode(prompt)
    gen_ids = tok.encode("cde")

    # Broken calls:
    # 1. prefill prompt
    # 2. prefill final chunk
    # 3. feed g2 (2), predict g3
    # 4. feed g3 (3), predict g4 (doesn't exist!)
    expected_received = prompt_ids + prompt_ids + gen_ids[1:]

    assert received == expected_received
    # Crucially, g1 (gen_ids[0]) was NEVER fed, and a nonexistent g4
    # was predicted.
    assert gen_ids[0] not in received[4:]  # g1 never fed after prefill


def test_correct_loop_length_assertion() -> None:
    """After prefill, correct loop produces exactly len(gen_ids) log-probs."""
    tok = FakeTokenizer()

    for gen_len in [1, 2, 5, 10]:
        model = FakeModel(tok)
        prompt = "ab"
        gen_text = "".join("c" * gen_len)
        target = prompt + gen_text

        _simulate_teacher_forced_loop(prompt, target, tok, model)

        gen_ids = tok.encode(gen_text)

        # 2 prefill calls + len(gen_ids)-1 teacher-forced calls
        expected_calls = 2 + max(0, len(gen_ids) - 1)
        assert model.call_count == expected_calls, (
            f"gen_len={gen_len}: expected {expected_calls} calls, "
            f"got {model.call_count}"
        )

        # Total log-probs = 1 (prefill) + len(gen_ids)-1 = len(gen_ids)
        # We can't check logprob_list directly, but call_count is a proxy.


def test_teacher_forced_alignment_uses_previous_generated_tokens() -> None:
    """forced_input_tokens_for_generated must return all but the last token."""
    gen_ids = [101, 102, 103, 104]
    assert forced_input_tokens_for_generated(gen_ids) == [101, 102, 103]


def test_teacher_forced_logprob_count_matches_generated_count() -> None:
    """After prefill, the number of log-prob vectors equals len(gen_ids)."""
    assert expected_logprob_count([101, 102, 103]) == 3
    assert expected_logprob_count([]) == 0
    assert expected_logprob_count([101]) == 1


def test_teacher_forced_alignment_with_varying_lengths() -> None:
    """Alignment must be correct for edge-case lengths."""
    tok = FakeTokenizer()

    # Single generated token
    model = FakeModel(tok)
    _simulate_teacher_forced_loop("a", "ab", tok, model)
    assert model.call_count == 2  # prefill + final (no loop iterations)

    # Two generated tokens
    model = FakeModel(tok)
    _simulate_teacher_forced_loop("a", "abc", tok, model)
    assert model.call_count == 3  # prefill + prefill final + 1 loop

    # Many generated tokens
    model = FakeModel(tok)
    _simulate_teacher_forced_loop("a", "a" + "b" * 20, tok, model)
    assert model.call_count == 21  # 2 + 19
