"""Pure helper for teacher-forced token alignment.

This module exists so that the correct teacher-forcing logic is defined in
exactly one place and can be tested independently of MLX or any model.
"""
from __future__ import annotations


def forced_input_tokens_for_generated(gen_ids: list[int]) -> list[int]:
    """Return the tokens that must be fed to the model to teacher-force
    the prediction of every token in ``gen_ids``.

    After prefill, the model already predicts the first generated token.
    To predict the second generated token we feed the first, and so on.
    Therefore we feed all generated tokens *except* the last one.

    Parameters
    ----------
    gen_ids
        List of generated token ids (the suffix after the prompt).

    Returns
    -------
    list[int]
        The tokens to feed into the model during teacher-forced decode.

    Examples
    --------
    >>> forced_input_tokens_for_generated([101, 102, 103, 104])
    [101, 102, 103]
    """
    return gen_ids[:-1]


def expected_logprob_count(gen_ids: list[int]) -> int:
    """Return the number of log-probability vectors expected from a
    teacher-forced run over ``gen_ids``.

    The prefill final call produces the log-prob for the first generated
    token.  Each subsequent forced token produces one more log-prob.
    Total = len(gen_ids).

    Parameters
    ----------
    gen_ids
        List of generated token ids.

    Returns
    -------
    int
        Expected count of log-probability vectors.
    """
    return len(gen_ids)
