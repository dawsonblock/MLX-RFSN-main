"""Tests for rfsn_v11/candidates/logit_metrics.py."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_cosine_similarity_identical():
    from rfsn_v11.candidates.logit_metrics import cosine_similarity
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


@pytest.mark.unit
def test_cosine_similarity_opposite():
    from rfsn_v11.candidates.logit_metrics import cosine_similarity
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


@pytest.mark.unit
def test_cosine_similarity_2d():
    from rfsn_v11.candidates.logit_metrics import cosine_similarity
    a = np.ones((3, 4))
    b = np.ones((3, 4))
    assert cosine_similarity(a, b) == pytest.approx(1.0)


@pytest.mark.unit
def test_kl_divergence_identical():
    from rfsn_v11.candidates.logit_metrics import kl_divergence_from_logits
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    assert kl_divergence_from_logits(logits, logits) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_kl_divergence_different():
    from rfsn_v11.candidates.logit_metrics import kl_divergence_from_logits
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 1.0, 0.0]])
    kl = kl_divergence_from_logits(a, b)
    assert kl > 0.0


@pytest.mark.unit
def test_topk_overlap_perfect():
    from rfsn_v11.candidates.logit_metrics import topk_overlap
    a = np.array([[1.0, 2.0, 3.0, 4.0]])
    b = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert topk_overlap(a, b, k=2) == pytest.approx(1.0)


@pytest.mark.unit
def test_topk_overlap_zero():
    from rfsn_v11.candidates.logit_metrics import topk_overlap
    a = np.array([[1.0, 2.0, 3.0, 4.0]])
    b = np.array([[4.0, 3.0, 2.0, 1.0]])
    assert topk_overlap(a, b, k=2) == pytest.approx(0.0)


@pytest.mark.unit
def test_max_logit_delta():
    from rfsn_v11.candidates.logit_metrics import max_logit_delta
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[1.5, 2.0, 3.0]])
    assert max_logit_delta(a, b) == pytest.approx(0.5)


@pytest.mark.unit
def test_first_divergent_token():
    from rfsn_v11.candidates.logit_metrics import first_divergent_token
    assert first_divergent_token([1, 2, 3], [1, 2, 4]) == 2
    assert first_divergent_token([1, 2, 3], [1, 2, 3]) is None
    assert first_divergent_token([1, 2], [1, 2, 3]) == 2
