import pytest
from evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_perfect_and_partial():
    rel = ["c1", "c2"]

    # Perfect retrieval in top 2
    assert recall_at_k(["c1", "c2", "c3"], rel, 2) == 1.0

    # Partial retrieval (1 of 2 in top 1)
    assert recall_at_k(["c1", "c3"], rel, 1) == 0.5

    # Rank outside K
    assert recall_at_k(["x", "y", "c1"], rel, 2) == 0.0


def test_recall_at_k_edge_cases():
    # Empty relevant list
    assert recall_at_k(["c1"], [], 3) == 0.0

    # Empty retrieved list
    assert recall_at_k([], ["c1"], 3) == 0.0

    # Invalid k <= 0
    assert recall_at_k(["c1"], ["c1"], 0) == 0.0
    assert recall_at_k(["c1"], ["c1"], -1) == 0.0


def test_precision_at_k_normal_and_fewer_results():
    rel = ["c1", "c2"]

    # 1 of top 1 is relevant -> 1/1 = 1.0
    assert precision_at_k(["c1", "x"], rel, 1) == 1.0

    # 1 of top 2 is relevant -> 1/2 = 0.5
    assert precision_at_k(["c1", "x"], rel, 2) == 0.5

    # Fewer than K results returned (1 result returned when k=3, 1 relevant)
    assert precision_at_k(["c1"], rel, 3) == 1.0

    # Fewer than K results returned (1 result returned when k=3, 0 relevant)
    assert precision_at_k(["x"], rel, 3) == 0.0

    # Zero results returned -> 0.0
    assert precision_at_k([], rel, 3) == 0.0

    # Invalid k <= 0
    assert precision_at_k(["c1"], rel, 0) == 0.0


def test_hit_rate_at_k():
    rel = ["c1", "c2"]

    # Hit in top 1
    assert hit_rate_at_k(["c1", "x", "y"], rel, 1) == 1.0

    # Hit in top 3 (c2 at rank 3)
    assert hit_rate_at_k(["x", "y", "c2"], rel, 3) == 1.0

    # Miss in top 2 (c2 at rank 3)
    assert hit_rate_at_k(["x", "y", "c2"], rel, 2) == 0.0

    # Edge cases
    assert hit_rate_at_k([], rel, 3) == 0.0
    assert hit_rate_at_k(["c1"], [], 3) == 0.0
    assert hit_rate_at_k(["c1"], rel, 0) == 0.0


def test_ndcg_at_k():
    rel = ["c1", "c2"]

    # Ideal ranking: c1 at 1, c2 at 2 -> nDCG@2 = 1.0
    assert ndcg_at_k(["c1", "c2", "x"], rel, 2) == 1.0

    # Relevant at rank 1 only -> DCG@2 = 1/log2(2) = 1.0, IDCG@2 = 1/log2(2) + 1/log2(3) = 1.6309 -> nDCG@2 = 1/1.6309 = 0.6131
    assert ndcg_at_k(["c1", "x"], rel, 2) == round(1.0 / (1.0 + 1.0 / 1.58496), 4)

    # Relevant at rank 2 only -> DCG@2 = 1/log2(3) = 0.6309 -> nDCG@2 = (1/1.58496) / 1.6309 = 0.3869
    assert ndcg_at_k(["x", "c1"], rel, 2) == round((1.0 / 1.58496) / (1.0 + 1.0 / 1.58496), 4)

    # Edge cases
    assert ndcg_at_k([], rel, 3) == 0.0
    assert ndcg_at_k(["c1"], [], 3) == 0.0
    assert ndcg_at_k(["c1"], rel, 0) == 0.0


def test_reciprocal_rank_ranks():
    rel = ["c2", "c3"]

    # Relevant at rank 1
    assert reciprocal_rank(["c2", "c1", "c3"], rel) == 1.0

    # Relevant at rank 2
    assert reciprocal_rank(["c1", "c2", "c3"], rel) == 0.5

    # Relevant at rank 3
    assert reciprocal_rank(["x", "y", "c3"], rel) == 1.0 / 3.0

    # No relevant retrieved
    assert reciprocal_rank(["x", "y", "z"], rel) == 0.0

    # Empty retrieved / empty relevant
    assert reciprocal_rank([], rel) == 0.0
    assert reciprocal_rank(["c1"], []) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == (1.0 + 0.5 + 0.0) / 3.0
    assert mean_reciprocal_rank([]) == 0.0
