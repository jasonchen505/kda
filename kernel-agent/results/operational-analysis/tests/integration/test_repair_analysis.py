import itertools

import pytest

from scripts.summarize_repair import task_cluster_interval


@pytest.mark.parametrize("values", [[0], [0, 1], [-1, 0, 1], [0, .25, -.5, 1], [0, 0, 0, 1, 1]])
def test_weighted_bootstrap_matches_every_ordered_resample(values):
    brute = sorted(sum(sample) / len(values)
                   for sample in itertools.product(values, repeat=len(values)))
    actual = task_cluster_interval(values)
    assert actual["resamples"] == len(brute)
    assert actual["low"] == brute[int(.025 * (len(brute) - 1))]
    assert actual["high"] == brute[int(.975 * (len(brute) - 1))]


def test_eight_task_clusters_use_exact_binomial_quantiles():
    actual = task_cluster_interval([0] * 4 + [1] * 4)
    assert actual["clusters"] == 8
    assert actual["resamples"] == 8 ** 8
    assert actual["weighted_patterns"] == 6435
    # For Binomial(8, .5), P(K<=0)=1/256 and P(K<=1)=9/256.
    assert (actual["low"], actual["high"]) == (.125, .875)


def test_constant_task_effect_keeps_a_degenerate_interval():
    actual = task_cluster_interval([.25] * 8)
    assert actual["low"] == actual["high"] == .25


@pytest.mark.parametrize("values", [[], [0] * 9])
def test_unsupported_cluster_count_fails_explicitly(values):
    with pytest.raises(ValueError, match="one to eight"):
        task_cluster_interval(values)
