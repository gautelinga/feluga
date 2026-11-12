"""`femtools.periodic_core.dorfler_threshold_values`: the bulk-chasing threshold on arrays,
one rank."""
import numpy as np

from femtools.periodic_core import dorfler_threshold_values


def test_threshold_marks_the_smallest_bulk():
    rng = np.random.default_rng(0)
    eta = rng.standard_normal(500) * rng.exponential(1.0, 500)
    a = np.sort(np.abs(eta))[::-1]
    for theta in (0.1, 0.5, 0.9):
        t = dorfler_threshold_values(eta, theta)
        marked = np.abs(eta) > t
        assert np.abs(eta)[marked].sum() >= theta * a.sum()
        # the fewest cells that carry the share, up to the bisection's resolution
        assert marked.sum() == np.searchsorted(np.cumsum(a), theta * a.sum()) + 1


def test_zero_and_empty_indicators():
    assert dorfler_threshold_values(np.zeros(4), 0.5) == 0.0
    assert dorfler_threshold_values(np.zeros(0), 0.5) == 0.0
