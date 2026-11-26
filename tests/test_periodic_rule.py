"""The periodic pairing rule has one reference, `periodic_core.classify`, which DOLFINx's dof
pairing uses: it must judge points exactly as dolfin's compiled box does (`inside`: on a
lower periodic face and on no upper one; `map`: every upper coordinate down by the box),
the tolerance absolute and inclusive. On a box away from the origin, for face, edge and
corner points just inside and just outside the tolerance, and points inside the box.

    python3 -m pytest tests/test_periodic_rule.py
"""
import itertools

import numpy as np
import pytest

pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

LOWER = np.array([0.5, -1.0, 2.0])
UPPER = np.array([1.5, 2.0, 3.5])
PERIODIC = (True, False, True)
TOL = 1e-8


def _points():
    rng = np.random.default_rng(3)
    pts = [LOWER + rng.random((50, 3)) * (UPPER - LOWER)]
    for offsets in itertools.product(*[(None, "lo", "hi")] * 3):
        for scale in (0.0, 0.9, 1.1):
            for sign in (1, -1):
                p = LOWER + rng.random(3) * (UPPER - LOWER)
                for a, where in enumerate(offsets):
                    if where is not None:
                        p[a] = (LOWER if where == "lo" else UPPER)[a] + sign * scale * TOL
                pts.append(p[None])
    return np.concatenate(pts)


def test_classify_is_the_compiled_box():
    from feluga.periodic import periodic_box
    from feluga.periodic_core import classify
    box = periodic_box(UPPER, TOL, PERIODIC, lower=LOWER)
    x = _points()
    on_lower, on_upper, image = classify(x, LOWER, UPPER, PERIODIC, TOL)
    inside = np.array([box.inside(p, True) for p in x])
    assert np.array_equal(inside, on_lower & ~on_upper)
    mapped = np.empty_like(x)
    for k, p in enumerate(x):
        y = np.empty(3)
        box.map(p, y)
        mapped[k] = y
    assert np.array_equal(mapped, image)          # bit for bit: the same subtraction
    assert on_upper.sum() > 50 and (on_lower & ~on_upper).sum() > 20      # both sides met
