"""feluga.backend's Probes and GradProbes on dolfin and DOLFINx: the values and
gradients of a P2 scalar (two snapshots), the gradient of a P1 vector, the values of
a DG0 field and of a mixed P2 x P1 field, at seeded random points inside the unit cube
and one outside it, on one rank and two. The fields are polynomials the spaces hold
exactly, so values and gradients are checked against the formulas on each backend; the
two backends are compared with each other, and DOLFINx's probes on cell faces between
one rank and two, in the slow tier.

    python3 -m pytest tests/test_backend_probes.py
"""
import os
import sys

import numpy as np
import pytest

from tests import support
from tests.probe_points import face_points, points

pytestmark = [pytest.mark.dolfin, pytest.mark.dolfinx]

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_probes_worker.py")
RANKS = [1, pytest.param(2, marks=pytest.mark.mpi)]


@pytest.fixture(scope="module")
def probed(tmp_path_factory):
    """probed(backend, ranks): the worker's arrays, run once a module."""
    done = {}

    def get(backend, ranks):
        if (backend, ranks) not in done:
            pytest.importorskip(backend)
            out = tmp_path_factory.mktemp("probes") / "{}{}.npz".format(backend, ranks)
            support.run([sys.executable, WORKER, backend, str(out)], ranks)
            done[backend, ranks] = dict(np.load(out))
        return done[backend, ranks]
    return get


@pytest.mark.parametrize("ranks", RANKS)
@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
def test_probes_exact(probed, backend, ranks):
    r = probed(backend, ranks)
    x = points()
    inside = slice(0, 40)
    X, Y, Z = x[inside].T
    f = X**2 + 2 * Y * Z - Z
    assert r["f"].shape == (41, 2)
    assert np.allclose(r["f"][inside, 0], f, atol=1e-12)
    assert np.allclose(r["f"][inside, 1], 2 * f, atol=1e-12)
    df = np.column_stack((2 * X, 2 * Z, 2 * Y - 1))
    assert np.allclose(r["df"][inside, :, 0], df, atol=1e-11)
    assert np.allclose(r["df"][inside, :, 1], 2 * df, atol=1e-11)
    assert np.allclose(r["u"][inside], np.column_stack((X + 2 * Y, 3 * Z - X, Y)),
                       atol=1e-12)
    du = np.array([1, 2, 0, -1, 0, 3, 0, 1, 0], dtype=float)
    assert np.allclose(r["du"][inside], du, atol=1e-11)
    assert r["mixed"].shape == (41, 2)
    assert np.allclose(r["mixed"][inside], np.column_stack((f, X - Y)), atol=1e-12)
    for name in ("f", "df", "u", "du", "m", "mixed"):
        assert not r[name][40].any(), name          # outside the mesh: 0
    # on a face z = c, the centroid z of the cell on either side: c -+ 1/16 or 3/16
    side = np.abs(np.ravel(r["face"]) - face_points()[:, 2])
    assert np.all((side > 0.05) & (side < 0.2))
    # dump: one probe for each point found, read back as `array` has it
    assert list(r["dump_ids"]) == list(range(40))
    assert np.allclose(r["dump_f"], r["f"][inside], rtol=1e-6)
    assert np.allclose(r["dump_x"], x[inside], rtol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("ranks", RANKS)
def test_probes_backends_agree(probed, ranks):
    a, b = probed("dolfin", ranks), probed("dolfinx", ranks)
    for name in ("f", "df", "u", "du", "m", "mixed"):
        assert np.allclose(a[name], b[name], atol=1e-11), name


@pytest.mark.mpi
def test_dolfinx_face_probes_independent_of_ranks(probed):
    """A probe on a face is evaluated in the cell of smallest original index, whatever
    the partition: the DG0 values there are the same on one rank and two."""
    assert np.array_equal(probed("dolfinx", 1)["face"], probed("dolfinx", 2)["face"])
