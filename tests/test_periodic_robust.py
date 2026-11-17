"""The periodic pairing and its refinement where they used to break.

- Far from the origin: a box at x = 1e6 whose upper face's nodes lie one ulp off (1.2e-10
  there, beyond an absolute 1e-10) keeps its pairs on both backends, the tolerances being
  relative to the box (`femtools.periodic_core.box_scale`).
- Ghost cells among DOLFINx markers (`locate_entities` returns them) are dropped, their
  owners marking them; indices outside the local cells raise on every rank.
- A build failing on rank 0 raises on every rank instead of leaving the others waiting.
- DOLFINx's dof pairing raises on every rank when slaves find no master, and pairs a
  Taylor-Hood space component by component.
- Ranks holding no cells: DOLFINx refines; dolfin, whose refinement hangs there, raises
  on every rank.
- The pairing of face edges is O(n log n) whichever axis is periodic (it was quadratic
  along x).
- The congruence check matches partners that straddle a rounding boundary of its keys.
"""
import json
import os
import sys
import time

import numpy as np
import pytest

from tests import support

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "periodic_robust_worker.py")


def _run(mode, tmp_path, ranks, env=None):
    out = tmp_path / "{}_{}.json".format(mode, ranks)
    support.run([sys.executable, WORKER, mode, str(out)], ranks,
                env=env)
    return json.loads(out.read_text())


@pytest.mark.dolfinx
@pytest.mark.parametrize("ranks", [1, pytest.param(2, marks=pytest.mark.mpi)])
def test_far_from_origin_dolfinx(tmp_path, ranks):
    pytest.importorskip("dolfinx")
    r = _run("far_dolfinx", tmp_path, ranks)[0]
    assert r["npair"] > 0 and r["congruent"], r


@pytest.mark.dolfin
def test_far_from_origin_dolfin(tmp_path):
    pytest.importorskip("dolfin")
    r = _run("far_dolfin", tmp_path, 1)[0]
    assert r["npair"] > 0 and r["congruent"], r


@pytest.mark.dolfinx
@pytest.mark.mpi
def test_ghost_markers(tmp_path):
    pytest.importorskip("dolfinx")
    rs = _run("ghost", tmp_path, 2)
    assert rs[0]["had_ghosts"]
    for r in rs:
        assert r["with_ghosts"] == r["owned_only"], r
        assert r["out_of_range"] == "ValueError", r
        assert r["frozen_out_of_range"] == "ValueError", r


@pytest.mark.dolfinx
@pytest.mark.mpi
def test_failed_build_raises_everywhere(tmp_path):
    pytest.importorskip("dolfinx")
    rs = _run("jit", tmp_path, 2, env={"FEMTOOLS_JIT_DIR": str(tmp_path / "cache")})
    assert rs == [{"generic": "FileNotFoundError", "refine": "FileNotFoundError"}] * 2, rs


@pytest.mark.dolfinx
@pytest.mark.mpi
def test_dof_pairing_raises_everywhere(tmp_path):
    pytest.importorskip("dolfinx")
    rs = _run("pairing", tmp_path, 3)
    for r in rs:
        assert r["wrong"] == "RuntimeError", r
        assert r["masters_known"] and r["n_slaves"] > 0, r


@pytest.mark.dolfin
def test_pairing_is_not_quadratic_along_x():
    """A strip whose periodic face along x has n edges against one along y: the two
    take about the same time (the x one took 75 times longer at n = 80000)."""
    df = pytest.importorskip("dolfin")
    from femtools.plaza_periodic import refine_periodic
    n = 80000
    times = []
    for axis in (0, 1):
        nx, ny = (2, n) if axis == 0 else (n, 2)
        Lx, Ly = (1.0, n / 100.) if axis == 0 else (n / 100., 1.0)
        mesh = df.RectangleMesh(df.MPI.comm_self, df.Point(0, 0), df.Point(Lx, Ly), nx, ny)
        m = df.MeshFunction("bool", mesh, 2, False)
        refine_periodic(mesh, m, [Lx, Ly], [axis], redistribute=False)    # warm
        t = time.perf_counter()
        _, npair, _ = refine_periodic(mesh, m, [Lx, Ly], [axis], redistribute=False)
        times.append(time.perf_counter() - t)
        assert npair == 2 * n
    assert times[0] < 4 * times[1] + 0.2, times


def test_congruence_across_a_rounding_boundary():
    """Partners one ulp apart on either side of a rounding boundary of the check's keys
    (x / tol at k + 1/2) are partners."""
    from mpi4py import MPI
    from femtools.periodic_core import faces_congruent_arrays
    tol = 1e-9
    # a y whose y / tol rounds one way, and the next double the other way
    y = None
    for k in range(10**6, 10**6 + 1000):
        a = (k + 0.5) * tol
        for c in (a, np.nextafter(a, 0), np.nextafter(a, 1)):
            d = np.nextafter(c, 1)
            if np.round(c / tol) != np.round(d / tol):
                y = (c, d)
                break
        if y:
            break
    assert y is not None
    # the box [0, 1] x [0, 2e-3 + ...]: two nodes a side, one at y[0] on x = 0 and its
    # partner at y[1] on x = 1, the others at y = 0
    x = np.array([[0.0, 0.0], [0.0, y[0]], [1.0, 0.0], [1.0, y[1]]])
    ok, axis, _, _ = faces_congruent_arrays(MPI.COMM_SELF, x, None, [1.0, 1.0], [0], tol,
                                            lower=[0.0, 0.0])
    assert ok, axis


@pytest.mark.dolfinx
@pytest.mark.parametrize("ranks", [1, pytest.param(3, marks=pytest.mark.mpi)])
def test_taylor_hood_pairing(tmp_path, ranks):
    pytest.importorskip("dolfinx")
    rs = _run("taylor_hood", tmp_path, ranks)
    # the dofs on the upper faces: 2 x 21 P2 nodes (velocity) and 11 P1 ones (pressure)
    assert sum(r["n_unique"] for r in rs) >= 2 * 21 + 11
    for r in rs:
        assert r["worst"] < 1e-12 and r["same_leaf"], r


@pytest.mark.dolfin
@pytest.mark.mpi
def test_zero_cell_ranks_dolfin(tmp_path):
    pytest.importorskip("dolfin")
    rs = _run("zero_cells_dolfin", tmp_path, 4)
    assert any(r["cells"] == 0 for r in rs)
    for r in rs:
        assert r["periodic"] == "ValueError" and r["plain"] == "ValueError", r


@pytest.mark.dolfinx
@pytest.mark.mpi
def test_zero_cell_ranks_dolfinx(tmp_path):
    pytest.importorskip("dolfinx")
    rs = _run("zero_cells_dolfinx", tmp_path, 4)
    assert any(r["cells"] == 0 for r in rs)
    for r in rs:
        assert r["periodic"][:2] == [8, 4] and r["congruent"], r
        assert r["capped"] == 8 and r["plain"] == 8, r
