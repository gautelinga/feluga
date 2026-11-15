"""`femtools.couplings`: a bilinear form's coupled components, and matrices storing only
those blocks, on both backends.

    python3 -m pytest tests/test_couplings.py [-m mpi]

`form_couplings` reads the Stokes operator's mask off the form on both UFLs (each velocity
component with itself and the pressure, both ways; nothing else), and a vector Laplacian's
(each velocity component with itself; the pressure with nothing). On a periodic Taylor-Hood space, on one rank and three, on legacy dolfin
and DOLFINx: `assemble(a, couplings="form")` equals the full assembly entry for entry (to
round-off on three ranks), stores 31464 entries against the full pattern's 71640 on both
backends, its diagonal included, and keeps them for a second assembly into it; a coupled
block whose values are all zero is stored all the same (by component, never by value);
a nonzero in a skipped block raises on every rank. With no-slip conditions on two plates
(`bcs=`), the matrix is the full one with their dofs' rows and columns zeroed and on the
diagonal the number of cells holding the dof (dolfin's assemble_system), symmetric, with fewer entries, the same count on both backends and rank counts.
"""
import json
import os
import sys

import pytest

from tests import support

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STOKES = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0]]


def _run(backend, ranks):
    pytest.importorskip({"dolfin": "dolfin", "dolfinx": "dolfinx"}[backend])
    proc = support.run([sys.executable, os.path.join(HERE, "couplings_worker.py"), backend],
                       ranks, env={"PYTHONPATH": ROOT})
    out = [json.loads(line[7:]) for line in proc.stdout.splitlines()
           if line.startswith("RESULT ")]
    assert len(out) == ranks, proc.stdout[-2000:]
    return out


def _check(out, ranks):
    for r in out:
        assert r["mask"] == STOKES
        assert r["mask_laplace"] == [[int(i == j < 3) for j in range(4)] for i in range(4)]
        assert r["diff"] <= (0.0 if ranks == 1 else 1e-14 * r["scale"])
        assert (r["nnz"], r["nnz_full"]) == (31464, 71640)
        assert r["diagonal"]
        assert r["nnz_zero_coupling"] == r["nnz"]
        assert r["refused"]
        # no-slip: the full matrix with the rows and columns zeroed; on the diagonal the
        # number of cells holding the dof (dolfin's assemble_system)
        assert r["bc_diff"] <= (0.0 if ranks == 1 else 1e-14 * r["scale"])
        assert r["bc_diag_whole"]
        assert r["bc_asymmetry"] <= 1e-14 * r["scale"]
        assert r["n_fixed"] > 0 and r["bc_nnz"] < r["nnz"]
    return out[0]["bc_nnz"], out[0]["bc_diag_sum"]


BC_NNZ = {}


def _same_everywhere(backend, ranks, nnz):
    """The pattern with the conditions, and the sum of the fixed dofs' diagonal, are the
    same on either backend and rank count."""
    BC_NNZ[(backend, ranks)] = nnz
    assert len(set(BC_NNZ.values())) == 1, BC_NNZ


@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
def test_one_rank(backend):
    _same_everywhere(backend, 1, _check(_run(backend, 1), 1))


@pytest.mark.mpi
@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
def test_three_ranks(backend):
    _same_everywhere(backend, 3, _check(_run(backend, 3), 3))


def test_a_mask_of_the_wrong_shape_is_refused():
    np = pytest.importorskip("numpy")
    ufl = pytest.importorskip("ufl")
    basix = pytest.importorskip("basix.ufl")
    from femtools.couplings import resolve
    cell = basix.element("P", "tetrahedron", 1).cell_type.name
    mesh = ufl.Mesh(basix.element("P", cell, 1, shape=(3,)))
    V = ufl.FunctionSpace(mesh, basix.mixed_element([basix.element("P", cell, 1)] * 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    assert (resolve("form", a) == np.eye(2, dtype=bool)).all()
    with pytest.raises(ValueError, match="mask of shape"):
        resolve(np.ones((3, 2), bool), a)
    with pytest.raises(ValueError, match='None, "form" or a mask'):
        resolve("all", a)
