"""femtools.backend's cell tools on dolfin and DOLFINx, on one rank and two:
`cell_values` gives a DG0 field's value on each owned cell, once per cell over the ranks,
and a vector one's as (cells, components); `refine_cells` refuses markers of the wrong
length on every rank, and refines the cells so marked into the same cells on both
backends and any number of ranks (compared in the slow tier); `interpolate` carries a
Function onto a space of another mesh (exactly, for a linear field onto P1 of a
refinement, also when repeated), and evaluates an Expression at the target space's own
points wherever the Expression's default mesh is (here a box outside it); and DOLFINx's
BoundaryMarkers take dolfin's `rename`.

    python3 -m pytest tests/test_backend_cells.py
"""
import json
import os
import sys

import pytest

from tests import support

pytestmark = [pytest.mark.dolfin, pytest.mark.dolfinx]
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_cells_worker.py")
RANKS = [1, pytest.param(2, marks=pytest.mark.mpi)]


@pytest.fixture(scope="module")
def tooled(tmp_path_factory):
    """tooled(backend, ranks): the worker's results, run once a module."""
    done = {}

    def get(backend, ranks):
        if (backend, ranks) not in done:
            pytest.importorskip(backend)
            out = tmp_path_factory.mktemp("cells") / "{}{}.json".format(backend, ranks)
            support.run([sys.executable, WORKER, backend, str(out)], ranks)
            with open(out) as handle:
                done[backend, ranks] = json.load(handle)
        return done[backend, ranks]
    return get


@pytest.mark.parametrize("ranks", RANKS)
@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
def test_cell_tools(tooled, backend, ranks):
    r = tooled(backend, ranks)
    assert r["n_values"] == 384                     # 6 * 4^3 cells, each once
    assert abs(r["sum_values"] - 384.0) < 1e-12     # centroids' x + y average 1
    assert r["vec_ok"]
    assert r["refused"]
    assert r["n_marked"] == 176
    assert r["n_fine"] == len(r["cells"])
    assert r["err_cross"] < 1e-13
    assert abs(r["integral"] - 8.0 / 3.0) < 1e-12
    if backend == "dolfinx":
        assert r["renamed"] == "walls"


@pytest.mark.slow
@pytest.mark.parametrize("ranks", RANKS)
def test_refined_cells_agree(tooled, ranks):
    assert tooled("dolfin", ranks)["cells"] == tooled("dolfinx", ranks)["cells"]
