"""Batch 3 of stokes-beadpack's general code across the two backends:
`fast_interpolation.p1_to_p2`, `cell_mean.cell_mean_norm`,
`recovery.recovered_vertex_gradient`, `vertex_field`.

    python3 -m pytest tests/test_p1_tools.py [-m mpi]

Needs legacy dolfin and DOLFINx in one interpreter; tests/test_p1_tools_dolfin.py checks
the legacy halves alone. On a box periodic in x and y, read from a mesh file whose rows are
shuffled (so file order is not cell order), on one rank and three, legacy dolfin then
DOLFINx then dolfin again: the P1 -> P2 matrix from the dofmaps holds weights 1 and 1/2
exactly, its rows summing to 1, and equals dolfin's geometric transfer matrix, and on
DOLFINx the backend's interpolation, to round-off; the cell-mean norm of a P1 and a P2
velocity (vector or three scalars) is the assembled cell mean's norm, and a P3 one is
refused; a linear field's recovered gradient is its gradient; the recovered vertex
gradients of the two backends agree vertex by vertex; a P1 field holding each vertex's
file row is written by either backend, on either rank count, as 0, 1, 2, ... (the file's
vertex order), and reads back exactly into either.
"""
import os

import numpy as np
import pytest

pytest.importorskip("dolfin")
pytest.importorskip("dolfinx")

from tests.test_p1_tools_dolfin import _check, _mesh, _run  # noqa: E402


def _case(tmp_path, ranks):
    folder = str(tmp_path)
    _mesh(folder)
    results = [_run(folder, "dolfin", ranks, "write"), _run(folder, "dolfinx", ranks, "read"),
               _run(folder, "dolfinx", ranks, "write")]
    os.rename(os.path.join(folder, "field-dolfin-{}.h5".format(ranks)),        # kept aside:
              os.path.join(folder, "kept-dolfin-{}.h5".format(ranks)))        # read dolfinx's
    results.append(_run(folder, "dolfin", ranks, "read"))
    for r in results:
        _check(r)
    for r in (results[0], results[2]):
        assert r["vertex_field_in_file_order"], r
    for r in results:
        assert r["vertex_field_read"] == 0.0, r      # its own file, or the other's
    a, b = (np.load(os.path.join(folder, "grad-{}-{}.npz".format(k, ranks)))
            for k in ("dolfin", "dolfinx"))
    assert np.array_equal(np.round(a["x"], 9), np.round(b["x"], 9))
    assert np.abs(a["g"] - b["g"]).max() <= 1e-12 * np.abs(a["g"]).max()
    return folder


def test_one_rank(tmp_path):
    _case(tmp_path, 1)


@pytest.mark.mpi
def test_three_ranks(tmp_path):
    _case(tmp_path, 3)
