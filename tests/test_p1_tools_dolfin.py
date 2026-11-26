"""Batch 3 of stokes-beadpack's general code on legacy dolfin alone:
`fast_interpolation.p1_to_p2`, `cell_mean.cell_mean_norm`,
`recovery.recovered_vertex_gradient`, `vertex_field`.

    python3 -m pytest tests/test_p1_tools_dolfin.py [-m mpi]

On a box periodic in x and y, read from a mesh file whose rows are shuffled (so file order
is not cell order), on one rank and three: the P1 -> P2 matrix from the dofmaps holds
weights 1 and 1/2 exactly, its rows summing to 1, and equals dolfin's geometric transfer
matrix to round-off; the cell-mean norm of a P1 and a P2 velocity (vector or three
scalars) is the assembled cell mean's norm, and a P3 one is refused; a linear field's
recovered gradient is its gradient; a P1 field holding each vertex's file row is written
as 0, 1, 2, ... (the file's vertex order) and reads back exactly. tests/test_p1_tools.py
runs the same worker across both backends.
"""
import json
import os
import sys

import numpy as np
import pytest

from tests import support

HERE = os.path.dirname(os.path.abspath(__file__))
pytest.importorskip("dolfin")
h5py = pytest.importorskip("h5py")


def _mesh(folder):
    """A 4 x 4 x 3 box's mesh file in dolfin's layout, its coordinate rows shuffled."""
    from feluga.dolfin_layout import write_mesh_arrays
    n = (4, 4, 3)
    g = np.stack(np.meshgrid(*[np.linspace(0, 1, k + 1) for k in n], indexing="ij"),
                 axis=-1).reshape(-1, 3)
    idx = np.arange(len(g)).reshape([k + 1 for k in n])
    cells = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                v = idx[i:i + 2, j:j + 2, k:k + 2].ravel()     # the cube's 8 corners
                for t in ((0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7),
                          (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)):
                    cells.append(v[list(t)])
    cells = np.asarray(cells)
    perm = np.random.default_rng(4).permutation(len(g))
    inv = np.argsort(perm)
    write_mesh_arrays(os.path.join(folder, "mesh.h5"), g[perm], inv[cells])


def _run(folder, backend, ranks, stage):
    support.run([sys.executable, os.path.join(HERE, "p1_tools_worker.py"), backend, folder,
                 stage], ranks)
    with open(os.path.join(folder, "{}-{}-{}.json".format(backend, ranks, stage))) as fh:
        return json.load(fh)


def _check(r):
    """The checks every run of the worker passes, on either backend."""
    assert r["p1_to_p2"] <= 1e-14 * r["p1_to_p2_scale"], r
    assert r["p1_to_p2_exact"], r        # weights 1 and 1/2 exactly, rows summing to 1
    assert r["cell_mean"] <= 1e-14, r
    assert r["cell_mean_p3_refused"], r
    assert r["recovery_linear"] <= 1e-12, r


def _case(tmp_path, ranks):
    folder = str(tmp_path)
    _mesh(folder)
    r = _run(folder, "dolfin", ranks, "write")      # writes, and reads its own file back
    _check(r)
    assert r["vertex_field_in_file_order"], r
    assert r["vertex_field_read"] == 0.0, r


def test_one_rank(tmp_path):
    _case(tmp_path, 1)


@pytest.mark.mpi
def test_three_ranks(tmp_path):
    _case(tmp_path, 3)
