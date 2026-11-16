"""read_mesh_file on both backends reads the meshers' XDMF mesh (meshio's layout: an XDMF
naming two datasets of the HDF5 file beside it), on one rank and two (dolfin then through
the periodic partition): the same cells and vertices, the box's volume, and every cell's
global id its row in the file, as dolfin numbers it.

    python3 -m pytest tests/test_read_mesh_file.py [-m mpi]
"""
import json
import os
import sys

import numpy as np
import pytest

from tests import support

HERE = os.path.dirname(os.path.abspath(__file__))
h5py = pytest.importorskip("h5py")

XDMF = ('<Xdmf Version="3.0"><Domain><Grid Name="Grid"><Geometry GeometryType="XYZ">'
        '<DataItem DataType="Float" Dimensions="{nx} 3" Format="HDF" Precision="8">'
        'cube.h5:/data0</DataItem></Geometry><Topology TopologyType="Tetrahedron" '
        'NumberOfElements="{nc}" NodesPerElement="4"><DataItem DataType="Int" '
        'Dimensions="{nc} 4" Format="HDF" Precision="4">cube.h5:/data1</DataItem>'
        '</Topology></Grid></Domain></Xdmf>')


def _cube(folder, n=4):
    """A Kuhn-split n^3 cube on [0, 1.5]^3 (congruent opposite faces), rows shuffled."""
    g = np.stack(np.meshgrid(*[np.linspace(0, 1.5, n + 1)] * 3, indexing="ij"),
                 axis=-1).reshape(-1, 3)
    idx = np.arange(len(g)).reshape([n + 1] * 3)
    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v = idx[i:i + 2, j:j + 2, k:k + 2].ravel()
                for t in ((0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7),
                          (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)):
                    cells.append(v[list(t)])
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(g))
    cells = np.argsort(perm)[np.asarray(cells)][rng.permutation(len(cells))]
    with h5py.File(os.path.join(folder, "cube.h5"), "w") as h5:
        h5["data0"] = g[perm]
        h5["data1"] = cells.astype(np.int32)
    path = os.path.join(folder, "cube.xdmf")
    with open(path, "w") as fh:
        fh.write(XDMF.format(nx=len(g), nc=len(cells)))
    return path, len(cells), len(g)


@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
@pytest.mark.parametrize("ranks", [1, pytest.param(2, marks=pytest.mark.mpi)])
def test_read_mesh_file_xdmf(tmp_path, backend, ranks):
    pytest.importorskip(backend)
    path, n_cells, n_vertices = _cube(str(tmp_path))
    proc = support.run([sys.executable, os.path.join(HERE, "xdmf_mesh_worker.py"), backend,
                        path], ranks, timeout=300)
    out = [json.loads(line[7:]) for line in proc.stdout.splitlines()
           if line.startswith("RESULT ")]
    assert len(out) == 1, proc.stdout[-2000:]
    r = out[0]
    assert (r["cells"], r["vertices"]) == (n_cells, n_vertices), r
    assert r["volume"] == pytest.approx(1.5 ** 3, rel=1e-12), r
    assert r["ids_are_rows"] and r["every_cell_once"], r
