"""DG0 Functions in XDMF: XDMFFile2 (dolfin) and XDMFSeries (DOLFINx), through
femtools.backend's XDMFFile, write a piecewise constant field as a cell-centred attribute
of the grid, one value per cell in the topology's own order.

A P1 field `a`, its cell means `m` and the cell centroids `x` (a 2D vector, padded to 3)
are written at one time, one call each, on one and two ranks; the file is read here
without femtools, and every cell's `m` must be the mean of `a` at its three nodes and its
`x` the mean of their positions. With `functions_share_mesh` the three are one grid,
without it one grid each, on both backends. A DG1 Function, alone in its file, is
written by dolfin's own XDMFFile on dolfin and refused on DOLFINx.

    python3 -m pytest tests/test_dg0_xdmf.py
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from tests import support

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "dg0_xdmf_worker.py")


def _grids(path):
    """Each grid of the file as {name: (center, values)} with its cells and node
    positions, from the XML and the HDF5 file directly."""
    import h5py
    root = ET.parse(path).getroot()
    out = []
    with h5py.File(os.path.splitext(path)[0] + ".h5", "r") as h5:
        def read(item):
            return h5[item.text.strip().split(":")[1]][()]
        for grid in root.iter("Grid"):
            if grid.get("GridType") == "Collection":
                continue
            out.append(dict(
                cells=read(grid.find("Topology").find("DataItem")),
                x=read(grid.find("Geometry").find("DataItem")),
                fields={a.get("Name"): (a.get("Center"), read(a.find("DataItem")))
                        for a in grid.iter("Attribute")}))
    return out


@pytest.mark.parametrize("ranks", [1, pytest.param(2, marks=pytest.mark.mpi)])
@pytest.mark.parametrize("backend", [pytest.param("dolfin", marks=pytest.mark.dolfin),
                                     pytest.param("dolfinx", marks=pytest.mark.dolfinx)])
def test_dg0_cell_attributes(tmp_path, backend, ranks):
    pytest.importorskip(backend)
    support.run([sys.executable, WORKER, backend, str(tmp_path)], ranks)
    shared = _grids(str(tmp_path / (backend + "_share.xdmf")))
    assert len(shared) == 1, len(shared)
    grid = shared[0]
    assert {k: c for k, (c, _) in grid["fields"].items()} == \
        {"a": "Node", "m": "Cell", "x": "Cell"}, grid["fields"].keys()
    cells, nodes = grid["cells"], grid["x"]
    a = grid["fields"]["a"][1][:, 0]
    m = grid["fields"]["m"][1]
    x = grid["fields"]["x"][1]
    assert m.shape == (len(cells), 1) and x.shape == (len(cells), 3), (m.shape, x.shape)
    assert np.abs(m[:, 0] - a[cells].mean(axis=1)).max() < 1e-13
    assert np.abs(x[:, :2] - nodes[cells][:, :, :2].mean(axis=1)).max() < 1e-13
    assert not x[:, 2].any()

    apart = _grids(str(tmp_path / (backend + "_apart.xdmf")))
    assert [sorted(g["fields"]) for g in apart] == [["a"], ["m"], ["x"]]
    assert np.array_equal(apart[1]["fields"]["m"][1], m)

    with open(tmp_path / (backend + ".json")) as f:
        assert json.load(f)["dg1_refused"] == (backend == "dolfinx")
