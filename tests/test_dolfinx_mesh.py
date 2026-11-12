"""`femtools.dolfinx.mesh.read_mesh`: meshes dolfin writes, read on DOLFINx.

dolfin writes a rectangle and a box on three ranks (so the file's cell_indices are not
the identity) and a P2 field on each; DOLFINx reads them on one rank and on three. Every
cell's original index names the file's cell with its vertices, at their coordinates, and
the field reads back (femtools.dolfinx.checkpoint, tetrahedra included) to round-off.

    python3 -m pytest tests/test_dolfinx_mesh.py -m "dolfinx and dolfin and mpi"
"""
import json
import os
import sys

import pytest

pytest.importorskip("dolfinx")
pytestmark = [pytest.mark.dolfinx, pytest.mark.dolfin, pytest.mark.mpi]

from tests import support  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKER = os.path.join(HERE, "dolfinx_mesh_worker.py")


@pytest.mark.parametrize("ranks", [1, 3])
def test_dolfin_mesh_on_dolfinx(tmp_path, ranks):
    pytest.importorskip("dolfin")
    env = {"PYTHONPATH": ROOT}
    support.run([sys.executable, WORKER, "write-dolfin", str(tmp_path)], 3, env=env)
    out = tmp_path / "res.json"
    support.run([sys.executable, WORKER, "read-dolfinx", str(tmp_path), str(out)], ranks, env=env)
    res = json.loads(out.read_text())
    for name, n_cells in (("rect", 140), ("box", 270)):
        assert res[name + "_cells"], name
        assert not res[name + "_identity_ids"], name + ": the test needs scrambled ids"
        assert res[name + "_field"] < 1e-13, name
        assert res[name + "_cells_global"] == n_cells, name
