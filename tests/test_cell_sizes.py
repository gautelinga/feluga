"""The backends' hmin and hmax are over all ranks, the same on each (dolfin's own
mesh.hmin and mesh.hmax are a rank's): a mesh refined on one rank's cells alone, on
three ranks.

    python3 -m pytest tests/test_cell_sizes.py -m mpi
"""
import os
import sys

import pytest

from tests import support

SCRIPT = r"""
import importlib, sys
from mpi4py import MPI
B = importlib.import_module("femtools.backend." + sys.argv[1])
comm = MPI.COMM_WORLD
if sys.argv[1] == "dolfin":
    import dolfin as df
    m = df.UnitSquareMesh(comm, 16, 16)
    m = df.refine(m, df.MeshFunction("bool", m, 2, comm.rank == 0))
else:
    import numpy as np
    import dolfinx.mesh as dm
    m = B.UnitSquareMesh(16, 16)
    m.topology.create_entities(1)
    n = m.topology.index_map(2).size_local if comm.rank == 0 else 0
    edges = dm.compute_incident_entities(m.topology, np.arange(n, dtype=np.int32), 2, 1)
    m = dm.refine(m, np.unique(edges))[0]
sizes = comm.gather((B.hmin(m), B.hmax(m)), root=0)
if comm.rank == 0:
    print("SAME", len(set(sizes)) == 1, sizes[0][0] < sizes[0][1])
"""


@pytest.mark.mpi
@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
def test_hmin_hmax_over_all_ranks(backend):
    if backend == "dolfinx":
        pytest.importorskip("dolfinx")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = support.run([sys.executable, "-c", SCRIPT, backend], 3, cwd=root)
    assert "SAME True True" in proc.stdout, proc.stdout[-2000:]
