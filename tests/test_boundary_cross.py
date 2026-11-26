"""Boundary facet files across the backends: facet markers written by
feluga.boundary_markers on dolfin read by feluga.backend.dolfinx's BoundaryMarkers, and
the other way round, each written and read on one rank and on three, on a rectangle and
a box.

    python3 -m pytest tests/test_boundary_cross.py -m "dolfinx and dolfin and mpi"
"""
import os
import sys

import pytest

pytest.importorskip("dolfin")
pytest.importorskip("dolfinx")
pytestmark = [pytest.mark.dolfinx, pytest.mark.dolfin, pytest.mark.mpi]

from tests import support  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "boundary_cross_worker.py")


def test_boundary_files_cross_backends(tmp_path):
    for backend in ("dolfin", "dolfinx"):
        for ranks in (1, 3):
            support.run([sys.executable, WORKER, "write", backend, str(tmp_path)], ranks)
    for backend, other in (("dolfin", "dolfinx"), ("dolfinx", "dolfin")):
        for ranks in (1, 3):
            out = support.run([sys.executable, WORKER, "read", backend, str(tmp_path)],
                              ranks).stdout
            for dim in (2, 3):
                for written in (1, 3):
                    name = "{}_{}_{}d.xdmf".format(other, written, dim)
                    assert "OK " + name in out, (backend, ranks, name, out)
