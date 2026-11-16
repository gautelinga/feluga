"""femtools.backend.dolfinx: the backend's caches (compiled forms, constant matrices,
linear solvers, components of mixed spaces, nodal values) are kept on the meshes and
spaces they are built on, so a mesh the caller drops is freed (a refinement ladder does not
keep every level), while a form assembled again on a live mesh is compiled once.

    python3 -m pytest tests/test_dolfinx_lifetime.py
"""
import os
import sys

import pytest

from tests import support

pytest.importorskip("dolfinx")
pytestmark = [pytest.mark.dolfinx]

HERE = os.path.dirname(os.path.abspath(__file__))


def test_dropped_meshes_are_freed():
    proc = support.run([sys.executable, os.path.join(HERE, "dolfinx_lifetime_worker.py")], 1,
                       timeout=300)
    out = [line.split() for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(out) == 1, proc.stdout[-2000:]
    assert out[0][1:] == ["0", "True"], proc.stdout[-2000:]
