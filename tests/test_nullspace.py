"""`femtools.nullspace.constant_nullspace`: the constants on a space, or on one
component of a mixed space, attached to a matrix, on one rank and on three.

    python3 -m pytest tests/test_nullspace.py
"""
import os
import sys

import pytest

df = pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

from tests import support  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nullspace_worker.py")


@pytest.mark.parametrize("ranks", [1, pytest.param(3, marks=pytest.mark.mpi)])
def test_constants(tmp_path, ranks):
    out = tmp_path / "err.txt"
    support.run([sys.executable, WORKER, str(out)], ranks,
                env={"PYTHONPATH": os.path.dirname(os.path.dirname(WORKER))})
    assert float(out.read_text()) < 1e-12
