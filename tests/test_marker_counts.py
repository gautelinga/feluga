"""marker_counts with a negative marker on one rank's exterior facet alone (on dolfin a
size_t of 2**63 or more, a wrapped -1) raises on every rank, rather than leaving the other
ranks waiting in the count's reduction; on one rank and two, both backends.

    python3 -m pytest tests/test_marker_counts.py [-m mpi]
"""
import os
import sys

import pytest

from tests import support

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
@pytest.mark.parametrize("ranks", [1, pytest.param(2, marks=pytest.mark.mpi)])
def test_negative_marker_refused_on_every_rank(backend, ranks):
    pytest.importorskip(backend)
    proc = support.run([sys.executable, os.path.join(HERE, "marker_counts_worker.py"),
                        backend], ranks, timeout=120)
    raised = [line for line in proc.stdout.splitlines() if line.startswith("RAISED ")]
    assert len(raised) == ranks, proc.stdout[-2000:]
    assert all("negative marker" in line for line in raised), raised
