"""femtools.backend.dolfinx: a Function's compact vector and DOLFINx's array kept
consistent, whatever writes them, on one rank and two.

    python3 -m pytest tests/test_dolfinx_sync.py [-m mpi]

A write to the compact vector on one rank only does not hang the next assembly (the
decision to prolong is every rank's); a partial write through DOLFINx's array after the
compact vector changed starts from the vector's values; interpolating one component keeps
the other's values, ghosts included; on a periodic space a no-op write does not change
what assembly sees (slaves are their masters', never their own positions' values); a
subclass's own sync (a copy following another Function) runs before each assembly.
"""
import json
import os
import sys

import pytest

from tests import support

pytest.importorskip("dolfinx")
pytestmark = [pytest.mark.dolfinx]

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(ranks):
    proc = support.run([sys.executable, os.path.join(HERE, "dolfinx_sync_worker.py")], ranks,
                       timeout=300)
    out = [json.loads(line[7:]) for line in proc.stdout.splitlines()
           if line.startswith("RESULT ")]
    assert len(out) == ranks, proc.stdout[-2000:]
    for r in out:
        assert r["one_rank_write"] == pytest.approx(1.0, rel=1e-12)
        assert r["partial_array_write"] == pytest.approx(3.0, rel=1e-12)
        assert r["component_kept"] == pytest.approx(5.0, rel=1e-12)
        assert r["component_set"] == pytest.approx(4.0, rel=1e-12)
        assert r["periodic_same"]
        assert r["follower"] == pytest.approx(7.0, rel=1e-12)


def test_one_rank():
    _run(1)


@pytest.mark.mpi
def test_two_ranks():
    _run(2)
