"""`femtools.shared_amg`: one BoomerAMG for a Stokes operator's velocity components.

    python3 -m pytest tests/test_shared_amg.py [-m mpi]

On a periodic [P2, P2, P2, P1] space, the operator stored by component
(femtools.couplings), legacy dolfin and DOLFINx, one rank and three: the component index
sets list every node once (on three ranks too), the block check passes and refuses an operator whose second
component's Laplacian differs, the same values on two components (in node order) come back
from the shared AMG the same, and the pressure is divided by diag(M). stokes-beadpack's
tests solve with it against the fieldsplit, nodes owned apart included.
"""
import json
import os
import sys

import pytest

from tests import support

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _run(backend, ranks):
    pytest.importorskip(backend)
    proc = support.run([sys.executable, os.path.join(HERE, "shared_amg_worker.py"), backend],
                       ranks, env={"PYTHONPATH": ROOT})
    out = [json.loads(line[7:]) for line in proc.stdout.splitlines()
           if line.startswith("RESULT ")]
    assert len(out) == ranks, proc.stdout[-2000:]
    for r in out:
        assert len(set(r["sizes"])) == 1, r["sizes"]
        assert r["check_passes"] and r["changed_block_refused"]
        assert r["same_input_same_output"] and r["pressure"]
    return out


@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
def test_one_rank(backend):
    _run(backend, 1)


@pytest.mark.mpi
@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
def test_three_ranks(backend):
    out = _run(backend, 3)
    assert sum(r["sizes"][0] for r in out) == 8 * 8 * 9     # P2 nodes, periodic in x and y
