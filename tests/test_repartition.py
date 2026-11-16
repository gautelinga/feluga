"""femtools.periodic_partition's rebalance: `repartition_plan` on an unbalanced periodic
mesh, built, keeps every cell and vertex with its global id, balances the ranks, keeps the
periodic pairing, and `carry_by_id` brings rank-local arrays along.

    python3 -m pytest tests/test_repartition.py

**System python3** (dolfin) and `mpirun`: two ranks. The worker is
`tests/repartition_worker.py`.
"""
import os
import sys

import pytest

pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin, pytest.mark.mpi]

from tests import support  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repartition_worker.py")
CHECKS = ("skewed", "planned", "balanced", "counts_kept", "cell_ids_kept", "vertex_ids_kept",
          "carried", "plan_carry", "paired", "again_none")


@pytest.fixture(scope="module")
def results():
    proc = support.run([sys.executable, WORKER], 2, timeout=600)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            _, name, value = line.split(None, 2)
            out[name] = value
    return out


@pytest.mark.parametrize("check", CHECKS)
def test_repartition(results, check):
    assert results.get(check) == "True", results
