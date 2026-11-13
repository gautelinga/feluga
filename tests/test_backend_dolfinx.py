"""`femtools.backend.dolfinx`: dolfin's object API on DOLFINx. A Function's DOLFINx writes
(also on a component view, and by interpolation) reach its compact vector, sync copies
only after a change, facets marked after a measure was made count, a Dirichlet
condition on periodic slaves constrains their masters on any number of ranks, and a
periodic Poisson solve through dolfin's names gives the exact quadratic -- on one rank
and on three.

    python3 -m pytest tests/test_backend_dolfinx.py [-m mpi]
"""
import json
import os
import sys

import pytest

pytest.importorskip("dolfinx")
pytestmark = [pytest.mark.dolfinx]

from tests import support  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _run(tmp_path, ranks):
    out = tmp_path / "backend.{}.json".format(ranks)
    support.run([sys.executable, os.path.join(HERE, "backend_dolfinx_worker.py"), str(out)],
                ranks, env={"PYTHONPATH": ROOT})
    return json.loads(out.read_text())


def _check(r):
    assert r["interpolate"] == pytest.approx(1.0) and r["array"] == pytest.approx(2.0)
    assert r["sub"] == pytest.approx(4.0) and r["function"] == pytest.approx(2.0)
    assert (r["syncs"], r["syncs_after_write"]) == (1, 1)
    assert r["ds_before"] == 0.0 and r["ds_after"] == pytest.approx(1.0)
    assert r["n_unique"] == 72              # 81 nodes, the 9 on x = 1 slaves
    assert r["bc_rows"] == 5                # y = 0, x >= 1/2: 5 masters
    assert r["poisson"] < 1e-3
    assert r["interpolate_vs_dolfinx"] < 1e-14
    assert r["expressions_share_space"] and r["sub_made_once"]


def test_one_rank(tmp_path):
    _check(_run(tmp_path, 1))


@pytest.mark.mpi
def test_three_ranks(tmp_path):
    _check(_run(tmp_path, 3))
