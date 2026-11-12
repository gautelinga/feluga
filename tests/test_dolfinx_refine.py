"""`femtools.dolfinx.refine.refine_periodic`: DOLFINx's Plaza refinement with periodic
faces bisected alike (femtools.plaza_periodic's port).

A rectangle (crossed diagonals) and a box, refined near the upper and near the lower x
face: the two periodic faces carry the same P2 nodes and every slave finds its master,
on one rank and on three (plain DOLFINx refinement leaves them unequal); the same cells
and vertices as dolfin's plaza_periodic makes from the same input; uniform refinement
makes dolfinx.mesh.refine's cells; frozen edges stay unbisected (and their partners);
max_cells refuses; every child lies in the cell the parent map names.

    python3 -m pytest tests/test_dolfinx_refine.py -m "dolfinx"
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
WORKER = os.path.join(HERE, "dolfinx_refine_worker.py")
LOCAL = ("rect_upper", "rect_lower", "box_upper", "box_lower")


def _run(mode, tmp_path, ranks):
    out = tmp_path / "{}_{}.json".format(mode, ranks)
    support.run([sys.executable, WORKER, mode, str(out)], ranks, env={"PYTHONPATH": ROOT})
    return json.loads(out.read_text())


def _check(r):
    for k in LOCAL:
        assert r[k]["lo"] == r[k]["hi"] and r[k]["paired"], k
        assert r[k]["parents"], k
        lo, hi = r[k]["plain"]
        assert lo != hi, k + ": plain DOLFINx refinement was expected to break the faces"
    for name in ("rect", "box"):
        assert r[name + "_uniform"][0] == r[name + "_uniform"][1]
        f = r[name + "_frozen"]
        assert f["lo"] == f["hi"] == f["before"] and f["paired"] and f["frozen_cells"] > 0
        assert r[name + "_capped"]


def test_serial(tmp_path):
    _check(_run("dolfinx", tmp_path, 1))


@pytest.mark.mpi
def test_three_ranks(tmp_path):
    serial, parallel = _run("dolfinx", tmp_path, 1), _run("dolfinx", tmp_path, 3)
    _check(parallel)
    for k in LOCAL:
        assert parallel[k]["cells"] == serial[k]["cells"], k


@pytest.mark.dolfin
def test_same_as_plaza_periodic(tmp_path):
    pytest.importorskip("dolfin")
    x, d = _run("dolfinx", tmp_path, 1), _run("dolfin", tmp_path, 1)
    for k in LOCAL:
        assert x[k]["cells"] == d[k]["cells"], k
        assert x[k]["vertices"] == d[k]["vertices"], k
