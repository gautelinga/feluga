"""`femtools.dolfinx.checkpoint`: DOLFINx Functions in dolfin's HDF5 layout, across the
backends and the numbers of ranks.

dolfin writes on one rank, DOLFINx reads on three; DOLFINx writes on three, dolfin reads
on one and DOLFINx on two. Scalar P1 and P2, mixed P2xP2 and P2xP1, a P2 vector, periodic in
x, and P2 without periodicity; every value read equals the field at the reader's own dof.

    python3 -m pytest tests/test_dolfinx_checkpoint.py -m "dolfinx and dolfin"
"""
import json
import os
import sys

import pytest

pytest.importorskip("dolfinx")
pytestmark = [pytest.mark.dolfinx, pytest.mark.dolfin]

from tests import support  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKER = os.path.join(HERE, "dolfinx_checkpoint_worker.py")
SPACES = ("P2", "P1", "W22", "W21", "P2_walls", "V2")
TOL = 1e-13


def _run(mode, folder, ranks, out=None):
    argv = [sys.executable, WORKER, mode, str(folder)] + ([str(out)] if out else [])
    support.run(argv, ranks, env={"PYTHONPATH": ROOT})
    return json.loads(out.read_text()) if out else None


def _check(res):
    assert sorted(res) == sorted(SPACES)
    for name, d in res.items():
        assert d < TOL, name


def test_dolfin_written_read_on_dolfinx(tmp_path):
    pytest.importorskip("dolfin")
    _run("write-dolfin", tmp_path, 1)
    _check(_run("read-dolfinx", tmp_path, 1, tmp_path / "x1.json"))


@pytest.mark.mpi
def test_dolfin_written_read_on_dolfinx_three_ranks(tmp_path):
    pytest.importorskip("dolfin")
    _run("write-dolfin", tmp_path, 1)
    _check(_run("read-dolfinx", tmp_path, 3, tmp_path / "x3.json"))


@pytest.mark.mpi
def test_dolfinx_written_on_three_ranks(tmp_path):
    pytest.importorskip("dolfin")
    _run("write-dolfinx", tmp_path, 3)
    _check(_run("read-dolfin", tmp_path, 1, tmp_path / "d1.json"))
    _check(_run("read-dolfinx", tmp_path, 2, tmp_path / "x2.json"))
    _check(_run("read-dolfinx", tmp_path, 1, tmp_path / "x1.json"))
