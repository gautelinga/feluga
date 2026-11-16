"""`femtools.dolfinx.checkpoint`: DOLFINx Functions in dolfin's HDF5 layout, across the
backends and the numbers of ranks.

dolfin writes on one rank, DOLFINx reads on three; DOLFINx writes on three, dolfin reads
on one and DOLFINx on two. dolfin also reads through the reader DOLFINx uses
(femtools.tables.read_values, femtools.dolfin_layout.read_function): both backends' files,
on one rank and on two. Scalar P1 and P2, mixed P2xP2 and P2xP1, a P2 vector, periodic in
x, P2 without periodicity, and Taylor-Hood (a vector P2 and a P1) periodic in x too
(2026-11-16; it was run without before); every value read
equals the field at the reader's own dof.

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
SPACES = ("P2", "P1", "W22", "W21", "P2_walls", "V2", "TH")
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


@pytest.mark.mpi
def test_one_reader_on_dolfin(tmp_path):
    """femtools.dolfin_layout.read_function (the reader DOLFINx uses) on dolfin: a file
    dolfin wrote, read on two ranks, and one DOLFINx wrote on three, on one and two."""
    pytest.importorskip("dolfin")
    (tmp_path / "d").mkdir()
    (tmp_path / "x").mkdir()
    _run("write-dolfin", tmp_path / "d", 1)
    _check(_run("read-dolfin-tables", tmp_path / "d", 2, tmp_path / "dt2.json"))
    _run("write-dolfinx", tmp_path / "x", 3)
    _check(_run("read-dolfin-tables", tmp_path / "x", 1, tmp_path / "xt1.json"))
    _check(_run("read-dolfin-tables", tmp_path / "x", 2, tmp_path / "xt2.json"))
