"""`femtools.dolfinx.xdmf.XDMFSeries`: DOLFINx time series in XDMFFile2's layout.

DOLFINx writes two steps of a periodic P2 scalar, a P1 scalar and a periodic P2 vector,
on one rank and on three; dolfin's XDMFFile2 reads every step back into its own spaces
(the periodic ones included) equal to the fields to round-off, and the XML is the one
XDMFFile2 writes for the same fields, word for word but for the HDF5 file's name.

    python3 -m pytest tests/test_dolfinx_xdmf.py -m "dolfinx and dolfin"
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
WORKER = os.path.join(HERE, "dolfinx_xdmf_worker.py")
FIELDS = ("s2", "s1", "v2")


def _run(mode, folder, ranks, out=None):
    argv = [sys.executable, WORKER, mode, str(folder)] + ([str(out)] if out else [])
    support.run(argv, ranks, env={"PYTHONPATH": ROOT})
    return json.loads(out.read_text()) if out else None


def _check(folder, ranks):
    pytest.importorskip("dolfin")
    _run("write-dolfinx", folder, ranks)
    _run("write-dolfin", folder, 1)
    res = _run("read-dolfin", folder, 1, folder / "res.json")
    for name in FIELDS:
        assert res[name] < 1e-13, name
        assert res[name + "_structure"], name + ": not XDMFFile2's XML"
        assert res[name + "_steps"] == 2, name


def test_series_serial(tmp_path):
    _check(tmp_path, 1)


@pytest.mark.mpi
def test_series_three_ranks(tmp_path):
    _check(tmp_path, 3)


@pytest.mark.parametrize("ranks", [1, 3])
def test_read_on_dolfinx(tmp_path, ranks):
    """DOLFINx reads every step of both backends' series (written on three ranks and by
    dolfin) into its own spaces, periodic ones included, to round-off."""
    pytest.importorskip("dolfin")
    _run("write-dolfinx", tmp_path, 3)
    _run("write-dolfin", tmp_path, 1)
    res = _run("read-dolfinx", tmp_path, ranks, tmp_path / "res.json")
    for name in FIELDS:
        assert res[name] < 1e-13, name


def test_read_refuses_other_components(tmp_path):
    """A field of another number of components than u, or a mixed u, is refused, not
    read through its first column."""
    import basix.ufl
    from dolfinx import fem, mesh as dmesh
    from mpi4py import MPI
    from femtools.dolfinx.xdmf import XDMFSeries
    from femtools.dolfinx.xdmf_read import read_nodal
    msh = dmesh.create_unit_square(MPI.COMM_SELF, 3, 3)
    v = fem.Function(fem.functionspace(msh, ("Lagrange", 2, (2,))), name="v")
    s = fem.Function(fem.functionspace(msh, ("Lagrange", 2)), name="s")
    path = str(tmp_path / "f.xdmf")
    with XDMFSeries(MPI.COMM_SELF, path) as f:
        f.write(v, 0.0)
        f.write(s, 0.0)
    with pytest.raises(ValueError):
        read_nodal(path, "v", -1, fem.Function(s.function_space))
    with pytest.raises(ValueError):
        read_nodal(path, "s", -1, fem.Function(v.function_space))
    el = basix.ufl.element("Lagrange", "triangle", 2)
    W = fem.functionspace(msh, basix.ufl.mixed_element([el, el]))
    with pytest.raises(NotImplementedError):
        read_nodal(path, "s", -1, fem.Function(W))
    read_nodal(path, "v", -1, fem.Function(v.function_space))
