"""femtools.backend.dolfin.XDMFFile writes what femtools.xdmf.XDMFFile2 does not lay out
(DG1, P3, a whole mixed Function) through dolfin's own XDMFFile, the file dolfin's writes,
XML and datasets alike; it keeps a file to one writer, refusing such a Function after a grid
of XDMFFile2's and a checkpoint after dolfin's grids.

    python3 -m pytest tests/test_backend_xdmf_fallback.py
"""
import os

import numpy as np
import pytest

pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

import dolfin as df  # noqa: E402
import h5py  # noqa: E402

import femtools.backend.dolfin as L  # noqa: E402


def _function(mesh, kind):
    cell = mesh.ufl_cell()
    V = {"DG1": lambda: df.FunctionSpace(mesh, "DG", 1),
         "P3": lambda: df.FunctionSpace(mesh, "CG", 3),
         "mixed": lambda: df.FunctionSpace(mesh, df.MixedElement(
             [df.VectorElement("CG", cell, 2), df.FiniteElement("CG", cell, 1)]))}[kind]()
    u = df.Function(V)
    u.vector().set_local(np.random.default_rng(1).random(u.vector().local_size()))
    u.vector().apply("insert")
    u.rename("u", "u")
    return u


def _datasets(path):
    with h5py.File(path, "r") as f:
        out = {}
        f.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset)
                     else None)
    return out


@pytest.mark.parametrize("kind", ["DG1", "P3", "mixed"])
def test_dolfins_file(tmp_path, kind):
    mesh = df.UnitCubeMesh(2, 2, 2)
    u = _function(mesh, kind)
    for name, cls in (("dolfin", df.XDMFFile), ("backend", L.XDMFFile)):
        with cls(mesh.mpi_comm(), str(tmp_path / (name + ".xdmf"))) as f:
            f.write(u, 0.0)
            f.write(u, 1.0)
    xml = [open(str(tmp_path / (n + ".xdmf"))).read().replace(n + ".h5", "x.h5")
           for n in ("dolfin", "backend")]
    assert xml[0] == xml[1]
    ours, theirs = (_datasets(str(tmp_path / (n + ".h5"))) for n in ("backend", "dolfin"))
    assert sorted(ours) == sorted(theirs)
    assert all(np.array_equal(ours[k], theirs[k]) for k in ours)


def test_one_writer_a_file(tmp_path):
    mesh = df.UnitCubeMesh(2, 2, 2)
    p2 = df.Function(df.FunctionSpace(mesh, "CG", 2))
    with L.XDMFFile(mesh.mpi_comm(), str(tmp_path / "p2.xdmf")) as f:
        f.write(p2)
        with pytest.raises(ValueError, match="XDMFFile2's grids"):
            f.write(_function(mesh, "DG1"))
    with L.XDMFFile(mesh.mpi_comm(), str(tmp_path / "dg.xdmf")) as f:
        f.write(_function(mesh, "DG1"))
        with pytest.raises(RuntimeError, match="checkpoint"):
            f.write_checkpoint(p2, "p2")
    assert os.path.getsize(str(tmp_path / "dg.h5")) > 0
