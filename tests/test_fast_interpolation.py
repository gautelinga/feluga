"""`femtools.fast_interpolation`: interpolate and values give dolfin.interpolate's
values between Lagrange spaces of any degrees, scalar and vector, from a component of
a mixed or a vector Function, on a periodic space and on three ranks; refuse what they
cannot do; and build the transfer once per pair of spaces."""
import os
import sys

import numpy as np
import pytest

df = pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

from femtools import fast_interpolation as fi  # noqa: E402
from femtools.periodic import PeriodicBox  # noqa: E402
from tests import support  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_interpolation_worker.py")
S = "sin(pi*x[0])*x[1] + x[0]*x[0]*x[1]"
VEC = ("sin(pi*x[0])*x[1]", "cos(pi*x[0]) - x[1]*x[1]")


def mesh_and_box():
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(2, 1), 12, 6)
    return mesh, PeriodicBox([2, 1], periodic=[True, False])


def close(f, V):
    exact = df.interpolate(f, V).vector().get_local()
    return max(np.abs(fi.interpolate(f, V).vector().get_local() - exact).max(),
               np.abs(fi.values(f, V) - exact).max()) < 1e-13


@pytest.mark.parametrize("degrees", [(2, 1), (1, 2), (3, 2), (2, 2), (1, 1)])
@pytest.mark.parametrize("periodic", [False, True])
def test_scalar(degrees, periodic):
    mesh, pb = mesh_and_box()
    cd = pb if periodic else None
    source, target = (df.FunctionSpace(mesh, "CG", k, constrained_domain=cd) for k in degrees)
    assert close(df.interpolate(df.Expression(S, degree=4), source), target)


@pytest.mark.parametrize("periodic", [False, True])
def test_components_and_vectors(periodic):
    mesh, pb = mesh_and_box()
    cd = pb if periodic else None
    P1 = df.FunctionSpace(mesh, "CG", 1, constrained_domain=cd)
    el = df.FiniteElement("CG", mesh.ufl_cell(), 2)
    W = df.FunctionSpace(mesh, df.MixedElement([el, el]), constrained_domain=cd)
    w = df.interpolate(df.Expression((S, "cos(pi*x[0]) + x[1]"), degree=4), W)
    u = df.interpolate(df.Expression(VEC, degree=4),
                       df.VectorFunctionSpace(mesh, "CG", 2, constrained_domain=cd))
    for c in w.split():                                         # components of a mixed one
        assert close(c, P1)
    assert close(u.split()[1], P1)                              # a component of a vector
    assert close(u, df.VectorFunctionSpace(mesh, "CG", 1, constrained_domain=cd))
    assert close(u, df.VectorFunctionSpace(mesh, "CG", 2))      # onto a plain space
    t = df.interpolate(df.Expression((VEC, (S, "x[1]")), degree=4),
                       df.TensorFunctionSpace(mesh, "CG", 2, constrained_domain=cd))
    assert close(t, df.TensorFunctionSpace(mesh, "CG", 2))      # a tensor


def test_transfer_is_built_once():
    """Also for the components a split() makes anew on every call."""
    mesh, _ = mesh_and_box()
    el = df.FiniteElement("CG", mesh.ufl_cell(), 2)
    w = df.Function(df.FunctionSpace(mesh, df.MixedElement([el, el])))
    P1 = df.FunctionSpace(mesh, "CG", 1)
    fi.values(w.split()[0], P1)
    n = len(fi._TRANSFERS)
    fi.values(w.split()[0], P1)
    assert len(fi._TRANSFERS) == n


@pytest.mark.parametrize("what", ["DG target", "DG source", "vector onto scalar", "other mesh",
                                  "tensor onto vector", "component target"])
def test_refuses(what):
    mesh, _ = mesh_and_box()
    source = df.FunctionSpace(mesh, "DG", 1) if what == "DG source" else \
        df.VectorFunctionSpace(mesh, "CG", 2) if what == "vector onto scalar" else \
        df.TensorFunctionSpace(mesh, "CG", 2) if what == "tensor onto vector" else \
        df.FunctionSpace(mesh, "CG", 2)
    P1 = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    target = df.FunctionSpace(mesh, "DG", 1) if what == "DG target" else \
        df.FunctionSpace(df.UnitSquareMesh(3, 3), "CG", 1) if what == "other mesh" else \
        df.VectorFunctionSpace(mesh, "CG", 1, dim=4) if what == "tensor onto vector" else \
        df.FunctionSpace(mesh, df.MixedElement([P1, P1])).sub(1) if what == "component target" \
        else df.FunctionSpace(mesh, "CG", 1)
    with pytest.raises(ValueError, match="fast_interpolation"):
        fi.interpolate(df.Function(source), target)


@pytest.mark.mpi
def test_three_ranks(tmp_path):
    out = tmp_path / "err.txt"
    support.run([sys.executable, WORKER, str(out)], 3,
                env={"PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))})
    assert float(out.read_text()) < 1e-13
