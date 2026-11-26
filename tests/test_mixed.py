"""`feluga.mixed`: a component of a mixed Function read and set through its
collapsed space, built once; a vector Function put together from scalar ones.

    python3 -m pytest tests/test_mixed.py
"""
import numpy as np
import pytest

df = pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

from feluga.mixed import (  # noqa: E402
    AssignedVectorFunction, set_subfunction, subfunction, subspace)
from feluga.periodic import periodic_box  # noqa: E402


def mixed():
    mesh = df.UnitSquareMesh(6, 6)
    P1 = df.FiniteElement("P", mesh.ufl_cell(), 1)
    return df.FunctionSpace(mesh, df.MixedElement([P1, P1]))


def test_components_read_and_set():
    W = mixed()
    q = df.Function(W)
    set_subfunction(q, 1, df.Expression("x[0] + 2*x[1]", degree=1))
    g = subfunction(q, 1)
    assert g.function_space().id() == subspace(W, 1).id()
    assert abs(g(0.5, 0.25) - 1.0) < 1e-12 and subfunction(q, 0).vector().norm("linf") == 0
    g.vector()[:] = 3.
    set_subfunction(q, 0, g)
    assert np.allclose(q.split(deepcopy=True)[0].vector().get_local(), 3.)


def test_built_once():
    W = mixed()
    q = df.Function(W)
    assert subspace(W, 0) is subspace(W, 0) and subfunction(q, 0) is subfunction(q, 0)


def test_subfunction_into_own_function():
    """`out`: a Function of one's own, not the shared one a later call overwrites."""
    W = mixed()
    q = df.Function(W)
    set_subfunction(q, 0, df.Expression("x[0]", degree=1))
    mine = df.Function(subspace(W, 0))
    assert subfunction(q, 0, out=mine) is mine and mine is not subfunction(q, 0)
    set_subfunction(q, 0, df.Expression("2*x[0]", degree=1))
    subfunction(q, 0)
    assert abs(mine(0.5, 0.5) - 0.5) < 1e-12
    with pytest.raises(ValueError):
        subfunction(q, 0, out=df.Function(df.FunctionSpace(W.mesh(), "P", 2)))


def test_assigned_vector_function():
    mesh = df.UnitSquareMesh(6, 6)
    V = df.FunctionSpace(mesh, "P", 2,
                         constrained_domain=periodic_box([1, 1], periodic=[True, False]))
    u = df.as_vector([df.interpolate(df.Expression(e, degree=2), V)
                      for e in ("sin(2*pi*x[0])", "x[1]")])
    uv = AssignedVectorFunction(u, name="u")
    uv()
    assert uv.name() == "u" and uv.function_space().dofmap().constrained_domain is not None
    assert np.allclose([uv.sub(0)(0.3, 0.6), uv.sub(1)(0.3, 0.6)], [np.sin(0.6 * np.pi), 0.6],
                       atol=1e-2)
