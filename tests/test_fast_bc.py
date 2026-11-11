"""`femtools.fast_bc`: apply gives DirichletBC.apply's vector for a Constant, a vector
Constant and a Function value, follows changes to the value, leaves an Expression and
a component to dolfin, sets a matrix as dolfin does, and agrees on three ranks, ghost
values included."""
import os
import sys

import numpy as np
import pytest

df = pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

from femtools import fast_bc  # noqa: E402
from femtools.periodic import PeriodicBox  # noqa: E402
from tests import support  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_bc_worker.py")


def space(periodic=False, vector=False):
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(2, 1), 12, 6)
    cd = PeriodicBox([2, 1], periodic=[True, False]) if periodic else None
    make = df.VectorFunctionSpace if vector else df.FunctionSpace
    return make(mesh, "CG", 2, constrained_domain=cd)


def same(bc, V):
    """fast_bc.apply and bc.apply on two copies of one vector."""
    n = 2 if V.num_sub_spaces() == 2 else 1
    f = df.interpolate(df.Expression(("x[0]*x[1]", "x[0]")[:n] if n > 1 else "x[0]*x[1]",
                                     degree=2), V)
    a, b = f.vector().copy(), f.vector().copy()
    bc.apply(a)
    fast_bc.apply(bc, b)
    return np.array_equal(a.get_local(), b.get_local())


@pytest.mark.parametrize("periodic", [False, True])
def test_constant(periodic):
    V = space(periodic)
    c = df.Constant(0.3)
    bc = df.DirichletBC(V, c, "on_boundary")
    assert same(bc, V)
    c.assign(7.)                                      # changed in place
    assert same(bc, V)
    bc.set_value(df.Constant(-1.))                    # replaced
    assert same(bc, V)
    assert same(df.DirichletBC(V, 2., "near(x[1], 0)"), V)


def test_vector_constant():
    V = space(vector=True)
    c = df.Constant((1., -2.))
    bc = df.DirichletBC(V, c, "on_boundary")
    assert same(bc, V)
    c.assign(df.Constant((3., 4.)))                   # changed in place
    assert same(bc, V)


def test_function():
    V = space()
    g = df.interpolate(df.Expression("sin(x[0])", degree=2), V)
    bc = df.DirichletBC(V, g, "on_boundary")
    assert same(bc, V)
    g.vector()[:] = 5.                                # changed in place
    assert same(bc, V)


def test_left_to_dolfin():
    """An Expression, changed between calls, and a component of a vector space."""
    V = space()
    e = df.Expression("t*x[0]", t=1., degree=2)
    bc = df.DirichletBC(V, e, "on_boundary")
    assert same(bc, V)
    e.t = 3.
    assert same(bc, V)
    W = space(vector=True)
    assert same(df.DirichletBC(W.sub(0), 1., "on_boundary"), W)


def test_matrix_and_vector():
    V = space()
    u, v = df.TrialFunction(V), df.TestFunction(V)
    form = df.inner(df.grad(u), df.grad(v)) * df.dx
    bc = df.DirichletBC(V, 1., "on_boundary")
    A, B = df.assemble(form), df.assemble(form)
    a = df.assemble(v * df.dx)
    b = a.copy()
    bc.apply(A, a)
    fast_bc.apply(bc, B, b)
    assert np.array_equal(A.array(), B.array())
    assert np.array_equal(a.get_local(), b.get_local())
    fast_bc.apply(bc, B)
    assert np.array_equal(A.array(), B.array())


def test_found_once():
    V = space()
    bc = df.DirichletBC(V, 1., "on_boundary")
    x = df.Function(V).vector()
    fast_bc.apply(bc, x)
    entry = fast_bc._CACHE[id(bc)]
    fast_bc.apply(bc, x)
    assert fast_bc._CACHE[id(bc)] is entry
    c = df.Constant(1.)
    bc = df.DirichletBC(V, c, "on_boundary")
    fast_bc.apply(bc, x)
    dofs = fast_bc._CACHE[id(bc)].dofs
    c.assign(2.)                                      # a new value, the same dofs
    fast_bc.apply(bc, x)
    assert fast_bc._CACHE[id(bc)].dofs is dofs


@pytest.mark.mpi
def test_three_ranks(tmp_path):
    out = tmp_path / "err.txt"
    support.run([sys.executable, WORKER, str(out)], 3,
                env={"PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))})
    assert float(out.read_text()) == 0.
