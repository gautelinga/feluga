"""`femtools.fast_assembly`: assemble gives dolfin.assemble's matrices and vectors,
with cell and facet integrals, for a form compiled once; a coefficient changed in
place is assembled with its new values; each form object is compiled once."""
import numpy as np
import pytest

df = pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

from femtools import fast_assembly as fa  # noqa: E402


def setup():
    mesh = df.UnitSquareMesh(6, 6)
    V = df.FunctionSpace(mesh, "CG", 2)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    mu = df.interpolate(df.Expression("1 + x[0]*x[1]", degree=2), V)
    a = mu * df.inner(df.grad(u), df.grad(v)) * df.dx + mu * u * v * df.ds
    L = mu * v * df.dx + df.avg(mu) * df.avg(v) * df.dS
    return V, mu, a, L


def same(x, y):
    return np.abs(x - y).max() < 1e-12


def test_matrix_and_vector():
    V, mu, a, L = setup()
    A, b = fa.assemble(a), fa.assemble(L)
    fa.assemble(a, tensor=A)
    fa.assemble(L, tensor=b)
    assert same(A.array(), df.assemble(a).array())
    assert same(b.get_local(), df.assemble(L).get_local())


def test_coefficients_changed_in_place():
    V, mu, a, L = setup()
    A, b = fa.assemble(a), fa.assemble(L)
    fa.assemble(a, tensor=A)
    mu.vector()[:] = 3.0
    fa.assemble(a, tensor=A)
    fa.assemble(L, tensor=b)
    assert same(A.array(), df.assemble(a).array())
    assert same(b.get_local(), df.assemble(L).get_local())


def test_compiled_once():
    V, mu, a, L = setup()
    A = fa.assemble(a)
    fa.assemble(a, tensor=A)
    n = len(fa._COMPILED)
    fa.assemble(a, tensor=A)
    assert len(fa._COMPILED) == n


def test_forms_built_in_the_call_do_not_fill_the_cache():
    V, mu, a, L = setup()
    b = fa.assemble(L)
    for _ in range(fa.CAPACITY + 10):
        fa.assemble(mu * df.TestFunction(V) * df.dx, tensor=b)
    assert len(fa._COMPILED) <= fa.CAPACITY


def test_scalar():
    """A functional assembled into a Scalar gives a number, as dolfin.assemble does."""
    V, mu, a, L = setup()
    M = mu * df.dx
    value = fa.assemble(M, tensor=df.cpp.la.Scalar(V.mesh().mpi_comm()))
    assert isinstance(value, float) and abs(value - df.assemble(M)) < 1e-14


def test_assemble_matrix_once_per_form_and_conditions():
    """An equal form built again finds the same matrix; other conditions another."""
    V, mu, a, L = setup()
    u, v = df.TrialFunction(V), df.TestFunction(V)
    bc = df.DirichletBC(V, df.Constant(0.), "on_boundary")
    M = fa.assemble_matrix(u * v * df.dx)
    assert fa.assemble_matrix(df.TrialFunction(V) * df.TestFunction(V) * df.dx) is M
    Mbc = fa.assemble_matrix(u * v * df.dx, [bc])
    expected = df.assemble(u * v * df.dx)
    bc.apply(expected)
    assert Mbc is not M and same(Mbc.array(), expected.array())
    with pytest.raises(ValueError):
        fa.assemble_matrix(v * df.dx)
    # conditions on the same rows with other values (a copy, homogenized) apply the same
    # rows to the matrix: the same matrix; conditions on other rows another
    same_rows = df.DirichletBC(bc)
    same_rows.homogenize()
    assert fa.assemble_matrix(u * v * df.dx, [same_rows]) is Mbc
    left = df.DirichletBC(V, df.Constant(1.), "near(x[0], 0.0)")
    assert fa.assemble_matrix(u * v * df.dx, [left]) is not Mbc
