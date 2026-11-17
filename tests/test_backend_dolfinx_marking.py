"""dolfin's DirichletBC(V, value, "C++ code") on DOLFINx (femtools.backend.dolfinx
.compiled_facets): the facets dolfin's CompiledSubDomain marks, compared by their
midpoints on the same unit square, with and without on_boundary; and what else of
dolfin's marking and DirichletBC the backend keeps: the midpoint check of
BoundaryMarkers.mark, the pointwise method (on a scalar space and on a mixed space's
components), the refusal of another mesh's markers, and an Expression in a given
element."""
import numpy as np
import pytest

pytestmark = [pytest.mark.dolfinx]

CODES = ["std::abs(x[1] - 1) < 1e-8",          # PlaneCouette's top wall
         "near(x[0], 0.5)",                     # an interior line: dolfin marks it too
         "on_boundary && x[0] < 0.5 + DOLFIN_EPS",
         "on_boundary"]


def _dolfin_midpoints(code):
    dolfin = pytest.importorskip("dolfin")
    mesh = dolfin.UnitSquareMesh(dolfin.MPI.comm_self, 8, 8)
    marked = dolfin.MeshFunction("size_t", mesh, 1, 0)
    dolfin.CompiledSubDomain(code).mark(marked, 1)
    return sorted(tuple(np.round(f.midpoint().array()[:2], 12))
                  for f in dolfin.facets(mesh) if marked[f] == 1)


def _dolfinx_midpoints(code):
    pytest.importorskip("dolfinx")
    import dolfinx.mesh as dm
    from mpi4py import MPI
    from femtools.backend.dolfinx import compiled_facets
    mesh = dm.create_unit_square(MPI.COMM_SELF, 8, 8)
    facets = compiled_facets(mesh, code)
    mid = dm.compute_midpoints(mesh, 1, facets)[:, :2]
    return sorted(tuple(np.round(m, 12)) for m in mid)


@pytest.mark.parametrize("code", CODES)
def test_same_facets_as_dolfin(code):
    expected = _dolfin_midpoints(code)
    assert expected, code
    assert _dolfinx_midpoints(code) == expected


def _dolfinx_backend():
    pytest.importorskip("dolfinx")
    from femtools.backend import dolfinx as b
    return b


def test_marking_checks_the_midpoint():
    """A predicate true at a facet's ends but not at its midpoint marks it on neither
    backend (dolfin's SubDomain.mark checks the midpoint too)."""
    dolfin = pytest.importorskip("dolfin")
    b = _dolfinx_backend()

    def inside(x):
        return abs(x[0] - 0.5) > 0.2 and x[1] < 1e-8

    mesh = dolfin.UnitSquareMesh(dolfin.MPI.comm_self, 1, 1)
    marked = dolfin.MeshFunction("size_t", mesh, 1, 0)

    class Ends(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and inside(x)

    Ends().mark(marked, 3)
    m = b.BoundaryMarkers(b.UnitSquareMesh(1, 1), 0)
    m.mark(lambda x: (np.abs(x[0] - 0.5) > 0.2) & (x[1] < 1e-8), 3)
    assert int((marked.array() == 3).sum()) == len(m.where(3)) == 0


def test_pointwise_pins_a_dof():
    b = _dolfinx_backend()
    Q = b.FunctionSpace(b.UnitSquareMesh(4, 4), "CG", 1)
    bc = b.DirichletBC(Q, 0.0, "near(x[0], 0.0) && near(x[1], 0.0)", "pointwise")
    assert len(bc.get_boundary_values()) == 1
    with pytest.raises(ValueError, match="method"):
        b.DirichletBC(Q, 0.0, "near(x[0], 0.0)", "nearby")


def test_pointwise_on_mixed_components():
    """A pointwise condition on the pressure W.sub(1) and on the velocity W.sub(0) of a
    Taylor-Hood space pins the dofs at the point: one pressure dof, one a velocity
    component, each among that component's own."""
    b = _dolfinx_backend()
    mesh = b.UnitSquareMesh(4, 4)
    cell = mesh.ufl_cell()
    W = b.FunctionSpace(mesh, b.MixedElement([b.VectorElement("CG", cell, 2),
                                              b.FiniteElement("CG", cell, 1)]))
    at = "near(x[0], 0.0) && near(x[1], 0.0)"
    p = b.DirichletBC(W.sub(1), 0.0, at, "pointwise").global_rows()
    u = b.DirichletBC(W.sub(0), (0.0, 0.0), at, "pointwise").global_rows()
    assert len(p) == 1 and np.isin(p, b.owned_dofs(W, 1)).all()
    assert len(u) == 2
    assert [int(np.isin(u, b.owned_dofs(W, (0, j))).sum()) for j in range(2)] == [1, 1]


def test_markers_of_another_mesh_refused():
    b = _dolfinx_backend()
    m = b.BoundaryMarkers(b.UnitSquareMesh(4, 4), 0)
    b.Faces(1, 0.0).mark(m, 2)
    V = b.FunctionSpace(b.UnitSquareMesh(8, 8), "CG", 1)
    with pytest.raises(ValueError, match="another mesh"):
        b.DirichletBC(V, 0.0, m, 2)


def test_expression_in_its_element():
    b = _dolfinx_backend()
    mesh = b.UnitSquareMesh(4, 4)
    e = b.Expression("x[0]", element=b.FiniteElement("CG", mesh.ufl_cell(), 1), domain=mesh)
    assert e.function_space.ufl_element().embedded_superdegree == 1
    assert b.assemble(e * b.dx) == pytest.approx(0.5)
