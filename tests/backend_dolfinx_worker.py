"""Worker for tests/test_backend_dolfinx.py: femtools.backend.dolfinx's dolfin semantics,
results as JSON (rank 0)."""
import json
import sys

import numpy as np

import femtools.backend.dolfinx as L


def main(out):
    comm = L.MPI.comm_world
    r = {}
    mesh = L.UnitSquareMesh(8, 8)
    box = L.PeriodicBox({0: (0., 1.)})
    V = L.FunctionSpace(mesh, "CG", 1, constrained_domain=box)
    # DOLFINx's writes reach the compact vector
    u = L.Function(V)
    u.interpolate(lambda x: 1 + 0 * x[0])
    r["interpolate"] = L.assemble(u * L.dx)
    u.x.array[:] = 2.0
    u.x.scatter_forward()
    r["array"] = L.assemble(u * L.dx)
    r["n_unique"] = u.vector().size()
    # ... on a mixed space's component view, and by interpolating a Function
    W = L.FunctionSpace(mesh, L.MixedElement([L.FiniteElement("CG", mesh.ufl_cell(), 1)] * 2))
    q = L.Function(W)
    q.sub(1).interpolate(lambda x: 4 + 0 * x[0])
    r["sub"] = L.assemble(L.split(q)[1] * L.dx)
    w = L.Function(V)
    w.interpolate(u)
    r["function"] = L.assemble(w * L.dx)
    # sync copies the compact vector into DOLFINx's array only after it changed
    calls = []
    prolong = w._unique.prolong
    w._unique.prolong = lambda c, f: calls.append(1) or prolong(c, f)
    w.vector()[:] = 1.0
    w.sync()
    w.sync()
    r["syncs"] = len(calls)
    w.vector().axpy(1.0, w.vector())
    w.sync()
    r["syncs_after_write"] = len(calls) - r["syncs"]
    # facets marked after a measure was made count in its integrals
    marks = L.BoundaryMarkers(mesh, 0)
    ds = L.Measure("ds", domain=mesh, subdomain_data=marks)
    one = L.Constant(1.0, mesh)
    r["ds_before"] = L.assemble(one * ds(2))
    L.Faces(1, 0.0).mark(marks, 2)
    r["ds_after"] = L.assemble(one * ds(2))
    # a Dirichlet condition on periodic slaves constrains their masters on any ranks
    m = L.BoundaryMarkers(mesh, 0)
    m.mark(lambda x: (np.abs(x[1]) < 1e-10) & (x[0] > 0.5 - 1e-10), 2)
    r["bc_rows"] = comm.allreduce(len(L.DirichletBC(V, 0.0, m, 2).global_rows()))
    # a periodic Poisson solve: -u'' = f in y, u = 0 on y = 0 and 1, periodic in x
    walls = L.BoundaryMarkers(mesh, 0)
    L.Faces(1, 0.0, 1.0).mark(walls, 1)
    w = L.Function(V)
    v, t = L.TestFunction(V), L.TrialFunction(V)
    L.solve(L.inner(L.grad(t), L.grad(v)) * L.dx == L.Constant(2.0, mesh) * v * L.dx, w,
            [L.DirichletBC(V, 0.0, walls, 1)])
    exact = L.Function(V)
    exact.interpolate(lambda x: x[1] * (1 - x[1]))
    r["poisson"] = abs(L.assemble(w * L.dx) - L.assemble(exact * L.dx))
    # interpolation at dof coordinates equals DOLFINx's own, on scalar, vector, DG0 and
    # P2 spaces; Expressions in one element share one space
    from dolfinx import fem
    worst = 0.0
    for family, degree, shape in (("CG", 1, ()), ("CG", 2, ()), ("CG", 2, (2,)),
                                  ("DG", 0, ())):
        S = fem.functionspace(mesh, (family if family == "DG" else "Lagrange", degree,
                                     shape))
        if shape:
            def f(x):
                return np.stack([np.sin(3 * x[0]) * x[1], x[0] ** 2 - x[1]])
        else:
            def f(x):
                return np.where(x[0] < 0.5, np.sin(3 * x[0]) * x[1], x[0] ** 2 - x[1])
        ours, theirs = fem.Function(S), fem.Function(S)
        L._interpolate(ours, f)
        fem.Function.interpolate(theirs, f)
        worst = max(worst, float(np.abs(ours.x.array - theirs.x.array).max(initial=0)))
    from mpi4py import MPI
    r["interpolate_vs_dolfinx"] = comm.allreduce(worst, op=MPI.MAX)
    e1 = L.Expression("x[0]", degree=2, domain=mesh)
    e2 = L.Expression("x[1]*a", a=2.0, degree=2, domain=mesh)
    r["expressions_share_space"] = e1.function_space is e2.function_space
    # a mesh with an Expression's space is freed with the mesh
    import gc
    import weakref
    other = L.UnitSquareMesh(3, 3)
    L.Expression("x[0]", degree=1, domain=other)
    gone = weakref.ref(other)
    del other
    gc.collect()
    r["expression_mesh_freed"] = gone() is None
    r["sub_made_once"] = W.sub(0) is W.sub(0) and W.sub(1) is not W.sub(0)
    if comm.rank == 0:
        with open(out, "w") as f:
            json.dump(r, f)


if __name__ == "__main__":
    main(sys.argv[1])
