"""tests/test_dolfinx_lifetime.py's worker: meshes the backend's caches have seen (compiled
forms, constant matrices, linear solvers, subfunctions, nodal values) are freed once the
caller drops them. Prints `RESULT <meshes still alive> <compiled once>`. The default mesh
(set_default_mesh; else the first mesh made) is held by design, so it is set first to one
outside the levels."""
import gc
import weakref

from femtools.backend.dolfinx import (Expression, FiniteElement, Function, FunctionSpace,
                                      MixedElement, TestFunction, TrialFunction,
                                      UnitSquareMesh, assemble, assemble_matrix,
                                      collect_garbage, dx, interpolate, nodal_values,
                                      set_default_mesh, solve, subfunction)


set_default_mesh(UnitSquareMesh(2, 2))


def level(n):
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, MixedElement([FiniteElement("CG", mesh.ufl_cell(), 1)] * 2))
    u, v = TrialFunction(V), TestFunction(V)
    f = interpolate(Expression("x[0]", degree=1, domain=mesh), V)
    a = f * dx
    assemble(a)
    assemble_matrix(u * v * dx)
    w = Function(V)
    solve(u * v * dx == f * v * dx, w)
    subfunction(Function(W), 0)
    nodal_values(f, V)
    # a form assembled again on a live mesh is compiled once: the same entry
    import femtools.backend.dolfinx as B
    once = B.compiled(a) is B.compiled(a)
    return weakref.ref(mesh), once


refs = [level(n) for n in (4, 6, 8)]
collect_garbage()
gc.collect()
print("RESULT", sum(r() is not None for r, _ in refs), all(o for _, o in refs), flush=True)
