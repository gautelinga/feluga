"""Worker for tests/test_dolfinx_legacy.py: femtools.dolfinx.legacy's dolfin semantics,
results as JSON (rank 0)."""
import json
import sys

import numpy as np

import femtools.dolfinx.legacy as L


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
    if comm.rank == 0:
        with open(out, "w") as f:
            json.dump(r, f)


if __name__ == "__main__":
    main(sys.argv[1])
