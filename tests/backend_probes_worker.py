"""Worker for tests/test_backend_probes.py: femtools.backend's Probes and GradProbes on
dolfin or DOLFINx.

    [mpirun -n k] python3 tests/backend_probes_worker.py BACKEND OUT.npz

On the unit cube (4^3, tetrahedra), at 40 seeded random points and one outside: f, a
quadratic in P2 (values and gradients, two snapshots, the second of 2f), a linear
vector in P1 (gradients) and a DG0 field (its cell means); and at 48 points on cell faces
the DG0 field of the cells' centroid z, which differs across those faces. Rank 0 saves
the gathered arrays.
"""
import sys

import numpy as np

from probe_points import face_points, points

F = "x[0]*x[0] + 2*x[1]*x[2] - x[2]"
U = ("x[0] + 2*x[1]", "3*x[2] - x[0]", "x[1]")


def main(backend, out_path):
    if backend == "dolfinx":
        from femtools.backend import dolfinx as be
    else:
        from femtools.backend import dolfin as be
    mesh = be.BoxMesh(be.Point(0, 0, 0), be.Point(1, 1, 1), 4, 4, 4)
    P2 = be.FunctionSpace(mesh, "Lagrange", 2)
    V1 = be.FunctionSpace(mesh, be.VectorElement("Lagrange", mesh.ufl_cell(), 1))
    D0 = be.FunctionSpace(mesh, "DG", 0)
    f = be.interpolate(be.Expression(F, degree=2), P2)
    f2 = be.Function(P2)
    f2.vector().set_local(2 * f.vector().get_local())
    f2.vector().apply("insert")
    u = be.interpolate(be.Expression(U, degree=1), V1)
    v = be.TestFunction(D0)
    m = be.Function(D0)
    m.vector().set_local(be.assemble(f * v * be.dx).get_local()
                         / be.assemble(v * be.dx).get_local())
    m.vector().apply("insert")
    x = points().flatten()
    g = be.GradProbes(x, P2)
    g.grad(f)
    g.grad(f2)
    gu = be.GradProbes(x, V1)
    gu.grad(u)
    p = be.Probes(x, D0)
    p(m)
    z = be.interpolate(be.Expression("x[2]", degree=1), D0)         # centroids' z
    pf = be.Probes(face_points().flatten(), D0)
    pf(z)
    out = dict(f=g.array(), df=g.array_grad(), u=gu.array(0), du=gu.array_grad(0),
               m=p.array(0), face=pf.array(0))
    g.dump(out_path[:-4] + "_f")
    be.mesh_comm(mesh).barrier()
    if be.mesh_comm(mesh).rank == 0:
        from femtools.probes import read_dump
        dumped = read_dump(out_path[:-4] + "_f")
        out["dump_ids"] = np.array(sorted(dumped))
        out["dump_f"] = np.array([dumped[i][1][:, 0] for i in sorted(dumped)])
        out["dump_x"] = np.array([dumped[i][0] for i in sorted(dumped)])
        np.savez(out_path, **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
