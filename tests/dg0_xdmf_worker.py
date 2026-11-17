"""Worker for tests/test_dg0_xdmf.py: a P1 field and DG0 ones written through
femtools.backend's XDMFFile, on dolfin or DOLFINx.

    [mpirun -n k] python3 tests/dg0_xdmf_worker.py BACKEND DIR

On a 2D rectangle: `a` (P1), `m` its cell means (DG0, assembled), `x` the cell centroids
(vector DG0, two components) written one call each at t = 0, into share.xdmf with
`functions_share_mesh` and apart.xdmf without; then a DG1 Function, which must be refused.
Rank 0 writes DIR/BACKEND.json: whether it was.
"""
import json
import os
import sys

BACKEND, FOLDER = sys.argv[1], sys.argv[2]
if BACKEND == "dolfinx":
    from femtools.backend import dolfinx as be
else:
    from femtools.backend import dolfin as be


def main():
    mesh = be.RectangleMesh(be.Point(0.0, 0.0), be.Point(1.0, 2.0), 5, 7)
    S = be.FunctionSpace(mesh, "Lagrange", 1)
    D = be.FunctionSpace(mesh, "DG", 0)
    a = be.Function(S, name="a")
    be.interpolate(be.Expression("x[0]*x[0] + 3*x[1] + x[0]*x[1]", degree=2), S, out=a)
    v = be.TestFunction(D)
    m = be.Function(D, name="m")
    m.vector().set_local(be.assemble(a * v * be.dx).get_local()
                         / be.assemble(v * be.dx).get_local())
    m.vector().apply("insert")
    X = be.FunctionSpace(mesh, be.VectorElement("DG", mesh.ufl_cell(), 0))
    x = be.Function(X, name="x")
    be.interpolate(be.Expression(("x[0]", "x[1]"), degree=1), X, out=x)
    for share in (True, False):
        path = os.path.join(FOLDER, "{}_{}.xdmf".format(BACKEND, "share" if share else "apart"))
        with be.XDMFFile(be.mesh_comm(mesh), path) as f:
            f.parameters["functions_share_mesh"] = share
            for u in (a, m, x):
                f.write(u, 0.0)
    refused = False
    try:
        with be.XDMFFile(be.mesh_comm(mesh), os.path.join(FOLDER, BACKEND + "_dg1.xdmf")) as f:
            f.write(be.Function(be.FunctionSpace(mesh, "DG", 1), name="d"), 0.0)
    except (ValueError, NotImplementedError):
        refused = True
    if be.mesh_comm(mesh).rank == 0:
        with open(os.path.join(FOLDER, BACKEND + ".json"), "w") as out:
            json.dump(dict(dg1_refused=refused), out)


if __name__ == "__main__":
    main()
