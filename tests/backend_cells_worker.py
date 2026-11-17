"""Worker for tests/test_backend_cells.py: femtools.backend's cell_values, refine_cells,
interpolate across meshes and Expressions off the default mesh, on dolfin or DOLFINx.

    [mpirun -n k] python3 tests/backend_cells_worker.py BACKEND OUT.json

On the unit cube (4^3): the cells whose centroid has x + y > 1 are marked (from a DG0
field, through cell_values) and refined, after markers one short on rank 0 are refused
on every rank; a vector DG0 field's cell_values are (cells, components); a linear field
on the coarse mesh is interpolated onto P1 of the refined one, twice (the second time
from the held point location); a quadratic Expression, made while the coarse mesh is the
default, is interpolated onto P2 of a box twice the size (outside the
coarse mesh). Rank 0 writes the counts, the refined mesh's cells as sorted vertex
positions, and the largest errors.
"""
import json
import sys

import numpy as np


def main(backend, out_path):
    if backend == "dolfinx":
        from femtools.backend import dolfinx as be
    else:
        from femtools.backend import dolfin as be
    from mpi4py import MPI
    mesh = be.BoxMesh(be.Point(0, 0, 0), be.Point(1, 1, 1), 4, 4, 4)
    comm = be.mesh_comm(mesh)
    D = be.FunctionSpace(mesh, "DG", 0)
    c = be.interpolate(be.Expression("x[0] + x[1]", degree=1), D)   # at the centroids
    values = be.cell_values(c)
    marked = values > 1.0
    # a vector DG0 field: (cells, components), its first component the scalar field's
    Dv = be.FunctionSpace(mesh, be.VectorElement("DG", mesh.ufl_cell(), 0))
    cv = be.interpolate(be.Expression(("x[0] + x[1]", "10.0", "-x[2]"), degree=1), Dv)
    vec = be.cell_values(cv)
    vec_ok = bool(vec.shape == (len(values), 3) and np.allclose(vec[:, 0], values)
                  and np.allclose(vec[:, 1], 10.0) and np.all(vec[:, 2] < 0))
    # markers of the wrong length on rank 0 only: refused on every rank
    short = marked[:-1] if comm.rank == 0 else marked
    try:
        be.refine_cells(mesh, short)
        refused = False
    except ValueError:
        refused = True
    fine = be.refine_cells(mesh, marked)

    S = be.FunctionSpace(mesh, "Lagrange", 1)
    f = be.interpolate(be.Expression("1 + 2*x[0] - x[1] + 3*x[2]", degree=1), S)
    Sf = be.FunctionSpace(fine, "Lagrange", 1)
    g = be.interpolate(f, Sf)                                          # across meshes
    exact = be.interpolate(be.Expression("1 + 2*x[0] - x[1] + 3*x[2]", degree=1), Sf)
    g2 = be.interpolate(f, Sf)                                         # again
    err_cross = max(np.abs(gi.vector().get_local() - exact.vector().get_local()).max(initial=0)
                    for gi in (g, g2))

    be.set_default_mesh(mesh)
    expr = be.Expression("x[0]*x[0] - x[1]*x[2]", degree=2)
    big = be.BoxMesh(be.Point(-1, -1, -1), be.Point(1, 1, 1), 3, 3, 3)
    P2 = be.FunctionSpace(big, "Lagrange", 2)
    h = be.interpolate(expr, P2)
    # the quadratic is in P2: its nodal values integrate to the exact integral, 8/3 * 1
    integral = be.assemble(h * be.dx)

    from femtools.tables import space_tables
    t = space_tables(be.FunctionSpace(fine, "Lagrange", 1))
    # a cell as its vertex positions, sorted: the same on either backend's numbering
    cells = [tuple(sorted(map(tuple, np.round(c, 9).tolist()))) for c in t.x]
    allcells = comm.gather(cells, root=0)
    out = dict(
        n_values=comm.allreduce(len(values), op=MPI.SUM),
        n_marked=comm.allreduce(int(marked.sum()), op=MPI.SUM),
        sum_values=comm.allreduce(float(values.sum()), op=MPI.SUM),
        n_fine=be.global_num_cells(fine),
        err_cross=comm.allreduce(float(err_cross), op=MPI.MAX),
        integral=float(integral),
        vec_ok=comm.allreduce(vec_ok, op=MPI.LAND),
        refused=comm.allreduce(refused, op=MPI.LAND))
    if backend == "dolfinx":
        markers = be.BoundaryMarkers(mesh)
        markers.rename("walls", "walls")
        out["renamed"] = markers.meshtags().name
    if comm.rank == 0:
        out["cells"] = sorted(str(c) for part in allcells for c in part)
        with open(out_path, "w") as handle:
            json.dump(out, handle)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
