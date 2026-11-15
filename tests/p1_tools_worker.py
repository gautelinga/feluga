"""tests/test_p1_tools.py's worker: on backend argv[1], in folder argv[2], stage argv[3]:
fast_interpolation.p1_to_p2, cell_mean.cell_mean_norm, recovery.recovered_vertex_gradient
and vertex_field on a box periodic in x and y read from a mesh file. Writes
<backend>-<ranks>.json (rank 0) and, in stage "write", the vertex field; in stage "read",
reads the other backend's."""
import importlib
import json
import os
import sys

import numpy as np

backend, folder, stage = sys.argv[1:4]
# the backend before mpi4py: legacy dolfin imported after it calls MPI at exit, after
# mpi4py has finalised MPI, and the process aborts
L = importlib.import_module("femtools.backend." + backend)
from mpi4py import MPI  # noqa: E402
from femtools import cell_mean, recovery, vertex_field  # noqa: E402
from femtools.fast_interpolation import p1_to_p2  # noqa: E402

comm = MPI.COMM_WORLD
mesh = L.read_mesh_file(os.path.join(folder, "mesh.h5"))
box = L.PeriodicBox({0: (0., 1.), 1: (0., 1.)})
S1 = L.FunctionSpace(mesh, "CG", 1, constrained_domain=box)
S2 = L.FunctionSpace(mesh, "CG", 2, constrained_domain=box)
# periodic to the bit: cos(2 pi) is 1, so a vertex and its periodic image give one value
# whichever of them a partition evaluates the dof at (sin(2 pi) is -2.4e-16, not 0)
FIELD = "cos(2*pi*x[0])*cos(2*pi*x[1]) + x[2]*x[2]"
out = dict(ranks=comm.size)


def gmax(a):
    return comm.allreduce(float(np.max(np.abs(a), initial=0.0)), op=MPI.MAX)


# p1_to_p2: one matrix-vector product, against the backend's interpolation (dolfin: the
# geometric transfer matrix, which p1_to_p2 replaced)
f1 = L.interpolate(L.Expression(FIELD, degree=2), S1)
P = p1_to_p2(S1, S2)
g = L.Function(S2)
P.mult(L.as_backend_type(f1.vector()).vec(), L.as_backend_type(g.vector()).vec())
if backend == "dolfin":
    import dolfin as df
    T = df.PETScDMCollection.create_transfer_matrix(S1, S2).mat()
    ref = T.createVecLeft()
    T.mult(L.as_backend_type(f1.vector()).vec(), ref)
    out["p1_to_p2"] = gmax(L.as_backend_type(g.vector()).vec().getArray() - ref.getArray())
else:
    out["p1_to_p2"] = gmax(g.vector().get_local() - L.interpolate(f1, S2).vector().get_local())
out["p1_to_p2_scale"] = gmax(g.vector().get_local())
_, _, w = P.getValuesCSR()
sums = P.createVecLeft()
ones = P.createVecRight()
ones.set(1.0)
P.mult(ones, sums)
out["p1_to_p2_exact"] = bool(comm.allreduce(bool(np.isin(w, (0.5, 1.0)).all()
                                                 and (sums.getArray() == 1.0).all()),
                                            op=MPI.LAND))

# cell_mean_norm against the assembled means, P1 and P2, vector and three scalars
DG0 = L.FunctionSpace(mesh, "DG", 0)
w = L.TestFunction(DG0)
vol = L.assemble(w * L.dx).get_local()
U = ("x[1]*x[2] - 0.3", "x[0]*x[0] + 0.1", "sin(x[0] + x[1]) - x[2]")
errs = []
for k in (1, 2):
    S = S1 if k == 1 else S2
    comps = [L.interpolate(L.Expression(c, degree=2), S) for c in U]
    u = L.as_vector(comps)
    mean = np.stack([L.assemble(c * w * L.dx).get_local() / vol for c in comps], axis=1)
    exact = np.sqrt((mean ** 2).sum(axis=1))
    errs.append(gmax(cell_mean.cell_mean_norm(u, DG0).vector().get_local() - exact))
out["cell_mean"] = max(errs)
try:
    S3 = L.FunctionSpace(mesh, "CG", 3)
    cell_mean.cell_mean_norm(L.as_vector([L.Function(S3)] * 3), DG0)
    out["cell_mean_p3_refused"] = False
except (ValueError, RuntimeError):
    out["cell_mean_p3_refused"] = True

# the recovered gradient at the vertices, by coordinate, to compare across backends
grad = recovery.recovered_vertex_gradient(mesh, f1, S1)
x = L.mesh_coordinates(mesh)
X = np.concatenate(comm.allgather(np.asarray(x[:len(grad)], dtype=float)))
G = np.concatenate(comm.allgather(grad))
if comm.rank == 0:
    # a vertex several ranks hold comes once (they agree on it: its dof is one)
    _, first = np.unique(np.round(X, 9), axis=0, return_index=True)
    order = first[np.lexsort(np.round(X[first], 9).T[::-1])]
    np.savez(os.path.join(folder, "grad-{}-{}.npz".format(backend, comm.size)),
             x=X[order], g=G[order])

# the vertex field: a P1 Function whose value at each vertex is the vertex's row in the
# mesh file, written by vertex and read back by either backend
V1 = L.FunctionSpace(mesh, "CG", 1)


def rows_field():
    """A Function on V1 holding each vertex's file row."""
    f = L.Function(V1)
    if backend == "dolfin":
        import dolfin as df
        v2d = df.vertex_to_dof_map(V1)
        gvi = np.asarray(mesh.topology().global_indices(0), dtype=float)
        vals = f.vector().get_local()
        own = v2d < len(vals)
        vals[v2d[own]] = gvi[own]
        f.vector().set_local(vals)
        f.vector().apply("insert")
        return f
    dofs = np.asarray(V1.dofmap.list).ravel()
    nodes = np.asarray(mesh.geometry.dofmap).ravel()
    f.x.array[dofs] = np.asarray(mesh.geometry.input_global_indices, dtype=float)[nodes]
    return f.restrict()


rows = rows_field()
mine = os.path.join(folder, "field-{}-{}.h5".format(backend, comm.size))
if stage == "write":
    vertex_field.write_function(mine, rows, field="delta")
    import h5py
    with h5py.File(mine, "r") as h5:
        stored = np.asarray(h5["delta/vertex_values"])
    out["vertex_field_in_file_order"] = bool(np.array_equal(stored, np.arange(len(stored))))
    path = mine
else:
    other = "dolfinx" if backend == "dolfin" else "dolfin"
    path = os.path.join(folder, [n for n in os.listdir(folder)
                                 if n.startswith("field-" + other)][0])
h = L.Function(V1)
vertex_field.read_function(path, h, field="delta")
out["vertex_field_read"] = gmax(h.vector().get_local() - rows.vector().get_local())
if comm.rank == 0:
    with open(os.path.join(folder, "{}-{}-{}.json".format(backend, comm.size, stage)), "w") as fh:
        json.dump(out, fh)
# PETSc objects destroyed before the interpreter ends (else one may outlive MPI)
P.destroy()
if backend == "dolfin":
    T.destroy()
L.collect_garbage()
