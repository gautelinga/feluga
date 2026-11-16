"""tests/test_backend_calls.py's worker: the same calls on backend argv[1], files in argv[2];
one JSON line on rank 0 with what each call did."""
import importlib
import json
import os
import sys

L = importlib.import_module("femtools.backend." + sys.argv[1])  # before mpi4py (dolfin)
import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

comm = MPI.COMM_WORLD
folder = sys.argv[2]
out = {}
mesh = L.UnitSquareMesh(4, 4)
cell = mesh.ufl_cell()
P2, P1 = L.FiniteElement("CG", cell, 2), L.FiniteElement("CG", cell, 1)

# MixedElement one by one and as a list: the same space (81 P2 nodes and 25 vertices)
W = L.FunctionSpace(mesh, L.MixedElement(P2, P1))
W_list = L.FunctionSpace(mesh, L.MixedElement([P2, P1]))
out["mixed_dims"] = [int(L.global_dim(W)), int(L.global_dim(W_list))]

# split(deepcopy=True): copies on the collapsed spaces, independent of the mixed Function
q = L.Function(W)
qv = q.vector()
qv.set_local(np.ones(qv.local_size()))
qv.apply("insert")
a, b = q.split(deepcopy=True)
out["split_values"] = [L.assemble(a * L.dx), L.assemble(b * L.dx)]   # the unit square: 1
av = a.vector()
av.set_local(np.zeros(av.local_size()))
av.apply("insert")
out["split_independent"] = L.assemble(L.split(q)[0] * L.dx)          # still 1

# norm(f, "L2", mesh=mesh): f = x, its L2 norm sqrt(1/3)
V1 = L.FunctionSpace(mesh, "CG", 1)
f = L.interpolate(L.Expression("x[0]", degree=1), V1)
out["norm"] = L.norm(f, "L2", mesh=mesh)

# the Krylov solver's three calls: u - lap u = 1 (natural conditions), so u = 1
u, v = L.TrialFunction(V1), L.TestFunction(V1)
A = L.assemble(u * v * L.dx + L.inner(L.grad(u), L.grad(v)) * L.dx)
rhs = L.assemble(L.Constant(1.0) * v * L.dx)
xs = []
for how in ("A, x, b", "set_operator", "set_operators"):
    s = L.PETScKrylovSolver("cg", "jacobi")
    s.parameters["relative_tolerance"] = 1e-12
    s.parameters["absolute_tolerance"] = 1e-14
    x = L.Function(V1)
    if how == "A, x, b":
        s.solve(A, x.vector(), rhs)
    elif how == "set_operator":
        s.set_operator(A)
        s.solve(x.vector(), rhs)
    else:
        s.set_operators(A, A)
        s.solve(x.vector(), rhs)
    L.sync(x)
    xs.append(x)
out["krylov_mean"] = [L.assemble(x * L.dx) for x in xs]
out["krylov_same"] = max(float(comm.allreduce(np.abs(x.vector().get_local()
                                                     - xs[0].vector().get_local()).max(initial=0),
                                              op=MPI.MAX)) for x in xs)

# write_mesh_h5 with name and mode: two meshes in one file
h5 = os.path.join(folder, "meshes.h5")
L.write_mesh_h5(h5, mesh, name="first", mode="w")
L.write_mesh_h5(h5, mesh, name="second", mode="a")
comm.Barrier()
if comm.rank == 0:
    import h5py
    with h5py.File(h5, "r") as fh:
        out["mesh_groups"] = sorted(fh.keys())

# XDMFFile: a P2 Function keeps its midside nodes; facet markers are written too
f2 = L.interpolate(L.Expression("x[0]*x[1]", degree=2), L.FunctionSpace(mesh, "CG", 2))
f2.rename("f", "f")
with L.XDMFFile(comm, os.path.join(folder, "f.xdmf")) as xf:
    xf.write(f2)
markers = L.BoundaryMarkers(mesh, 0)
L.Faces(0, 0.0, 1.0).mark(markers, 3)
with L.XDMFFile(comm, os.path.join(folder, "markers.xdmf")) as xf:
    xf.write(markers)
comm.Barrier()
if comm.rank == 0:
    out["p2_kept"] = "Triangle_6" in open(os.path.join(folder, "f.xdmf")).read()
    out["markers_written"] = os.path.getsize(os.path.join(folder, "markers.xdmf")) > 0
    print("RESULT " + json.dumps(out), flush=True)
L.collect_garbage()
