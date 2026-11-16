"""tests/test_backend_stokes.py's worker: one periodic Taylor-Hood Stokes solve written once
against femtools.backend, on backend argv[1]. Plane Poiseuille: the unit cube, periodic in x
and z, no-slip on y = 0 and y = 1, driven along x by a unit body force; P2 holds the parabola
exactly, so k = mean(u_x) / f = 1/12 to the solver's tolerance. Also the global counts
(global_dim, global_num_cells, global_num_vertices, marker_counts, fixed_dofs). Prints one
JSON line on rank 0."""
import importlib
import json
import sys

L = importlib.import_module("femtools.backend." + sys.argv[1])  # before mpi4py (dolfin)
import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

from femtools.ksp import diagonal_matrix  # noqa: E402

N = 4
comm = MPI.COMM_WORLD
box = L.PeriodicBox({0: (0.0, 1.0), 2: (0.0, 1.0)})
mesh = L.repair_partition(L.BoxMesh(L.Point(0, 0, 0), L.Point(1, 1, 1), N, N, N), box)
cell = mesh.ufl_cell()
W = L.FunctionSpace(mesh, L.MixedElement([L.FiniteElement("CG", cell, 2)] * 3
                                         + [L.FiniteElement("CG", cell, 1)]),
                    constrained_domain=box)
L.check_periodic_pairing(box, [W])
markers = L.BoundaryMarkers(mesh, 0)
L.Faces(1, 0.0, 1.0).mark(markers, 1)
bcs = [L.DirichletBC(W.sub(c), L.Constant(0.0), markers, 1) for c in range(3)]
U, V = L.TrialFunctions(W), L.TestFunctions(W)
u, p, v, q = L.as_vector(U[:3]), U[3], L.as_vector(V[:3]), V[3]
a = L.inner(L.grad(u), L.grad(v)) * L.dx - L.div(v) * p * L.dx - q * L.div(u) * L.dx
Am = L.assemble(a, couplings="form", bcs=bcs)
A = L.as_backend_type(Am).mat()
null = L.constant_nullspace(Am, W.sub(3))
b = L.assemble(1.0 * v[0] * L.dx)
for bc in bcs:
    bc.apply(b)
null.orthogonalize(b)
lumped = L.assemble(q * L.dx)
p_dofs = L.owned_dofs(W, 3)
diag = 0.4 * lumped.get_local()[p_dofs - lumped.local_range()[0]]
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A, A)
ksp.setType("minres")
ksp.setTolerances(rtol=1e-12, max_it=5000)
pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitIS(*[(n, PETSc.IS().createGeneral(L.owned_dofs(W, c), comm=comm))
                     for c, n in enumerate(("u0", "u1", "u2", "p"))])
opts = PETSc.Options()
opts["pc_fieldsplit_type"] = "additive"
for c in range(3):
    opts["fieldsplit_u{}_ksp_type".format(c)] = "preonly"
    opts["fieldsplit_u{}_pc_type".format(c)] = "hypre"
opts["fieldsplit_p_ksp_type"] = "preonly"
opts["fieldsplit_p_pc_type"] = "none"
ksp.setFromOptions()
ksp.setUp()
Dm = diagonal_matrix(diag, comm)
pk = pc.getFieldSplitSubKSP()[3]
pk.setOperators(Dm, Dm)
pk.getPC().setType("jacobi")
pk.setUp()
w = L.Function(W)
ksp.solve(L.as_backend_type(b).vec(), L.as_backend_type(w.vector()).vec())
L.sync(w)
uw = L.as_vector([w[i] for i in range(3)])
mean = [L.assemble(w[i] * L.dx) for i in range(3)]          # the box's volume is 1
out = dict(
    reason=ksp.getConvergedReason(), iterations=ksp.getIterationNumber(), k=mean[0],
    cross=max(abs(mean[1]), abs(mean[2])),
    divergence=float(np.sqrt(L.assemble(L.div(uw) ** 2 * L.dx))),
    dim=int(L.global_dim(W)), pressure_dim=int(L.global_dim(W.sub(3))),
    cells=int(L.global_num_cells(mesh)), vertices=int(L.global_num_vertices(mesh)),
    markers=[int(c) for c in L.marker_counts(markers)],
    fixed=int(comm.allreduce(len(L.fixed_dofs(W, bcs)))))
if comm.rank == 0:
    print("RESULT " + json.dumps(out), flush=True)
for o in (ksp, Dm):
    o.destroy()
L.collect_garbage()
