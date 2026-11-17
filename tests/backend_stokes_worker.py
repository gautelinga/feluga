"""tests/test_backend_stokes.py's worker: one periodic Taylor-Hood Stokes solve written once
against femtools.backend, on backend argv[1]. Plane Poiseuille: the unit cube, periodic in x
and z, no-slip on y = 0 and y = 1, driven along x by a unit body force; P2 holds the parabola
exactly, so k = mean(u_x) / f = 1/12 to the solver's tolerance. MINRES with
femtools.saddle's block preconditioner, kind argv[2] ("fieldsplit" or "shared"), on the
Taylor-Hood layout argv[3] ("vector": [P2^3, P1], one no-slip condition on the velocity;
"scalars": [P2, P2, P2, P1], one a component). Also the global counts (global_dim,
global_num_cells, global_num_vertices, marker_counts, fixed_dofs), and the solution split
as solve-stokes splits it (`subfunction`: the velocity, a vector on "vector"): with argv[4]
"write", u and p written to argv[5] in dolfin's layout; with "read", that file's velocity
(another backend's, say) compared with this one's; with "resetup", the KSP under an
options prefix, and A changed and solved again. Prints one JSON line on rank 0."""
import importlib
import json
import sys

L = importlib.import_module("femtools.backend." + sys.argv[1])  # before mpi4py (dolfin)
import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

from femtools.couplings import num_sub_spaces  # noqa: E402
from femtools.saddle import block_preconditioner, pressure_mass_diagonal, solve_checked  # noqa: E402,E501


N = 4
comm = MPI.COMM_WORLD
box = L.PeriodicBox({0: (0.0, 1.0), 2: (0.0, 1.0)})
mesh = L.repair_partition(L.BoxMesh(L.Point(0, 0, 0), L.Point(1, 1, 1), N, N, N), box)
cell = mesh.ufl_cell()
LAYOUT = sys.argv[3] if len(sys.argv) > 3 else "scalars"
P1 = L.FiniteElement("CG", cell, 1)
if LAYOUT == "vector":
    W = L.FunctionSpace(mesh, L.MixedElement([L.VectorElement("CG", cell, 2), P1]),
                        constrained_domain=box)
else:
    W = L.FunctionSpace(mesh, L.MixedElement([L.FiniteElement("CG", cell, 2)] * 3 + [P1]),
                        constrained_domain=box)
PRESSURE = W.sub(num_sub_spaces(W) - 1)
L.check_periodic_pairing(box, [W])
markers = L.BoundaryMarkers(mesh, 0)
L.Faces(1, 0.0, 1.0).mark(markers, 1)
if LAYOUT == "vector":
    bcs = [L.DirichletBC(W.sub(0), L.Constant((0.0, 0.0, 0.0)), markers, 1)]
    U, V = L.TrialFunctions(W), L.TestFunctions(W)
    (u, p), (v, q) = U, V
else:
    bcs = [L.DirichletBC(W.sub(c), L.Constant(0.0), markers, 1) for c in range(3)]
    U, V = L.TrialFunctions(W), L.TestFunctions(W)
    u, p, v, q = L.as_vector(U[:3]), U[3], L.as_vector(V[:3]), V[3]
a = L.inner(L.grad(u), L.grad(v)) * L.dx - L.div(v) * p * L.dx - q * L.div(u) * L.dx
Am = L.assemble(a, couplings="form", bcs=bcs)
A = L.as_backend_type(Am).mat()
null = L.constant_nullspace(Am, PRESSURE)
b = L.assemble(1.0 * v[0] * L.dx)
for bc in bcs:
    bc.apply(b)
null.orthogonalize(b)
KIND = sys.argv[2] if len(sys.argv) > 2 else "fieldsplit"
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A, A)
ksp.setType("minres")
ksp.setTolerances(rtol=1e-12, max_it=5000)
RESETUP = len(sys.argv) > 4 and sys.argv[4] == "resetup"
if RESETUP:
    ksp.setOptionsPrefix("stokes_")
pre = block_preconditioner(ksp, A, W, pressure_mass_diagonal(W), {"pc_hypre_type": "boomeramg"},
                           kind=KIND)
w = L.Function(W)
iterations = solve_checked(ksp, L.as_backend_type(b).vec(), L.as_backend_type(w.vector()).vec())
L.sync(w)
uw = L.as_vector([w[i] for i in range(3)])
mean = [L.assemble(w[i] * L.dx) for i in range(3)]          # the box's volume is 1
iu = 0
ip = num_sub_spaces(W) - 1
u_ = L.subfunction(w, iu)
p_ = L.subfunction(w, ip)
split_k = L.assemble((u_[0] if LAYOUT == "vector" else u_) * L.dx)
other = resetup = None
if RESETUP:
    # A changed (its first velocity diagonal set again: the same values) and solved again:
    # the preconditioner set up anew, as it was
    pc = ksp.getPC()
    subs = pc.getFieldSplitSubKSP() if pc.getType() == "fieldsplit" else []
    r0 = int(L.owned_dofs(W, (0, 0) if LAYOUT == "vector" else 0)[0])
    A.setValues([r0], [r0], [A.getValue(r0, r0)])
    A.assemble()
    w2 = L.Function(W)
    it2 = solve_checked(ksp, L.as_backend_type(b).vec(), L.as_backend_type(w2.vector()).vec())
    L.sync(w2)
    resetup = dict(prefix=ksp.getOptionsPrefix(), iterations=it2,
                   k=L.assemble(w2[0] * L.dx),
                   splits=[(k.getOptionsPrefix(), k.getPC().getType()) for k in subs])
if len(sys.argv) > 5 and sys.argv[4] == "write":
    L.write_functions_h5(sys.argv[5], [(u_, "u"), (p_, "p")])
elif len(sys.argv) > 5 and sys.argv[4] == "read":
    theirs = L.Function(L.subspace(W, iu))
    L.read_function_h5(sys.argv[5], theirs, "u")
    d = theirs.vector().get_local() - u_.vector().get_local()
    scale = comm.allreduce(float(np.abs(u_.vector().get_local()).max(initial=0)), op=MPI.MAX)
    other = comm.allreduce(float(np.abs(d).max(initial=0)), op=MPI.MAX) / scale
out = dict(split_k=split_k, other=other, resetup=resetup,
    reason=ksp.getConvergedReason(), iterations=iterations, k=mean[0],
    cross=max(abs(mean[1]), abs(mean[2])),
    divergence=float(np.sqrt(L.assemble(L.div(uw) ** 2 * L.dx))),
    dim=int(L.global_dim(W)), pressure_dim=int(L.global_dim(PRESSURE)),
    uy_dim=int(L.global_dim(W.sub(0).sub(1) if LAYOUT == "vector" else W.sub(1))),
    cells=int(L.global_num_cells(mesh)), vertices=int(L.global_num_vertices(mesh)),
    markers=[int(c) for c in L.marker_counts(markers)],
    fixed=int(comm.allreduce(len(L.fixed_dofs(W, bcs)))))
if comm.rank == 0:
    print("RESULT " + json.dumps(out), flush=True)
pre.destroy()
ksp.destroy()
L.collect_garbage()
