"""tests/test_couplings.py's worker: on backend argv[1], the Stokes operator on a periodic
Taylor-Hood space [P2, P2, P2, P1] assembled with couplings against without; one JSON line
a rank (RESULT ...)."""
import importlib
import json
import sys

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

L = importlib.import_module("femtools.backend." + sys.argv[1])
comm = MPI.COMM_WORLD
mesh = L.BoxMesh(L.Point(0, 0, 0), L.Point(1, 1, 1), 3, 3, 3)
cell = mesh.ufl_cell()
W = L.FunctionSpace(mesh, L.MixedElement([L.FiniteElement("CG", cell, 2)] * 3
                                         + [L.FiniteElement("CG", cell, 1)]),
                    constrained_domain=L.PeriodicBox({0: (0., 1.), 1: (0., 1.)}))
U, V = L.TrialFunctions(W), L.TestFunctions(W)
u, p, v, q = L.as_vector(U[:3]), U[3], L.as_vector(V[:3]), V[3]
laplace = L.inner(L.grad(u), L.grad(v)) * L.dx
a = laplace - L.div(v) * p * L.dx - q * L.div(u) * L.dx
STOKES = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0]], bool)


def info(M):
    return M.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)["nz_used"]


def diagonal_stored(M):
    lo, hi = M.getOwnershipRange()
    return all(i in M.getRow(i)[0] for i in range(lo, hi))


out = dict(rank=comm.rank)
out["mask"] = L.form_couplings(a).astype(int).tolist()
out["mask_laplace"] = L.form_couplings(laplace).astype(int).tolist()
A = L.assemble(a, couplings="form")
L.assemble(a, tensor=A)                       # again, into the pattern
F = L.assemble(a)
Am, Fm = L.as_backend_type(A).mat(), L.as_backend_type(F).mat()
D = Fm.copy()
D.axpy(-1.0, Am, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
out.update(diff=D.norm(PETSc.NormType.INFINITY), scale=Fm.norm(PETSc.NormType.INFINITY),
           nnz=info(Am), nnz_full=info(Fm), diagonal=bool(comm.allreduce(diagonal_stored(Am),
                                                                          op=MPI.LAND)))
# by component, never by value: the coupled pressure blocks all zero still stored
zero = L.constant_on(0.0, mesh)
Z = L.assemble(laplace + zero * (L.div(v) * p + q * L.div(u)) * L.dx, couplings=STOKES)
out["nnz_zero_coupling"] = info(L.as_backend_type(Z).mat())
# no-slip on the plates z = 0, 1: their dofs' rows and columns hold the diagonal alone
walls = "on_boundary && (near(x[2], 0.0) || near(x[2], 1.0))"
bcs = [L.DirichletBC(W.sub(c), 0.0, walls) for c in range(3)]
B = L.assemble(a, couplings="form", bcs=bcs)
L.assemble(a, tensor=B, bcs=bcs) if sys.argv[1] == "dolfin" else L.assemble(a, tensor=B)
Bm = L.as_backend_type(B).mat()
fixed = np.concatenate([np.asarray(L.fixed_dofs(W, [bc]), dtype=PETSc.IntType) for bc in bcs])
fixed = np.unique(fixed).astype(PETSc.IntType)
R = Fm.copy()
R.zeroRowsColumns(fixed, diag=1.0)
d = Bm.getDiagonal()                # dolfin's assemble_system: the cells holding the dof
R.setDiagonal(d)
lo = d.getOwnershipRange()[0]
counts = d.getArray()[fixed - lo]
out.update(bc_diag_whole=bool(comm.allreduce(bool(np.all((counts >= 1) & (counts == np.rint(counts)))),
                                              op=MPI.LAND)),
           bc_diag_sum=float(comm.allreduce(counts.sum())))
E = R.copy()
E.axpy(-1.0, Bm, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
T = Bm.copy()
T.axpy(-1.0, Bm.transpose(), structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
out.update(bc_diff=E.norm(PETSc.NormType.INFINITY), bc_nnz=info(Bm),
           bc_asymmetry=T.norm(PETSc.NormType.INFINITY),
           n_fixed=comm.allreduce(len(np.unique(fixed))))
bad = STOKES.copy()
bad[0, 3] = False
try:
    L.assemble(a, couplings=bad)
    out["refused"] = False
except RuntimeError:
    out["refused"] = True
print("RESULT " + json.dumps(out), flush=True)
