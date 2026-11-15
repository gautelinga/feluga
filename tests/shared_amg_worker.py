"""tests/test_shared_amg.py's worker: on backend argv[1], the shared velocity AMG on a
periodic [P2, P2, P2, P1] Stokes operator stored by component; one JSON line a rank."""
import importlib
import json
import sys

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from femtools.ksp import BOOMERAMG_CLASSICAL
from femtools.shared_amg import SharedVelocityPC, check_blocks, component_index_sets

L = importlib.import_module("femtools.backend." + sys.argv[1])
comm = MPI.COMM_WORLD
mesh = L.BoxMesh(L.Point(0, 0, 0), L.Point(1, 1, 1), 4, 4, 4)
cell = mesh.ufl_cell()
W = L.FunctionSpace(mesh, L.MixedElement([L.FiniteElement("CG", cell, 2)] * 3
                                         + [L.FiniteElement("CG", cell, 1)]),
                    constrained_domain=L.PeriodicBox({0: (0., 1.), 1: (0., 1.)}))
U, V = L.TrialFunctions(W), L.TestFunctions(W)
u, p, v, q = L.as_vector(U[:3]), U[3], L.as_vector(V[:3]), V[3]
mass = L.inner(u, v) * L.dx              # definite: the periodic Laplacian alone is not
a = L.inner(L.grad(u), L.grad(v)) * L.dx + mass - L.div(v) * p * L.dx - q * L.div(u) * L.dx
A = L.as_backend_type(L.assemble(a, couplings="form")).mat()
COMPONENTS = ((0,), (1,), (2,))
sets = component_index_sets(W, comm, COMPONENTS)
out = dict(rank=comm.rank, sizes=[len(s) for s in sets])
check_blocks(A, sets, comm)
out["check_passes"] = True
# component 1's Laplacian doubled: its block is not the first's
b = 2 * U[1].dx(0) * V[1].dx(0) * L.dx
B = L.as_backend_type(L.assemble(a + b, couplings="form")).mat()
try:
    check_blocks(B, sets, comm)
    out["changed_block_refused"] = False
except RuntimeError:
    out["changed_block_refused"] = True
# the preconditioner: the same values on components 0 and 1 (in node order) come back the
# same; the pressure is divided by diag_M
p_dofs = L.owned_dofs(W, 3)
diag_M = 1.0 + np.arange(len(p_dofs), dtype=float)
options = dict(pc_hypre_type="boomeramg", pc_hypre_boomeramg_strong_threshold=0.7,
               **BOOMERAMG_CLASSICAL)
shell = SharedVelocityPC(A, sets, p_dofs, diag_M, comm, options)
x, y = A.createVecs()
rng = np.random.default_rng(comm.rank)
x.setRandom()
xc = shell.xc.duplicate()
xc.setArray(rng.random(xc.getLocalSize()))
for c in (0, 1):
    shell.scatters[c].scatter(xc, x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.REVERSE)
shell.apply(None, x, y)
yc = [shell.xc.duplicate() for _ in range(2)]
for c in (0, 1):
    shell.scatters[c].scatter(y, yc[c], PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
yc[1].axpy(-1.0, yc[0])
out["same_input_same_output"] = yc[1].norm() <= 1e-14 * yc[0].norm() and yc[0].norm() > 0
xp, yp = shell.xp.duplicate(), shell.xp.duplicate()
shell.p_scatter.scatter(x, xp, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
shell.p_scatter.scatter(y, yp, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
out["pressure"] = bool(np.allclose(yp.getArray() * diag_M, xp.getArray(), rtol=1e-14, atol=0))
shell.destroy()
print("RESULT " + json.dumps(out), flush=True)
