"""Worker for tests/test_dolfinx_blocks.py: the C++ extension's assembly a block of
cells at a time against all cells in one block (matrices) and against DOLFINx's own
(vectors), for forms with coefficients over cells, exterior and interior facets, on
scalar and mixed spaces (vectors also on a blocked vector space). Writes the largest differences as JSON (rank 0).

    [mpirun -n k] python3 tests/dolfinx_blocks_worker.py out.json
"""
import json
import sys

import numpy as np
import ufl
import basix.ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

from feluga.dolfinx.assembly import _inserter, assemble_matrix, create_matrix
from feluga.dolfinx.unique import UniqueDofs

comm = MPI.COMM_WORLD
msh = mesh.create_unit_cube(comm, 4, 4, 4)
P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2)
P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
V = fem.functionspace(msh, P2)
Vv = fem.functionspace(msh, ("Lagrange", 1, (3,)))
W = fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
u_adv = fem.Function(Vv)
u_adv.interpolate(lambda x: np.stack([np.sin(3 * x[1]), x[0] * x[2], np.cos(x[0])]))
rho = fem.Function(V)
rho.interpolate(lambda x: 1 + x[0] ** 2 + x[1] * x[2])
phi = fem.Function(W)
phi.sub(0).interpolate(lambda x: np.tanh(x[0] - 0.4))
n = ufl.FacetNormal(msh)
ext = _inserter(comm)
res = {}


def forms(S, c):
    u, v = ufl.TrialFunction(S), ufl.TestFunction(S)
    a = (rho * ufl.inner(ufl.dot(ufl.grad(u), u_adv), v) * ufl.dx
         + c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + rho * ufl.inner(u, v) * ufl.ds
         + ufl.avg(rho) * ufl.inner(ufl.jump(u), ufl.jump(v)) * ufl.dS)
    w = ufl.as_vector([1.0] * ufl.shape(u)[0]) if ufl.shape(u) else 1.0
    L = (rho * ufl.inner(w, v) * ufl.dx + ufl.dot(u_adv, n) * ufl.inner(w, v) * ufl.ds
         + ufl.avg(rho) * ufl.inner(ufl.avg(v), w) * ufl.dS)
    return fem.form(a), fem.form(L)


worst_m, worst_v = 0.0, 0.0
for name, S, c in (("scalar", V, rho), ("vector", Vv, rho), ("mixed", W, phi[0] ** 2 + 1)):
    a, L = forms(S, c)
    blocked = S.dofmap.index_map_bs > 1     # matrices over unique dofs: not blocked
    U = None if blocked else UniqueDofs(S)
    mats = []
    for block in (() if blocked else (10 ** 9, 7)):
        ext.set_block(block)
        K = create_matrix(a, U)
        assemble_matrix(K, a)       # the first assembly, through the local maps
        assemble_matrix(K, a)       # then the extension's
        mats.append(K)
    if mats:
        D = mats[0].copy()
        D.axpy(-1.0, mats[1])
        worst_m = max(worst_m, D.norm(PETSc.NormType.INFINITY))
    ext.set_block(7)
    ours = petsc.create_vector(S)
    with ours.localForm() as loc:
        loc.set(0.0)
        assert ext.assemble_vector(L._cpp_object, loc.getArray())
    theirs = petsc.create_vector(S)
    with theirs.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(theirs, L)
    with ours.localForm() as a1, theirs.localForm() as a2:
        worst_v = max(worst_v, float(np.abs(a1.getArray() - a2.getArray()).max(initial=0)))
ext.set_block(8192)
res["matrix_blocks_vs_whole"] = comm.allreduce(worst_m, op=MPI.MAX)
res["vector_blocks_vs_dolfinx"] = comm.allreduce(worst_v, op=MPI.MAX)
if comm.rank == 0:
    with open(sys.argv[1], "w") as fh:
        json.dump(res, fh)
