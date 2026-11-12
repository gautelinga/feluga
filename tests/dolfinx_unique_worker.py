"""Worker for tests/test_dolfinx_unique.py: assembly over unique dofs checked against
plain DOLFINx, against PᵀAP with an explicit P (preallocated exactly, also where forms
share or differ in their pattern), and a periodic solve with Dirichlet data through the
backend's DirichletBC against dolfinx_mpc. Writes the differences as JSON (rank 0).

    [mpirun -n k] python3 tests/dolfinx_unique_worker.py out.json
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

from femtools.dolfinx.assembly import (assemble_matrix, assemble_vector, constant_nullspace,
                                       create_matrix)
from femtools.dolfinx.periodic import periodic_box
from femtools.dolfinx.unique import UniqueDofs

comm = MPI.COMM_WORLD
LX, LY = 1.0, 2.0
msh = mesh.create_rectangle(comm, [[0, 0], [LX, LY]], [12, 24])
tdim = msh.topology.dim
msh.topology.create_connectivity(tdim - 1, tdim)
x = ufl.SpatialCoordinate(msh)
res = {}


def rel(A, B):
    """Relative Frobenius / 2-norm difference of two PETSc matrices or vectors."""
    D = A.copy()
    if isinstance(A, PETSc.Mat):
        D.axpy(-1.0, B, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        return D.norm(PETSc.NormType.FROBENIUS) / max(B.norm(PETSc.NormType.FROBENIUS), 1e-300)
    D.axpy(-1.0, B)
    return D.norm() / max(B.norm(), 1e-300)


def explicit_P(U):
    """P: plain global dofs -> compact dofs, one 1 per row (owned plain dofs)."""
    imap = U.V.dofmap.index_map
    bs = U.bs
    n_own = imap.size_local * bs
    start = imap.local_range[0] * bs
    P = PETSc.Mat().createAIJ(((n_own, imap.size_global * bs), (U.n_owned, U.n_global)),
                              nnz=(1, 1), comm=comm)
    for i in range(n_own):
        P.setValue(start + i, U.compact[i], 1.0)
    P.assemble()
    return P


def plain_matrix(a):
    A = petsc.assemble_matrix(a)
    A.assemble()
    return A


def plain_vector(L):
    b = petsc.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return b


def exact(K):
    """Whether K's preallocation was exact: no entry added beyond it, none left unused.
    (DOLFINx's assembly drops an entry beyond it silently, which only a comparison of
    the values shows.)"""
    info = K.getInfo()
    return bool(comm.allreduce(int(info["mallocs"]) + int(info["nz_unneeded"])) == 0)


V = fem.functionspace(msh, ("Lagrange", 2))
Q = fem.functionspace(msh, ("Lagrange", 1))
P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
W = fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p = ufl.TrialFunction(Q)
kappa = 1.0 + 0.5 * ufl.sin(2 * ufl.pi * x[0]) ** 2
beta = ufl.as_vector((ufl.cos(2 * ufl.pi * x[0]), 0.3))
a_V = fem.form(kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
               + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx + u * v * ufl.dx)
a_VQ = fem.form(p * v.dx(0) * ufl.dx)  # rectangular: test V, trial Q
phi, g = ufl.TrialFunctions(W)
xi, eta = ufl.TestFunctions(W)
a_W = fem.form((phi * xi / 0.01 + ufl.dot(beta, ufl.grad(phi)) * xi
                + 1e-3 * ufl.inner(ufl.grad(g), ufl.grad(xi)) + g * eta
                - 0.02 * ufl.inner(ufl.grad(phi), ufl.grad(eta)) - 2 * phi * eta) * ufl.dx)
f = ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]) + 1.0
L_V = fem.form(f * v * ufl.dx)

walls = mesh.locate_entities_boundary(msh, tdim - 1,
                                      lambda X: np.isclose(X[1], 0) | np.isclose(X[1], LY))
gw = fem.Function(V)
gw.interpolate(lambda X: np.sin(2 * np.pi * X[0]) * (1 + X[1]) + 2.0)  # periodic, nonzero
bc = fem.dirichletbc(gw, fem.locate_dofs_topological(V, tdim - 1, walls))

# 1. without periodicity: identical to plain DOLFINx
U0 = UniqueDofs(V)
K = create_matrix(a_V, U0)
assemble_matrix(K, a_V)
res["plain_matrix"] = rel(K, plain_matrix(a_V))
b = U0.create_vector()
assemble_vector(b, L_V, U0)
res["plain_rhs"] = rel(b, plain_vector(L_V))

# 2. periodic in x: folded = PᵀAP with an explicit P (square, rectangular, mixed)
UV = UniqueDofs(V, *periodic_box(V, periodic=(True, False)))
UQ = UniqueDofs(Q, *periodic_box(Q, periodic=(True, False)))
UW = UniqueDofs(W, *periodic_box(W, periodic=(True, False)))
PV, PQ, PW = explicit_P(UV), explicit_P(UQ), explicit_P(UW)
for name, a, R, C, Pr, Pc in (("fold_square", a_V, UV, UV, PV, PV),
                              ("fold_rect", a_VQ, UV, UQ, PV, PQ),
                              ("fold_mixed", a_W, UW, UW, PW, PW)):
    K = create_matrix(a, R, C)
    assemble_matrix(K, a)
    res[name + "_exact"] = exact(K)
    A = plain_matrix(a)
    ref = Pr.transposeMatMult(A.matMult(Pc))
    res[name] = rel(K, ref)
bp = plain_vector(L_V)
ref = UV.create_vector()
PV.multTranspose(bp, ref)
b = UV.create_vector()
assemble_vector(b, L_V, UV)
res["fold_rhs"] = rel(b, ref)
res["unique_dofs_V"] = UV.n_global
res["slaves_V"] = V.dofmap.index_map.size_global - UV.n_global

# 2b. the preallocation shared by forms over the same entities, and not by forms over
# others: interior facets (wider), a subdomain of the cells (narrower)
cells = np.arange(msh.topology.index_map(tdim).size_local, dtype=np.int32)
mid = mesh.compute_midpoints(msh, tdim, cells)
half = mesh.meshtags(msh, tdim, cells, np.where(mid[:, 1] < LY / 2, 1, 2).astype(np.int32))
dx_half = ufl.Measure("dx", domain=msh, subdomain_data=half)
jumps = ufl.inner(ufl.jump(ufl.grad(u)), ufl.jump(ufl.grad(v))) * ufl.dS
for name, form in (("same_cells", 2.0 * u * v * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v))
                    * ufl.dx),
                   ("interior_facets", u * v * ufl.dx + jumps),
                   ("cell_subdomain", u * v * dx_half(1))):
    a = fem.form(form)
    K = create_matrix(a, UV)
    assemble_matrix(K, a)
    res["share_" + name + "_exact"] = exact(K)
    res["share_" + name] = rel(K, PV.transposeMatMult(plain_matrix(a).matMult(PV)))

# 3. periodic solve with nonzero wall data (corners on both walls and periodic faces),
# the conditions applied by the backend's DirichletBC
from femtools.backend import dolfinx as B  # noqa: E402

Vb = B.FunctionSpace(msh, "CG", 2, constrained_domain=B.PeriodicBox({0: (0, LX)}))
Ub = B.unique_dofs(Vb)
markers = B.BoundaryMarkers(msh)
markers.mark(lambda X: np.isclose(X[1], 0) | np.isclose(X[1], LY), 2)
gb = B.Function(Vb)
gb.interpolate(lambda X: np.sin(2 * np.pi * X[0]) * (1 + X[1]) + 2.0)
bcb = B.DirichletBC(Vb, gb, markers, 2)
ub, vb = ufl.TrialFunction(Vb), ufl.TestFunction(Vb)
Kb = B.assemble(kappa * ufl.inner(ufl.grad(ub), ufl.grad(vb)) * ufl.dx
                + ufl.dot(beta, ufl.grad(ub)) * vb * ufl.dx + ub * vb * ufl.dx)
bb = B.assemble(f * vb * ufl.dx)
bcb.apply(Kb, bb)
ksp = PETSc.KSP().create(comm)
ksp.setOperators(Kb.mat())
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
u_red = B.Function(Vb)
ksp.solve(bb.vec(), u_red.vector().vec())
u_red.sync()
n_own = Vb.dofmap.index_map.size_local
res["solution_norm"] = float(np.sqrt(comm.allreduce(float(np.sum(u_red.x.array[:n_own] ** 2)))))
# periodic: every slave equals its master
back = Ub.restrict(u_red, Ub.create_vector())
res["roundtrip"] = rel(back, u_red.vector().vec())
try:
    import dolfinx_mpc

    right = mesh.locate_entities_boundary(msh, tdim - 1, lambda X: np.isclose(X[0], LX))
    order = np.argsort(right)
    mt = mesh.meshtags(msh, tdim - 1, right[order].astype(np.int32),
                       np.full(len(right), 3, dtype=np.int32))

    def relation(X):
        Y = X.copy()
        Y[0] = X[0] - LX
        return Y

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_topological(V, mt, 3, relation, [bc])
    mpc.finalize()
    A_m = dolfinx_mpc.assemble_matrix(a_V, mpc, bcs=[bc])
    b_m = dolfinx_mpc.assemble_vector(L_V, mpc)
    dolfinx_mpc.apply_lifting(b_m, [a_V], [[bc]], mpc)
    b_m.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b_m, [bc])
    ksp_m = PETSc.KSP().create(comm)
    ksp_m.setOperators(A_m)
    ksp_m.setType("preonly")
    ksp_m.getPC().setType("lu")
    ksp_m.getPC().setFactorSolverType("mumps")
    x_m = A_m.createVecRight()
    ksp_m.solve(b_m, x_m)
    u_m = fem.Function(mpc.function_space)
    u_m.x.array[:n_own] = x_m.getArray(readonly=True)[:n_own]
    u_m.x.scatter_forward()
    mpc.backsubstitution(u_m)
    on_b = fem.Function(Vb)
    fem.Function.interpolate(on_b, u_m)
    d = u_red.x.array[:n_own] - on_b.x.array[:n_own]
    res["vs_dolfinx_mpc"] = float(np.sqrt(comm.allreduce(float(np.sum(d ** 2))))
                                  / np.sqrt(comm.allreduce(float(np.sum(on_b.x.array[:n_own] ** 2)))))
except ImportError:
    res["vs_dolfinx_mpc"] = None

# 4. constants are in the kernel of the periodic pure-Neumann Laplacian (no slave rows)
UQn = UniqueDofs(Q, *periodic_box(Q, periodic=(True, True)))
q = ufl.TestFunction(Q)
a_lap = fem.form(ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx)
K = create_matrix(a_lap, UQn)
assemble_matrix(K, a_lap)
ns = constant_nullspace(UQn)
res["nullspace"] = bool(ns.test(K))

# 5. interleaved numbering of a mixed space: the components at a node one after the
# other, the same operator up to the permutation, values through restrict and prolong
from femtools.dolfinx.periodic import flat_collapse_map  # noqa: E402

UWi = UniqueDofs(W, *periodic_box(W, periodic=(True, False)), interleave=True)
K = create_matrix(a_W, UWi)
assemble_matrix(K, a_W)
res["fold_mixed_interleaved_exact"] = exact(K)
res["fold_mixed_interleaved"] = rel(K, explicit_P(UWi).transposeMatMult(
    plain_matrix(a_W).matMult(explicit_P(UWi))))
n_loc = W.dofmap.index_map.size_local + W.dofmap.index_map.num_ghosts
comp, X = np.zeros(n_loc, dtype=int), np.zeros((n_loc, 3))
for i in range(2):
    Wi, parent = W.sub(i).collapse()
    parent = flat_collapse_map(parent)
    comp[parent] = i
    X[parent] = Wi.tabulate_dof_coordinates()
own = UWi.owned_unique
pairs_ok = (len(own) % 2 == 0 and np.all(comp[own] == np.arange(len(own)) % 2)
            and np.allclose(X[own[0::2]], X[own[1::2]]))
res["interleaved"] = bool(comm.allreduce(int(not pairs_ok)) == 0)
w = fem.Function(W)
w.sub(0).interpolate(lambda X: np.sin(2 * np.pi * X[0]) + X[1])
w.sub(1).interpolate(lambda X: np.cos(2 * np.pi * X[0]) * X[1])
c = UWi.restrict(w, UWi.create_vector())
c2 = UWi.restrict(UWi.prolong(c, fem.Function(W)), UWi.create_vector())
res["interleaved_roundtrip"] = rel(c2, c)

if comm.rank == 0:
    with open(sys.argv[1], "w") as fh:
        json.dump(res, fh)
