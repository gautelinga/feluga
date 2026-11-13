"""Worker for tests/test_garbage.py: PETSc's live allocation (rank 0, run with
PETSC_OPTIONS=-malloc_debug) as matrices and vectors are made and dropped through the
DOLFINx backend. Writes the growths in MB as JSON.

    [mpirun -n k] python3 tests/garbage_worker.py out.json
"""
import ctypes
import gc
import json
import sys

import ufl
from petsc4py import PETSc

import femtools.backend.dolfinx as B

# the PETSc library petsc4py runs on (dlsym searches the module's dependencies)
_lib = ctypes.CDLL(PETSc.__file__)


def live():
    d = ctypes.c_double()
    _lib.PetscMallocGetCurrentUsage(ctypes.byref(d))
    return d.value / 2 ** 20


mesh = B.UnitSquareMesh(64, 64)
V = B.FunctionSpace(mesh, "CG", 2)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
B.assemble(form)
B.collect_garbage()
gc.collect()
res = {}

# matrices dropped: the next create_matrix destroys them
base = live()
one = B.assemble(form)
size = live() - base
del one
for _ in range(5):
    B.assemble(form)
    gc.collect()
res["matrix_mb"] = size
res["five_dropped_matrices_mb"] = live() - base

# vectors dropped: kept until collect_garbage
base = live()
for _ in range(5):
    B.Vector(B.unique_dofs(V).create_vector())
gc.collect()
res["vectors_mb"] = 5 * B.unique_dofs(V).n_owned * 8 / 2 ** 20
res["dropped_vectors_kept_mb"] = live() - base
B.collect_garbage()
res["after_collect_garbage_mb"] = live() - base

if mesh.comm.rank == 0:
    with open(sys.argv[1], "w") as fh:
        json.dump(res, fh)
