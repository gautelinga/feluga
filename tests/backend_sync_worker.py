"""tests/test_backend_sync.py's worker, on backend argv[1]: a Function written through PETSc
(as a KSP solve writes it: owned entries only), then `sync(f)`, assembles as written (and
the backend's assemble, which syncs its coefficients, does in any case).
Prints one JSON line on rank 0."""
import importlib
import json
import sys

L = importlib.import_module("femtools.backend." + sys.argv[1])  # before mpi4py (dolfin)
from mpi4py import MPI  # noqa: E402

comm = MPI.COMM_WORLD
mesh = L.UnitSquareMesh(8, 8)
V = L.FunctionSpace(mesh, "CG", 1)
f = L.Function(V)
v = L.as_backend_type(f.vector()).vec()
with v as a:               # the owned entries alone, as a KSP solve writes them (VecSet would
    a[:] = 1.0             # set a ghosted vector's ghosts too)
# dolfin's own assemble (the backend's syncs its coefficients itself): the ghosts as written
stale = (__import__("dolfin").assemble(f * L.dx) if sys.argv[1] == "dolfin" else None)
L.sync(f)
out = dict(after_sync=L.assemble(f * L.dx), stale=stale)
if sys.argv[1] == "dolfin":                  # and dolfin's own assemble after the sync
    out["raw_after_sync"] = __import__("dolfin").assemble(f * L.dx)
if comm.rank == 0:
    print("RESULT " + json.dumps(out), flush=True)
L.collect_garbage()
