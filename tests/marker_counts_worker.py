"""tests/test_marker_counts.py's worker: marker_counts on backend argv[1] with one exterior
facet on rank 0 alone carrying a negative marker (DOLFINx: -1; dolfin's size_t: 2**64 - 1,
a wrapped -1). Each rank prints "RAISED <message>" or "COUNTS <counts>"."""
import importlib
import sys

L = importlib.import_module("feluga.backend." + sys.argv[1])  # before mpi4py (dolfin)
import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

comm = MPI.COMM_WORLD
mesh = L.BoxMesh(L.Point(0, 0, 0), L.Point(1, 1, 1), 3, 3, 3)
markers = L.BoundaryMarkers(mesh, 1)
if sys.argv[1] == "dolfinx":
    first = np.asarray(markers.boundary)[0]
else:
    import dolfin
    exterior = dolfin.MeshFunction("size_t", mesh, 2, 0)
    dolfin.CompiledSubDomain("on_boundary").mark(exterior, 1)   # compiled on every rank
    first = np.flatnonzero(exterior.array() == 1)[0]
if comm.rank == 0:
    markers.array()[first] = -1 if sys.argv[1] == "dolfinx" else np.uint64(2 ** 64 - 1)
try:
    print("COUNTS", L.marker_counts(markers).tolist(), flush=True)
except ValueError as e:
    print("RAISED", e, flush=True)
