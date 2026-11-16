"""tests/test_read_mesh_file.py's worker: backend argv[1] reads the meshers' XDMF mesh
argv[2] (read_mesh_file, periodic); prints, on rank 0, the global cell and vertex counts, the
volume, and whether every cell's global id is its row (its vertices' rows match the file's)."""
import importlib
import json
import sys

L = importlib.import_module("femtools.backend." + sys.argv[1])  # before mpi4py (dolfin)
import h5py  # noqa: E402
import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

comm = MPI.COMM_WORLD
path = sys.argv[2]
mesh = L.read_mesh_file(path, periodic=(True, True, True), tol=1e-10)
with h5py.File(path[:-5] + ".h5", "r") as h5:
    x_file, cells_file = h5["data0"][:], h5["data1"][:]
if sys.argv[1] == "dolfin":
    tdim = mesh.topology().dim()
    n = mesh.topology().ghost_offset(tdim)
    ids = np.asarray(mesh.topology().global_indices(tdim))[:n]
    x = mesh.coordinates()[mesh.cells()[:n]]
    counts = (mesh.num_entities_global(tdim), mesh.num_entities_global(0))
else:
    tdim = mesh.topology.dim
    n = mesh.topology.index_map(tdim).size_local
    ids = np.asarray(mesh.topology.original_cell_index)[:n]
    x = mesh.geometry.x[mesh.geometry.dofmap[:n]]
    counts = (mesh.topology.index_map(tdim).size_global,
              mesh.topology.index_map(0).size_global)
# each cell's vertices, as coordinates, are its file row's (in any order)
ok = all(np.allclose(np.sort(x[k], axis=0), np.sort(x_file[cells_file[i]], axis=0))
         for k, i in enumerate(ids))
volume = L.assemble(L.Constant(1.0, mesh) * L.dx(domain=mesh)) if sys.argv[1] == "dolfinx" \
    else L.assemble(L.Constant(1.0) * L.dx(domain=mesh))
out = dict(cells=int(counts[0]), vertices=int(counts[1]), volume=volume,
           ids_are_rows=bool(comm.allreduce(int(not ok)) == 0),
           every_cell_once=bool(np.array_equal(np.sort(np.concatenate(comm.allgather(ids))),
                                               np.arange(len(cells_file)))))
if comm.rank == 0:
    print("RESULT " + json.dumps(out), flush=True)
