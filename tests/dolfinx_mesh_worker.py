"""Worker for tests/test_dolfinx_mesh.py: meshes dolfin writes, read on DOLFINx.

    [mpirun -n k] python3 tests/dolfinx_mesh_worker.py write-dolfin DIR
    [mpirun -n k] python3 tests/dolfinx_mesh_worker.py read-dolfinx DIR out.json

dolfin writes a rectangle and a box (femtools.dolfin_layout.write_mesh; on several
ranks, so cell_indices is not the identity) and a P2 field on each. DOLFINx reads the
mesh (femtools.dolfinx.mesh.read_mesh) and checks every cell against the file -- its
original index names the file's cell with the same vertices, at the same coordinates --
and reads the field (femtools.dolfinx.checkpoint) against its exact values.
"""
import json
import os
import sys

import numpy as np

CASES = {"rect": 2, "box": 3}
CPP = {2: "sin(2*pi*x[0])*x[1] + x[0]*x[0]", 3: "sin(2*pi*x[0])*x[1] + x[0]*x[2] + x[2]*x[2]"}


def exact(X, gdim):
    v = np.sin(2 * np.pi * X[:, 0]) * X[:, 1] + X[:, 0] * (X[:, 0] if gdim == 2 else X[:, 2])
    return v + (X[:, 2] ** 2 if gdim == 3 else 0.0)


def write_dolfin(folder):
    import dolfin as df
    from femtools.dolfin_layout import write_function, write_mesh
    meshes = {"rect": df.RectangleMesh(df.Point(0, 0), df.Point(1, 2), 5, 7, "crossed"),
              "box": df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 2), 3, 3, 5)}
    for name, mesh in meshes.items():
        path = os.path.join(folder, name + ".h5")
        write_mesh(path, mesh, "mesh", mode="w")
        V = df.FunctionSpace(mesh, "CG", 2)
        u = df.interpolate(df.Expression(CPP[CASES[name]], degree=4), V)
        write_function(path, u, "u", mode="a")


def read_dolfinx(folder, out):
    import h5py
    from dolfinx import fem
    from mpi4py import MPI
    from femtools.dolfinx.checkpoint import read_function
    from femtools.dolfinx.mesh import read_mesh
    from femtools.dolfinx.unique import UniqueDofs
    comm = MPI.COMM_WORLD
    res = {}
    for name, gdim in CASES.items():
        path = os.path.join(folder, name + ".h5")
        mesh = read_mesh(path, comm)
        with h5py.File(path, "r") as h5:
            topo = np.asarray(h5["mesh/topology"])
            ids = np.asarray(h5["mesh/cell_indices"])
            coords = np.asarray(h5["mesh/coordinates"])
        by_id = np.empty_like(topo)
        by_id[ids] = topo
        tdim = mesh.topology.dim
        n = mesh.topology.index_map(tdim).size_local
        G = mesh.geometry.dofmaps[0][:n]
        gv = np.asarray(mesh.geometry.input_global_indices)
        oc = np.asarray(mesh.topology.original_cell_index[:n])
        same_cells = np.array_equal(np.sort(gv[G], axis=1), np.sort(by_id[oc], axis=1))
        same_x = np.allclose(mesh.geometry.x[:, :gdim], coords[gv], atol=1e-14)
        res[name + "_cells"] = bool(comm.allreduce(int(not (same_cells and same_x))) == 0)
        res[name + "_identity_ids"] = bool(np.array_equal(ids, np.arange(len(ids))))
        V = fem.functionspace(mesh, ("Lagrange", 2))
        U = UniqueDofs(V)
        got = read_function(path, "u", V, U)
        X = V.tabulate_dof_coordinates()[U.owned_unique]
        d = np.abs(got - exact(X, gdim))
        res[name + "_field"] = float(comm.allreduce(d.max() if d.size else 0.0, op=MPI.MAX))
        res[name + "_cells_global"] = int(mesh.topology.index_map(tdim).size_global)
    if comm.rank == 0:
        with open(out, "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    mode, folder = sys.argv[1:3]
    {"write-dolfin": lambda: write_dolfin(folder),
     "read-dolfinx": lambda: read_dolfinx(folder, sys.argv[3])}[mode]()
