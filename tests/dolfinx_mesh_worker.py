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


def write_dolfinx(folder):
    """A rectangle's mesh and its walls (y = 0 marked 2, y = 2 marked 3), written by
    DOLFINx in dolfin's layouts."""
    import dolfinx.mesh as dmesh
    from mpi4py import MPI
    from femtools.dolfinx.mesh import write_boundary_facets, write_mesh
    mesh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [1, 2]], [5, 7],
                                  dmesh.CellType.triangle, diagonal=dmesh.DiagonalType.crossed)
    write_mesh(os.path.join(folder, "x_mesh.h5"), mesh)
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim - 1)
    imap = mesh.topology.index_map(tdim - 1)
    values = np.zeros(imap.size_local + imap.num_ghosts, dtype=np.int64)
    for y, m in ((0.0, 2), (2.0, 3)):
        values[dmesh.locate_entities_boundary(mesh, tdim - 1,
                                              lambda X, y=y: np.isclose(X[1], y))] = m
    write_boundary_facets(os.path.join(folder, "x_walls.xdmf"), mesh, values)


def check_dolfin(folder, out):
    """dolfin reads DOLFINx's mesh (the same cells and vertices as its own rectangle),
    and writes the same walls with BoundaryMarkers: the facet files agree."""
    import dolfin as df
    from femtools import boundaries
    from femtools.boundary_markers import BoundaryMarkers
    ref = df.RectangleMesh(df.Point(0, 0), df.Point(1, 2), 5, 7, "crossed")
    m = df.Mesh()
    with df.HDF5File(df.MPI.comm_world, os.path.join(folder, "x_mesh.h5"), "r") as h:
        h.read(m, "mesh", False)
    res = {"mesh_cells": np.array_equal(np.sort(m.cells(), axis=1), np.sort(ref.cells(), axis=1)),
           "mesh_x": bool(np.allclose(m.coordinates(), ref.coordinates(), atol=1e-14))}
    b = BoundaryMarkers(ref)
    df.CompiledSubDomain("on_boundary && near(x[1], 0)").mark(b, 2)
    df.CompiledSubDomain("on_boundary && near(x[1], 2)").mark(b, 3)
    b.write(os.path.join(folder, "d_walls.xdmf"))

    def facet_set(path):
        points, facets, markers = boundaries.read(path)
        return sorted((tuple(sorted(map(tuple, np.round(points[f], 12)))), int(k))
                      for f, k in zip(facets, markers))
    res["walls"] = facet_set(os.path.join(folder, "x_walls.xdmf")) == \
        facet_set(os.path.join(folder, "d_walls.xdmf"))
    res["n_walls"] = len(facet_set(os.path.join(folder, "x_walls.xdmf")))
    with open(out, "w") as fh:
        json.dump({k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in res.items()}, fh)


if __name__ == "__main__":
    mode, folder = sys.argv[1:3]
    {"write-dolfin": lambda: write_dolfin(folder),
     "read-dolfinx": lambda: read_dolfinx(folder, sys.argv[3]),
     "write-dolfinx": lambda: write_dolfinx(folder),
     "check-dolfin": lambda: check_dolfin(folder, sys.argv[3])}[mode]()
