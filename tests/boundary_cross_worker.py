"""Worker for tests/test_boundary_cross.py: boundary facet files written on one backend and
read on the other.

    [mpirun -n k] python3 tests/boundary_cross_worker.py write dolfin|dolfinx DIR
    [mpirun -n k] python3 tests/boundary_cross_worker.py read dolfin|dolfinx DIR

`write` writes `<backend>_<k>_<2|3>d.xdmf`, a rectangle's and a box's exterior facets
marked by position (`marker`); `read` reads every file the other backend wrote in DIR onto
the same mesh and prints `OK <file>` on rank zero when each exterior facet holds its
marker and every other facet 0.
"""
import glob
import os
import sys

import numpy as np

LX, LY = 1.0, 2.0
N2, N3 = (5, 7), (3, 4, 2)


def marker(mid):
    """The marker of an exterior facet with midpoint `mid` (rows): x = 0 is 3, y = 0 is
    5, x = LX is 7, the rest 0."""
    x, y = mid[:, 0], mid[:, 1]
    return np.where(np.isclose(x, 0.0), 3, np.where(
        np.isclose(y, 0.0), 5, np.where(np.isclose(x, LX), 7, 0))).astype(np.int64)


def dolfin_mesh(dim):
    import dolfin as df
    comm = df.MPI.comm_world
    if dim == 2:
        return df.RectangleMesh(comm, df.Point(0, 0), df.Point(LX, LY), *N2)
    return df.BoxMesh(comm, df.Point(0, 0, 0), df.Point(LX, LY, 1.0), *N3)


def dolfin_exterior(mesh):
    """The local facets' midpoints and whether each is exterior."""
    import dolfin as df
    D = mesh.topology().dim()
    on = df.MeshFunction("size_t", mesh, D - 1, 0)
    df.DomainBoundary().mark(on, 1)
    mid = np.array([f.midpoint().array() for f in df.facets(mesh)]).reshape(-1, 3)
    return mid, on.array() == 1


def dolfinx_mesh(dim):
    import dolfinx.mesh as dm
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if dim == 2:
        return dm.create_rectangle(comm, [[0, 0], [LX, LY]], list(N2), dm.CellType.triangle)
    return dm.create_box(comm, [[0, 0, 0], [LX, LY, 1.0]], list(N3), dm.CellType.tetrahedron)


def dolfinx_exterior(mesh):
    import dolfinx.mesh as dm
    from femtools.dolfinx.boundary_markers import local_facets
    n, exterior = local_facets(mesh)
    mid = dm.compute_midpoints(mesh, mesh.topology.dim - 1, np.arange(n, dtype=np.int32))
    on = np.zeros(n, bool)
    on[exterior] = True
    return mid, on


def markers_of(backend):
    if backend == "dolfin":
        from femtools.boundary_markers import BoundaryMarkers
        return BoundaryMarkers, dolfin_mesh, dolfin_exterior
    from femtools.backend.dolfinx import BoundaryMarkers
    return BoundaryMarkers, dolfinx_mesh, dolfinx_exterior


def main(mode, backend, folder):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    BoundaryMarkers, make, exterior = markers_of(backend)
    for dim in (2, 3):
        mesh = make(dim)
        mid, on = exterior(mesh)
        want = np.where(on, marker(mid), 0)
        if mode == "write":
            b = BoundaryMarkers(mesh)
            b.array()[:] = want
            b.write(os.path.join(folder, "{}_{}_{}d.xdmf".format(backend, comm.size, dim)))
            continue
        other = "dolfinx" if backend == "dolfin" else "dolfin"
        paths = sorted(glob.glob(os.path.join(folder, "{}_*_{}d.xdmf".format(other, dim))))
        assert paths, "nothing to read"
        for path in paths:
            got = np.asarray(BoundaryMarkers.read(path, mesh).array(), dtype=np.int64)
            ok = comm.allreduce(int(np.array_equal(got, want)), op=MPI.MIN)
            if comm.rank == 0 and ok:
                print("OK", os.path.basename(path), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
