"""Worker for tests/test_dg0_h5.py: a DG0 Function in dolfin's HDF5 layout through
feluga.backend, written or read back.

    [mpirun -n k] python3 tests/dg0_h5_worker.py BACKEND PATH w|r

The field is the cell means of x + 3yz on the unit cube (3^3, tetrahedra). `w` writes it
as "f"; `r` reads "f" and prints the largest difference from this backend's own field.
"""
import sys

import numpy as np

from feluga.backend import load


def main(backend, path, mode):
    be = load(backend)
    mesh = be.BoxMesh(be.Point(0, 0, 0), be.Point(1, 1, 1), 3, 3, 3)
    D = be.FunctionSpace(mesh, "DG", 0)
    v = be.TestFunction(D)
    x = be.SpatialCoordinate(mesh)
    f = be.Function(D)
    f.vector().set_local(be.assemble((x[0] + 3 * x[1] * x[2]) * v * be.dx).get_local()
                         / be.assemble(v * be.dx).get_local())
    f.vector().apply("insert")
    if mode == "w":
        be.write_function_h5(path, f, "f", mode="w")
        return
    from mpi4py import MPI
    g = be.read_function_h5(path, be.Function(D), "f")
    err = np.abs(g.vector().get_local() - f.vector().get_local()).max(initial=0.0)
    err = be.mesh_comm(mesh).allreduce(err, op=MPI.MAX)
    if be.mesh_comm(mesh).rank == 0:
        print("max err {:.3e}".format(err))


if __name__ == "__main__":
    main(*sys.argv[1:4])
