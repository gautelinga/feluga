"""Worker for tests/test_dolfinx_xdmf.py: time series written on DOLFINx (XDMFSeries) and by
dolfin (XDMFFile2), read back on dolfin.

    [mpirun -n k] python3 tests/dolfinx_xdmf_worker.py write-dolfinx DIR
    python3 tests/dolfinx_xdmf_worker.py write-dolfin DIR
    python3 tests/dolfinx_xdmf_worker.py read-dolfin DIR out.json

Fields (two steps, the second scaled by 2): a periodic P2 scalar, a P1 scalar, a periodic
P2 vector. The reader compares each step with the field's interpolant, and the files'
structure (grids(): topology, attribute names and widths, times) with dolfin's.
"""
import json
import os
import sys

import numpy as np

LX, LY, NX, NY = 1.0, 2.0, 6, 10
TIMES = (0.0, 0.5)
# name: (degree, vector, periodic)
FIELDS = {"s2": (2, False, True), "s1": (1, False, False), "v2": (2, True, True)}


def f(x, y):
    return np.sin(2 * np.pi * x / LX) * y + np.cos(2 * np.pi * x / LX) + 0.5 * y**2


def g(x, y):
    return np.cos(2 * np.pi * x / LX) * (1.0 + y)


F_CPP = "sin(2*pi*x[0]/LX)*x[1] + cos(2*pi*x[0]/LX) + 0.5*x[1]*x[1]"
G_CPP = "cos(2*pi*x[0]/LX)*(1.0 + x[1])"


def dolfin_fields():
    import dolfin as df
    from femtools.periodic import periodic_box_from_bounds
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(LX, LY), NX, NY, "right")
    out = {}
    for name, (degree, vector, periodic) in FIELDS.items():
        cd = periodic_box_from_bounds({0: (0.0, LX)}) if periodic else None
        V = (df.VectorFunctionSpace(mesh, "CG", degree, constrained_domain=cd) if vector
             else df.FunctionSpace(mesh, "CG", degree, constrained_domain=cd))
        expr = df.Expression((F_CPP, G_CPP) if vector else F_CPP, LX=LX, degree=4)
        out[name] = df.interpolate(expr, V)
    return out


def write_dolfinx(folder):
    from dolfinx import fem, mesh as dmesh
    from mpi4py import MPI
    from femtools.dolfinx.xdmf import XDMFSeries
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [LX, LY]], [NX, NY],
                                 dmesh.CellType.triangle, diagonal=dmesh.DiagonalType.right)
    for name, (degree, vector, periodic) in FIELDS.items():
        V = fem.functionspace(msh, ("Lagrange", degree, (2,)) if vector
                              else ("Lagrange", degree))
        u = fem.Function(V, name=name)
        with XDMFSeries(MPI.COMM_WORLD, os.path.join(folder, "x_" + name + ".xdmf")) as out:
            for k, t in enumerate(TIMES):
                # a periodic field holds its master's value at a slave: evaluate the
                # periodic function, equal on both sides to round-off
                if vector:
                    u.interpolate(lambda X, k=k: np.vstack([(1 + k) * f(X[0], X[1]),
                                                            (1 + k) * g(X[0], X[1])]))
                else:
                    u.interpolate(lambda X, k=k: (1 + k) * f(X[0], X[1]))
                out.write(u, t)


def write_dolfin(folder):
    import dolfin as df
    from femtools.xdmf import XDMFFile2
    for name, u in dolfin_fields().items():
        u.rename(name, name)
        f2 = XDMFFile2(df.MPI.comm_world, os.path.join(folder, "d_" + name + ".xdmf"))
        f2.parameters["rewrite_function_mesh"] = False
        f2.parameters["flush_output"] = True
        w = u.copy(deepcopy=True)
        w.rename(name, name)
        for k, t in enumerate(TIMES):
            w.vector()[:] = (1 + k) * u.vector()
            f2.write(w, t)
        f2.close()


def read_dolfin(folder, out):
    import dolfin as df
    from femtools.xdmf import grids, XDMFFile2
    res = {}
    for name, u in dolfin_fields().items():
        errs = []
        for k in range(len(TIMES)):
            back = df.Function(u.function_space())
            with XDMFFile2(df.MPI.comm_world, os.path.join(folder, "x_" + name + ".xdmf")) as r:
                r.read_checkpoint(back, name, k)
            errs.append(float(np.max(np.abs(back.vector().get_local()
                                            - (1 + k) * u.vector().get_local()))))
        res[name] = max(errs)

        def xml(prefix):
            with open(os.path.join(folder, prefix + name + ".xdmf")) as fh:
                return fh.read().replace(prefix + name + ".h5", "FILE.h5")
        # the same XML, word for word, but for the HDF5 file's name
        res[name + "_structure"] = xml("x_") == xml("d_")
        res[name + "_steps"] = len(grids(os.path.join(folder, "x_" + name + ".xdmf")))
    if df.MPI.rank(df.MPI.comm_world) == 0:
        with open(out, "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    mode, folder = sys.argv[1:3]
    {"write-dolfinx": lambda: write_dolfinx(folder), "write-dolfin": lambda: write_dolfin(folder),
     "read-dolfin": lambda: read_dolfin(folder, sys.argv[3])}[mode]()
