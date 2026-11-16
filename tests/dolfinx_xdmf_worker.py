"""Worker for tests/test_dolfinx_xdmf.py: time series written on DOLFINx (XDMFSeries) and by
dolfin (XDMFFile2), read back on dolfin.

    [mpirun -n k] python3 tests/dolfinx_xdmf_worker.py write-dolfinx DIR
    python3 tests/dolfinx_xdmf_worker.py write-dolfin DIR
    python3 tests/dolfinx_xdmf_worker.py read-dolfin DIR out.json

Fields (two steps, the second scaled by 2): a periodic P2 scalar, a P1 scalar, a periodic
P2 vector, and PAIR, the vector and the P1 scalar written together on one node set (as
solve-stokes writes u and p). The reader compares each step with the field's interpolant,
and the files' structure (grids(): topology, attribute names and widths, times) with
dolfin's.
"""
import json
import os
import sys

import numpy as np

LX, LY, NX, NY = 1.0, 2.0, 6, 10
TIMES = (0.0, 0.5)
# name: (degree, vector, periodic)
FIELDS = {"s2": (2, False, True), "s1": (1, False, False), "v2": (2, True, True)}
PAIR = ("v2", "s1")         # written together, x_pair.xdmf and d_pair.xdmf


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
    pair = []
    for name in PAIR:
        degree, vector, _ = FIELDS[name]
        V = fem.functionspace(msh, ("Lagrange", degree, (2,)) if vector
                              else ("Lagrange", degree))
        pair.append(fem.Function(V, name=name))
    with XDMFSeries(MPI.COMM_WORLD, os.path.join(folder, "x_pair.xdmf")) as out:
        for k, t in enumerate(TIMES):
            pair[0].interpolate(lambda X, k=k: np.vstack([(1 + k) * f(X[0], X[1]),
                                                          (1 + k) * g(X[0], X[1])]))
            pair[1].interpolate(lambda X, k=k: (1 + k) * f(X[0], X[1]))
            out.write(tuple(pair), t)


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
    fields = dolfin_fields()
    pair = []
    for name in PAIR:
        w = fields[name].copy(deepcopy=True)
        w.rename(name, name)
        pair.append(w)
    f2 = XDMFFile2(df.MPI.comm_world, os.path.join(folder, "d_pair.xdmf"))
    f2.parameters["rewrite_function_mesh"] = False
    f2.parameters["flush_output"] = True
    for k, t in enumerate(TIMES):
        for w, name in zip(pair, PAIR):
            w.vector()[:] = (1 + k) * fields[name].vector()
        f2.write(tuple(pair), t)
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
    fields = dolfin_fields()
    errs = []
    for name in PAIR:
        u = fields[name]
        for k in range(len(TIMES)):
            back = df.Function(u.function_space())
            with XDMFFile2(df.MPI.comm_world, os.path.join(folder, "x_pair.xdmf")) as r:
                r.read_checkpoint(back, name, k)
            errs.append(float(np.max(np.abs(back.vector().get_local()
                                            - (1 + k) * u.vector().get_local()))))
    res["pair"] = max(errs)

    def xml(prefix):
        with open(os.path.join(folder, prefix + "pair.xdmf")) as fh:
            return fh.read().replace(prefix + "pair.h5", "FILE.h5")
    res["pair_structure"] = xml("x_") == xml("d_")
    res["pair_steps"] = len(grids(os.path.join(folder, "x_pair.xdmf")))
    if df.MPI.rank(df.MPI.comm_world) == 0:
        with open(out, "w") as fh:
            json.dump(res, fh)


def read_dolfinx(folder, out):
    """DOLFINx reads every step of both backends' series (femtools.dolfinx.xdmf_read) into
    its own spaces, the periodic ones included (UniqueDofs: restricted values compared)."""
    from dolfinx import fem, mesh as dmesh
    from mpi4py import MPI
    from femtools.dolfinx.periodic import periodic_box
    from femtools.dolfinx.unique import UniqueDofs
    from femtools.dolfinx.xdmf_read import read_nodal
    comm = MPI.COMM_WORLD
    msh = dmesh.create_rectangle(comm, [[0, 0], [LX, LY]], [NX, NY], dmesh.CellType.triangle,
                                 diagonal=dmesh.DiagonalType.right)
    res = {}
    for name, (degree, vector, periodic) in FIELDS.items():
        V = fem.functionspace(msh, ("Lagrange", degree, (2,)) if vector else ("Lagrange", degree))
        U = UniqueDofs(V, *periodic_box(V, periodic=(True, False))) if periodic else UniqueDofs(V)
        X = V.tabulate_dof_coordinates()[U.owned_unique // V.dofmap.index_map_bs]
        comp = U.owned_unique % V.dofmap.index_map_bs
        base = np.where(comp == 0, f(X[:, 0], X[:, 1]), g(X[:, 0], X[:, 1]))
        errs = []
        for prefix in ("x_", "d_"):
            for k in range(len(TIMES)):
                u = fem.Function(V)
                read_nodal(os.path.join(folder, prefix + name + ".xdmf"), name, k, u)
                d = np.abs(u.x.array[U.owned_unique] - (1 + k) * base)
                errs.append(comm.allreduce(float(d.max()) if d.size else 0.0, op=MPI.MAX))
        res[name] = max(errs)
    # read at the nodes of the spaces above: no plain space made
    from femtools.dolfinx import xdmf_read
    res["plain_spaces"] = len(xdmf_read._PLAIN)
    # onto another degree, through a plain space of the file's: the P2 field's values at
    # the P1 nodes, its own
    V1 = fem.functionspace(msh, ("Lagrange", 1))
    X = V1.tabulate_dof_coordinates()
    errs = []
    for prefix in ("x_", "d_"):
        u = fem.Function(V1)
        read_nodal(os.path.join(folder, prefix + "s2.xdmf"), "s2", 0, u)
        d = np.abs(u.x.array - f(X[:, 0], X[:, 1]))
        errs.append(comm.allreduce(float(d.max()) if d.size else 0.0, op=MPI.MAX))
    res["s2_into_P1"] = max(errs)
    res["plain_spaces_after"] = len(xdmf_read._PLAIN)
    # the pair's P1 field off its P2 grid goes through a plain space of the grid's too
    perrs = []
    for name in PAIR:
        degree, vector, periodic = FIELDS[name]
        V = fem.functionspace(msh, ("Lagrange", degree, (2,)) if vector else ("Lagrange", degree))
        U = UniqueDofs(V, *periodic_box(V, periodic=(True, False))) if periodic else UniqueDofs(V)
        X = V.tabulate_dof_coordinates()[U.owned_unique // V.dofmap.index_map_bs]
        comp = U.owned_unique % V.dofmap.index_map_bs
        base = np.where(comp == 0, f(X[:, 0], X[:, 1]), g(X[:, 0], X[:, 1]))
        for prefix in ("x_", "d_"):
            for k in range(len(TIMES)):
                u = fem.Function(V)
                read_nodal(os.path.join(folder, prefix + "pair.xdmf"), name, k, u)
                d = np.abs(u.x.array[U.owned_unique] - (1 + k) * base)
                perrs.append(comm.allreduce(float(d.max()) if d.size else 0.0, op=MPI.MAX))
    res["pair"] = max(perrs)
    if comm.rank == 0:
        with open(out, "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    mode, folder = sys.argv[1:3]
    {"write-dolfinx": lambda: write_dolfinx(folder), "write-dolfin": lambda: write_dolfin(folder),
     "read-dolfin": lambda: read_dolfin(folder, sys.argv[3]),
     "read-dolfinx": lambda: read_dolfinx(folder, sys.argv[3])}[mode]()
