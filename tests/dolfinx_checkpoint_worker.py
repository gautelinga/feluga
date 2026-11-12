"""Worker for tests/test_dolfinx_checkpoint.py: Functions in dolfin's HDF5 layout written
by one backend and read by the other (or the same), each run on its own number of ranks.

    [mpirun -n k] python3 tests/dolfinx_checkpoint_worker.py write-dolfin  DIR
    [mpirun -n k] python3 tests/dolfinx_checkpoint_worker.py write-dolfinx DIR
    [mpirun -n k] python3 tests/dolfinx_checkpoint_worker.py read-dolfin   DIR out.json
    [mpirun -n k] python3 tests/dolfinx_checkpoint_worker.py read-dolfinx  DIR out.json

The fields are analytic and periodic in x; a reader compares what it read with them at
its own dofs (the largest difference over the spaces, in JSON).
"""
import json
import os
import sys

import numpy as np

LX, LY, NX, NY = 1.0, 2.0, 6, 10
# name: (component degrees, periodic in x)
SPACES = {"P2": ((2,), True), "P1": ((1,), True), "W22": ((2, 2), True),
          "W21": ((2, 1), True), "P2_walls": ((2,), False)}


def f(x, y):
    return np.sin(2 * np.pi * x / LX) * y + np.cos(2 * np.pi * x / LX) + 0.5 * y**2


def g(x, y):
    return np.cos(2 * np.pi * x / LX) * (1.0 + y)


F_CPP = "sin(2*pi*x[0]/LX)*x[1] + cos(2*pi*x[0]/LX) + 0.5*x[1]*x[1]"
G_CPP = "cos(2*pi*x[0]/LX)*(1.0 + x[1])"


def dolfin_spaces():
    import dolfin as df
    from femtools.periodic import periodic_box_from_bounds
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(LX, LY), NX, NY, "right")
    out = {}
    for name, (degrees, periodic) in SPACES.items():
        cd = periodic_box_from_bounds({0: (0.0, LX)}) if periodic else None
        els = [df.FiniteElement("CG", mesh.ufl_cell(), d) for d in degrees]
        el = els[0] if len(els) == 1 else df.MixedElement(els)
        V = df.FunctionSpace(mesh, el, constrained_domain=cd)
        code = F_CPP if len(degrees) == 1 else (F_CPP, G_CPP)
        expr = df.Expression(code, LX=LX, degree=4)
        out[name] = (V, df.interpolate(expr, V))
    return out


def dolfinx_spaces():
    import basix.ufl
    from dolfinx import fem, mesh as dmesh
    from mpi4py import MPI
    from femtools.dolfinx.periodic import flat_collapse_map, periodic_box
    from femtools.dolfinx.unique import UniqueDofs
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [LX, LY]], [NX, NY],
                                 dmesh.CellType.triangle, diagonal=dmesh.DiagonalType.right)
    out = {}
    for name, (degrees, periodic) in SPACES.items():
        els = [basix.ufl.element("Lagrange", "triangle", d) for d in degrees]
        V = fem.functionspace(msh, els[0] if len(els) == 1 else basix.ufl.mixed_element(els))
        U = UniqueDofs(V, *periodic_box(V, periodic=(True, False))) if periodic else UniqueDofs(V)
        imap = V.dofmap.index_map
        n_loc = imap.size_local + imap.num_ghosts
        X, comp = np.zeros((n_loc, 3)), np.zeros(n_loc, dtype=int)
        if len(degrees) == 1:
            X = V.tabulate_dof_coordinates()
        else:
            for i in range(len(degrees)):
                Vi, parent = V.sub(i).collapse()
                parent = flat_collapse_map(parent)
                X[parent], comp[parent] = Vi.tabulate_dof_coordinates(), i
        own = U.owned_unique
        exact = np.where(comp[own] == 0, f(X[own, 0], X[own, 1]), g(X[own, 0], X[own, 1]))
        out[name] = (V, U, exact)
    return out


def main(mode, folder, out=None):
    res = {}
    if mode == "write-dolfin":
        from femtools.dolfin_layout import write_function
        for name, (V, u) in dolfin_spaces().items():
            write_function(os.path.join(folder, name + ".h5"), u, "u", mode="w")
    elif mode == "read-dolfin":
        import dolfin as df
        for name, (V, exact) in dolfin_spaces().items():
            u = df.Function(V)
            with df.HDF5File(V.mesh().mpi_comm(), os.path.join(folder, name + ".h5"), "r") as h:
                h.read(u, "u")
            d = np.abs(u.vector().get_local() - exact.vector().get_local())
            res[name] = float(V.mesh().mpi_comm().allreduce(d.max() if d.size else 0.0,
                                                           op=__import__("mpi4py").MPI.MAX))
    elif mode == "write-dolfinx":
        from femtools.dolfinx.checkpoint import write_dofs, write_function
        for name, (V, U, exact) in dolfinx_spaces().items():
            # the tables in a file of their own, linked, as checkpoints write them
            write_dofs(os.path.join(folder, name + "_dofs.h5"), V, U, "dofs")
            write_function(os.path.join(folder, name + ".h5"), exact, V, U, "u", mode="w",
                           dofs=(name + "_dofs.h5", "dofs"))
    elif mode == "read-dolfinx":
        from mpi4py import MPI
        from femtools.dolfinx.checkpoint import read_function
        for name, (V, U, exact) in dolfinx_spaces().items():
            got = read_function(os.path.join(folder, name + ".h5"), "u", V, U)
            d = np.abs(got - exact)
            res[name] = float(MPI.COMM_WORLD.allreduce(d.max() if d.size else 0.0, op=MPI.MAX))
    else:
        raise ValueError(mode)
    from mpi4py import MPI
    if out is not None and MPI.COMM_WORLD.rank == 0:
        with open(out, "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    main(*sys.argv[1:])
