"""Worker for tests/test_dolfinx_checkpoint.py: Functions in dolfin's HDF5 layout written
by one backend and read by the other (or the same), each run on its own number of ranks.

    [mpirun -n k] python3 tests/dolfinx_checkpoint_worker.py write-dolfin  DIR
    [mpirun -n k] python3 tests/dolfinx_checkpoint_worker.py write-dolfinx DIR
    [mpirun -n k] python3 tests/dolfinx_checkpoint_worker.py read-dolfin   DIR out.json
    [mpirun -n k] python3 tests/dolfinx_checkpoint_worker.py read-dolfin-tables DIR out.json
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
          "W21": ((2, 1), True), "P2_walls": ((2,), False), "V2": ("vector", True),
          "TH": ("taylor-hood", True)}


def f(x, y):
    return np.sin(2 * np.pi * x / LX) * y + np.cos(2 * np.pi * x / LX) + 0.5 * y**2


def g(x, y):
    return np.cos(2 * np.pi * x / LX) * (1.0 + y)


def h(x, y):
    return np.sin(2 * np.pi * x / LX) + y


F_CPP = "sin(2*pi*x[0]/LX)*x[1] + cos(2*pi*x[0]/LX) + 0.5*x[1]*x[1]"
G_CPP = "cos(2*pi*x[0]/LX)*(1.0 + x[1])"
H_CPP = "sin(2*pi*x[0]/LX) + x[1]"


def dolfin_spaces():
    import dolfin as df
    from femtools.periodic import periodic_box_from_bounds
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(LX, LY), NX, NY, "right")
    out = {}
    for name, (degrees, periodic) in SPACES.items():
        cd = periodic_box_from_bounds({0: (0.0, LX)}) if periodic else None
        if degrees == "vector":
            V = df.VectorFunctionSpace(mesh, "CG", 2, constrained_domain=cd)
            out[name] = (V, df.interpolate(df.Expression((F_CPP, G_CPP), LX=LX, degree=4), V))
            continue
        if degrees == "taylor-hood":           # a vector P2 and a P1: u_x, u_y, p
            el = df.MixedElement([df.VectorElement("CG", mesh.ufl_cell(), 2),
                                  df.FiniteElement("CG", mesh.ufl_cell(), 1)])
            V = df.FunctionSpace(mesh, el, constrained_domain=cd)
            expr = df.Expression((F_CPP, G_CPP, H_CPP), LX=LX, degree=4)
            out[name] = (V, df.interpolate(expr, V))
            continue
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
        if degrees == "vector":
            V = fem.functionspace(msh, ("Lagrange", 2, (2,)))
        elif degrees == "taylor-hood":
            V = fem.functionspace(msh, basix.ufl.mixed_element([
                basix.ufl.element("Lagrange", "triangle", 2, shape=(2,)),
                basix.ufl.element("Lagrange", "triangle", 1)]))
        else:
            els = [basix.ufl.element("Lagrange", "triangle", d) for d in degrees]
            V = fem.functionspace(msh, els[0] if len(els) == 1 else basix.ufl.mixed_element(els))
        U = UniqueDofs(V, *periodic_box(V, periodic=(True, False))) if periodic else UniqueDofs(V)
        imap = V.dofmap.index_map
        bs = V.dofmap.index_map_bs
        n_loc = (imap.size_local + imap.num_ghosts) * bs
        X, comp = np.zeros((n_loc, 3)), np.zeros(n_loc, dtype=int)
        if degrees == "vector":
            X = np.repeat(V.tabulate_dof_coordinates(), bs, axis=0)
            comp = np.tile(np.arange(bs), n_loc // bs)
        elif degrees != "taylor-hood" and len(degrees) == 1:
            X = V.tabulate_dof_coordinates()
        else:
            leaves = ([V.sub(0).sub(0), V.sub(0).sub(1), V.sub(1)] if degrees == "taylor-hood"
                      else [V.sub(i) for i in range(len(degrees))])
            for i, S in enumerate(leaves):
                Vi, parent = S.collapse()
                parent = flat_collapse_map(parent)
                X[parent], comp[parent] = Vi.tabulate_dof_coordinates(), i
        own = U.owned_unique
        x, y = X[own, 0], X[own, 1]
        exact = np.choose(comp[own], [f(x, y), g(x, y), h(x, y)])
        out[name] = (V, U, exact)
    return out


def main(mode, folder, out=None):
    res = {}
    if mode == "write-dolfin":
        from femtools.dolfin_layout import write_function
        for name, (V, u) in dolfin_spaces().items():
            write_function(os.path.join(folder, name + ".h5"), u, "u", mode="w")
    elif mode in ("read-dolfin", "read-dolfin-tables"):
        import dolfin as df
        from femtools.dolfin_layout import read_function
        for name, (V, exact) in dolfin_spaces().items():
            u = df.Function(V)
            path = os.path.join(folder, name + ".h5")
            if mode == "read-dolfin-tables":      # femtools.tables.read_values
                read_function(path, u, "u")
            else:                                 # dolfin's own reader
                with df.HDF5File(V.mesh().mpi_comm(), path, "r") as h:
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
