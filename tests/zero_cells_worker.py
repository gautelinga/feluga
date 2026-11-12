"""Worker for tests/test_zero_cells.py: femtools' collective data paths on a mesh of two
cells, run on more ranks than cells (some ranks hold none). Each step must finish on
every rank -- no rank-local exception, no hang -- and give the right values.

    mpirun -n 4 python3 tests/zero_cells_worker.py dolfinx|dolfin|dolfin-nested DIR out.json

`dolfin-nested` (the nested transfer on dolfin): dolfin cannot refine a mesh with ranks
that hold no cells (its edge numbering hangs), so femtools.plaza_periodic refuses it,
on every rank.
"""
import json
import os
import sys

import numpy as np


def f(x):
    return 1.0 + x[0] + 2.0 * x[1] + 0.5 * x[0] * x[1]


def dolfinx_run(folder):
    from mpi4py import MPI
    from dolfinx import fem, mesh as dmesh
    from femtools.dolfinx.checkpoint import read_function, write_dofs, write_function
    from femtools.dolfinx.unique import UniqueDofs
    from femtools.dolfinx.xdmf import XDMFSeries
    from femtools.dolfinx.xdmf_read import read_nodal
    from femtools.nested_transfer import Nested, refine_uniformly
    from femtools.tables import space_tables
    comm = MPI.COMM_WORLD
    msh = dmesh.create_rectangle(comm, [[0, 0], [1, 1]], [1, 1], dmesh.CellType.triangle)
    res = {"cells": comm.allgather(msh.topology.index_map(2).size_local)}
    V = fem.functionspace(msh, ("Lagrange", 2))
    U = UniqueDofs(V)
    u = fem.Function(V)
    u.interpolate(f)
    vals = U.restrict(u, U.create_vector()).getArray().copy()
    t = space_tables((V, U))
    res["tables_cells"] = comm.allreduce(len(t.gc))
    write_dofs(os.path.join(folder, "d.h5"), V, U)
    write_function(os.path.join(folder, "u.h5"), vals, V, U, "u", mode="w",
                   dofs=("d.h5", "dofs"))
    got = read_function(os.path.join(folder, "u.h5"), "u", V, U)
    res["checkpoint"] = comm.allreduce(float(np.abs(got - vals).max()) if len(vals) else 0.0,
                                       op=MPI.MAX)
    with XDMFSeries(comm, os.path.join(folder, "u.xdmf")) as s:
        s.write(u, 0.0)
    w = fem.Function(V)
    read_nodal(os.path.join(folder, "u.xdmf"), "f", -1, w)
    d = np.abs(w.x.array - u.x.array)
    res["xdmf"] = comm.allreduce(float(d.max()) if len(d) else 0.0, op=MPI.MAX)
    fine, anc = refine_uniformly(msh, 1, [1.0, 1.0], [])
    nest = Nested(msh, fine, anc, 1)
    Vf = fem.functionspace(fine, ("Lagrange", 2))
    Uf = UniqueDofs(Vf)
    P = nest.prolongation((V, U), (Vf, Uf))
    uf = fem.Function(Vf)
    uf.interpolate(f)
    exact = Uf.restrict(uf, Uf.create_vector())
    y = P.createVecLeft()
    x = U.create_vector()
    x.getArray()[:] = vals
    P.mult(x, y)
    y.axpy(-1.0, exact)
    res["prolongation"] = y.norm()
    R = nest.injection((Vf, Uf), (V, U))
    z = R.createVecLeft()
    R.mult(exact, z)
    z.axpy(-1.0, x)
    res["injection"] = z.norm()
    lat = nest.lattice((Vf, Uf))
    lat.write(os.path.join(folder, "lat.h5"), "phi", exact, mode="w")
    back = Uf.create_vector()
    lat.read(os.path.join(folder, "lat.h5"), "phi", back)
    back.axpy(-1.0, exact)
    res["lattice"] = back.norm()
    return comm, res


def dolfin_run(folder):
    import dolfin as df
    from femtools.dolfin_layout import read_function, write_function
    from femtools.tables import space_tables
    comm = df.MPI.comm_world
    msh = df.UnitSquareMesh(comm, 1, 1)
    res = {"cells": comm.allgather(msh.num_cells())}
    V = df.FunctionSpace(msh, "CG", 2)
    res["tables_cells"] = comm.allreduce(len(space_tables(V).gc))
    e = df.Expression("1.0 + x[0] + 2.0*x[1] + 0.5*x[0]*x[1]", degree=2)
    u = df.interpolate(e, V)
    write_function(os.path.join(folder, "ud.h5"), u, "u", mode="w")
    w = df.Function(V)
    read_function(os.path.join(folder, "ud.h5"), w, "u")
    res["checkpoint"] = (w.vector() - u.vector()).norm("linf")
    return comm, res


def dolfin_nested_run(folder):
    import dolfin as df
    from femtools.nested_transfer import Nested, refine_uniformly
    comm = df.MPI.comm_world
    msh = df.UnitSquareMesh(comm, 1, 1)
    res = {"cells": comm.allgather(msh.num_cells())}
    V = df.FunctionSpace(msh, "CG", 2)
    e = df.Expression("1.0 + x[0] + 2.0*x[1] + 0.5*x[0]*x[1]", degree=2)
    u = df.interpolate(e, V)
    try:
        fine, anc = refine_uniformly(msh, 1, [1.0, 1.0], [])
    except ValueError as e:
        res["refused"] = str(e)
        return comm, res
    nest = Nested(msh, fine, anc, 1)
    Vf = df.FunctionSpace(fine, "CG", 2)
    P = nest.prolongation(V, Vf)
    exact = df.as_backend_type(df.interpolate(e, Vf).vector()).vec()
    x = df.as_backend_type(u.vector()).vec()
    y = P.createVecLeft()
    P.mult(x, y)
    y.axpy(-1.0, exact)
    res["prolongation"] = y.norm()
    lat = nest.lattice(Vf)
    lat.write(os.path.join(folder, "latd.h5"), "phi", exact, mode="w")
    back = exact.duplicate()
    back.zeroEntries()
    lat.read(os.path.join(folder, "latd.h5"), "phi", back)
    back.axpy(-1.0, exact)
    res["lattice"] = back.norm()
    return comm, res


if __name__ == "__main__":
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ.get("ZC_TIMEOUT", "240")), exit=True)
    backend, folder, out = sys.argv[1:4]
    run = {"dolfinx": dolfinx_run, "dolfin": dolfin_run, "dolfin-nested": dolfin_nested_run}
    comm, res = run[backend](folder)
    if comm.rank == 0:
        with open(out, "w") as fh:
            json.dump(res, fh)
