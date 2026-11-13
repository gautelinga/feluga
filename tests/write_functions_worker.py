"""Worker for tests/test_dolfin_layout.py: `write_functions_h5` on a backend writes the
file that consecutive `write_function_h5` calls write -- the same groups, datasets (dtype,
shape, values), attributes and links -- with the dof tables linked and written, and the
Functions read back from it.

    mpirun -n R python3 tests/write_functions_worker.py BACKEND OUT

BACKEND is dolfin or dolfinx. Rank zero prints a line a check.
"""
import importlib
import os
import sys

import h5py
import numpy as np


def walk(path):
    """Every group, dataset and link of path, as comparable tuples."""
    out = []
    with h5py.File(path, "r") as h5:
        def visit(g):
            for k in sorted(g.keys()):
                link = g.get(k, getlink=True)
                name = g.name.rstrip("/") + "/" + k
                attrs = sorted((a, np.asarray(v).tobytes(), str(np.asarray(v).dtype))
                               for a, v in g[k].attrs.items()) if not isinstance(
                                   link, h5py.ExternalLink) else []
                if isinstance(link, h5py.ExternalLink):
                    out.append(("link", name, link.filename, link.path))
                elif isinstance(g[k], h5py.Group):
                    out.append(("group", name, attrs))
                    visit(g[k])
                else:
                    d = g[k]
                    out.append(("dataset", name, str(d.dtype), d.shape,
                                np.asarray(d[()]).tobytes(), attrs))
        out.append(("root", sorted(h5.attrs.keys())))
        visit(h5)
    return out


def main(backend, out):
    L = importlib.import_module("femtools.backend." + backend)
    comm = L.MPI.comm_world
    mesh = L.UnitSquareMesh(6, 5)
    V = L.FunctionSpace(mesh, "CG", 2, constrained_domain=L.PeriodicBox({0: (0., 1.)}))
    Q = L.FunctionSpace(mesh, "CG", 1)
    u, u1, p = L.Function(V), L.Function(V), L.Function(Q)
    for f, k in ((u, 1.0), (u1, 2.0), (p, 3.0)):
        x = f.vector()
        lo = x.local_range()[0] if backend == "dolfin" else 0
        x.set_local(k + 1e-3 * (lo + np.arange(len(x.get_local()))) + 1e-6 * comm.rank)
        x.apply("insert")
    folder = os.path.join(out, backend)
    if comm.rank == 0:
        os.makedirs(folder, exist_ok=True)
    comm.Barrier()

    def at(name):
        return os.path.join(folder, name)

    L.write_dof_tables(at("dofs.h5"), V, "dofs")
    lines = []
    for label, dofs in (("linked", ("dofs.h5", "dofs")), ("tables", None)):
        L.write_function_h5(at("one_" + label + ".h5"), u, "current", mode="w", dofs=dofs)
        L.write_function_h5(at("one_" + label + ".h5"), u1, "previous", dofs=dofs)
        L.write_functions_h5(at("all_" + label + ".h5"), [(u, "current"), (u1, "previous")],
                             dofs=dofs)
        same = None
        if comm.rank == 0:
            same = walk(at("one_" + label + ".h5")) == walk(at("all_" + label + ".h5"))
        same = comm.bcast(same, root=0)
        back = True
        for name, f in (("current", u), ("previous", u1)):
            g = L.Function(V)
            L.read_function_h5(at("all_" + label + ".h5"), g, name)
            back &= bool(np.array_equal(g.vector().get_local(), f.vector().get_local()))
        back = comm.allreduce(int(back), op=__import__("mpi4py").MPI.MIN)
        lines.append("{}: {} {}".format(label, "same" if same else "DIFFERENT",
                                        "read" if back else "UNREAD"))
    # two spaces in one call, the second appended to an existing file
    L.write_function_h5(at("one_two.h5"), u, "u", mode="w")
    L.write_function_h5(at("one_two.h5"), p, "p")
    L.write_functions_h5(at("all_two.h5"), [(u, "u")])
    L.write_functions_h5(at("all_two.h5"), [(p, "p")], mode="a")
    same = None
    if comm.rank == 0:
        same = walk(at("one_two.h5")) == walk(at("all_two.h5"))
    lines.append("two spaces: " + ("same" if comm.bcast(same, root=0) else "DIFFERENT"))
    # tables of another element are refused on every rank, before anything is written
    try:
        L.write_functions_h5(at("bad.h5"), [(u, "u"), (p, "p")], dofs=("dofs.h5", "dofs"))
        why = None
    except ValueError as e:
        why = str(e)
    lines.append("other element: " + ("refused" if why and "another element" in why
                                      else str(why)))
    if comm.rank == 0:
        print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
