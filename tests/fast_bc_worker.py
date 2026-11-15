"""fast_bc.apply against DirichletBC.apply on any number of ranks, on a periodic P2
space: a Constant, a Function changed in place and a vector Constant changed in
place, and a list of overlapping conditions, owned and ghost values, and the dolfin
backend's fixed_dofs against the owned dofs dolfin's apply changes (1 where they differ); rank zero writes the largest difference to
argv[1].

    mpirun -n 3 python3 tests/fast_bc_worker.py OUT.txt
"""
import sys

import dolfin as df
import numpy as np
from mpi4py import MPI

from femtools.fast_bc import apply
from femtools.periodic import PeriodicBox

mesh = df.RectangleMesh(df.Point(0, 0), df.Point(2, 1), 24, 12)
pb = PeriodicBox([2, 1], periodic=[True, False])
V = df.FunctionSpace(mesh, "CG", 2, constrained_domain=pb)
W = df.VectorFunctionSpace(mesh, "CG", 2, constrained_domain=pb)
g = df.interpolate(df.Expression("sin(x[0])", degree=2), V)
c = df.Constant((1., -2.))
cases = [(df.DirichletBC(V, df.Constant(0.3), "on_boundary"), V),
         (df.DirichletBC(V, g, "on_boundary"), V),
         (df.DirichletBC(W, c, "on_boundary"), W)]


def ghosted(x):
    with df.as_backend_type(x).vec().localForm() as local:
        return local.getArray().copy()


err = 0.
for step in range(2):
    g.vector()[:] = step + 1.
    c.assign(df.Constant((step, 3. * step)))
    for bc, S in cases:
        f = df.interpolate(df.Expression(("x[0]*x[1]", "x[0]") if S is W else "x[0]*x[1]",
                                         degree=2), S)
        a, b = f.vector().copy(), f.vector().copy()
        bc.apply(a)
        apply(bc, b)
        err = max(err, np.abs(ghosted(a) - ghosted(b)).max(initial=0.))
# several conditions at once, in order, overlapping, an Expression among them
h = df.interpolate(df.Expression("2 + sin(x[0])", degree=2), V)
several = [df.DirichletBC(V, df.Constant(0.3), "on_boundary"),
           df.DirichletBC(V, df.Expression("x[1] - 4", degree=2), "near(x[0], 0)"),
           df.DirichletBC(V, h, "near(x[1], 0)"),
           df.DirichletBC(V, df.Constant(-1.), "near(x[1], 1)")]
for step in range(2):
    h.vector()[:] = step - 3.
    f = df.interpolate(df.Expression("x[0]*x[1]", degree=2), V)
    a, b = f.vector().copy(), f.vector().copy()
    for bc in several:
        bc.apply(a)
    apply(several, b)
    err = max(err, np.abs(ghosted(a) - ghosted(b)).max(initial=0.))
# the dolfin backend's fixed_dofs: the owned global dofs, condition after condition
# (each condition's: the owned entries dolfin's own apply changes, which gathers from the
# ranks that hold the boundary facets; condition after condition)
from femtools.backend.dolfin import fixed_dofs
start = V.dofmap().ownership_range()[0]
got, want = [], []
for bc in several:
    x = df.Function(V).vector()
    x[:] = 12345.0
    bc.apply(x)
    changed = start + np.flatnonzero(x.get_local() != 12345.0)
    mine = fixed_dofs(V, [bc])
    err = max(err, 0. if sorted(mine.tolist()) == changed.tolist() else 1.)
    got.append(mine)
err = max(err, 0. if fixed_dofs(V, several).tolist() == np.concatenate(got).tolist() else 1.)
err = MPI.COMM_WORLD.allreduce(err, op=MPI.MAX)
if MPI.COMM_WORLD.rank == 0:
    open(sys.argv[1], "w").write(repr(err))
