"""fast_interpolation against dolfin.interpolate on any number of ranks, on a periodic
mesh: a P2 component of a mixed Function onto P1, P1 onto P2, and a periodic P2
vector onto a plain one; rank zero writes the largest difference to argv[1].

    mpirun -n 3 python3 tests/fast_interpolation_worker.py OUT.txt
"""
import sys

import dolfin as df
import numpy as np
from mpi4py import MPI

from femtools.fast_interpolation import interpolate
from femtools.periodic import PeriodicBox

mesh = df.RectangleMesh(df.Point(0, 0), df.Point(2, 1), 24, 12)
pb = PeriodicBox([2, 1], periodic=[True, False])
P1 = df.FunctionSpace(mesh, "CG", 1, constrained_domain=pb)
el = df.FiniteElement("CG", mesh.ufl_cell(), 2)
W = df.FunctionSpace(mesh, df.MixedElement([el, el]), constrained_domain=pb)
w = df.interpolate(df.Expression(("sin(pi*x[0])*x[1]", "cos(pi*x[0]) + x[1]*x[1]"), degree=4), W)
p1 = df.interpolate(df.Expression("sin(pi*x[0])*x[1]", degree=2), P1)
u = df.interpolate(df.Expression(("x[1]*sin(pi*x[0])", "x[0]"), degree=4),
                   df.VectorFunctionSpace(mesh, "CG", 2, constrained_domain=pb))
cases = [(c, P1) for c in w.split()]
cases += [(p1, df.FunctionSpace(mesh, "CG", 2, constrained_domain=pb)),
          (u, df.VectorFunctionSpace(mesh, "CG", 2))]
err = 0.
for f, V in cases:
    err = max(err, np.abs(interpolate(f, V).vector().get_local()
                          - df.interpolate(f, V).vector().get_local()).max(initial=0.))
err = MPI.COMM_WORLD.allreduce(err, op=MPI.MAX)
if MPI.COMM_WORLD.rank == 0:
    open(sys.argv[1], "w").write(repr(err))
