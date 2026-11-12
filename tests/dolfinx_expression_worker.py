"""Worker for tests/test_dolfinx_expression.py under MPI: every rank builds the same
CompiledExpression (rank 0 compiles into a fresh cache) and evaluates it.

    mpirun -n 3 python3 tests/dolfinx_expression_worker.py out.json
"""
import json
import sys

import numpy as np
from mpi4py import MPI

from femtools.dolfinx.expression import CompiledExpression

comm = MPI.COMM_WORLD
x = np.vstack([np.linspace(0, 1, 7), np.linspace(1, 2, 7), np.zeros(7)])
e = CompiledExpression("a*x[0] + n*x[1] + rank_unused", a=2.0, n=3.0, rank_unused=0.0)
d = float(np.abs(e(x) - (2 * x[0] + 3 * x[1])).max())
ok = comm.allreduce(int(d < 1e-14), op=MPI.MIN)
if comm.rank == 0:
    with open(sys.argv[1], "w") as f:
        json.dump({"ok": bool(ok), "size": comm.size}, f)
