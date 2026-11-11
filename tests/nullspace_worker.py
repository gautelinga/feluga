"""constant_nullspace on any number of ranks: on a P1 space and on the pressure of a
Taylor-Hood space, the vector is constant on the space's dofs, zero elsewhere, of unit
norm, and a Neumann problem's matrix maps it to zero; rank zero writes the largest
error to argv[1]. A petsc4py Mat gets the null space attached as a dolfin matrix does.

    mpirun -n 3 python3 tests/nullspace_worker.py OUT.txt
"""
import sys

import dolfin as df
import numpy as np

from femtools.nullspace import constant_nullspace

mesh = df.UnitSquareMesh(12, 12)
cell = mesh.ufl_cell()
Q = df.FunctionSpace(mesh, "P", 1)
p, q = df.TrialFunction(Q), df.TestFunction(Q)
A = df.assemble(df.inner(df.grad(p), df.grad(q)) * df.dx)
x = constant_nullspace(A, Q)[0]
n = Q.dim()
err = [abs(x.min() - n ** -0.5), abs(x.max() - n ** -0.5), (A * x).norm("linf")]

W = df.FunctionSpace(mesh, df.MixedElement([df.VectorElement("P", cell, 2),
                                            df.FiniteElement("P", cell, 1)]))
(u, p), (v, q) = df.TrialFunctions(W), df.TestFunctions(W)
A = df.as_backend_type(df.assemble(df.inner(df.grad(u), df.grad(v)) * df.dx
                                   + p * q * df.dx)).mat()
null = constant_nullspace(A, W.sub(1))
f = df.Function(W)
f.vector().axpy(1., null[0])
uf, pf = f.split(deepcopy=True)
err += [uf.vector().norm("linf"), abs(pf.vector().min() - pf.vector().max()),
        abs(null[0].norm("l2") - 1), abs(len(A.getNullSpace().getVecs()) - 1)]
if df.MPI.rank(mesh.mpi_comm()) == 0:
    with open(sys.argv[1], "w") as out:
        out.write(repr(float(np.max(err))))
