"""Worker for tests/test_dolfinx_unique.py: dolfin (constrained_domain from
feluga.periodic) against dolfinx (UniqueDofs from feluga.dolfinx) in one process, on
the same periodic mesh: unique dof count, norm and sum of a periodic field, Frobenius norm
of the folded operator. Serial. Writes JSON.

    python3 tests/dolfinx_cross_worker.py out.json
"""
import json
import sys

import numpy as np

import dolfin as df
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

from feluga.dolfinx.assembly import assemble_matrix, create_matrix
from feluga.dolfinx.periodic import periodic_box as periodic_box_x
from feluga.periodic import periodic_box

LX, LY, NX, NY, Q = 1.0, 2.0, 12, 24, 6
res = {}

# dolfin: periodic in x through the constrained domain
dm = df.RectangleMesh(df.Point(0, 0), df.Point(LX, LY), NX, NY, "right")
Vd = df.FunctionSpace(dm, "CG", 2, constrained_domain=periodic_box(
    upper=(LX, LY), periodic=(True, False), lower=(0, 0)))
fd = df.interpolate(df.Expression("cos(2*pi*x[0])*x[1] + 1", degree=4), Vd)
xd = df.SpatialCoordinate(dm)
ud, vd = df.TrialFunction(Vd), df.TestFunction(Vd)
kd = 1 + 0.5 * df.sin(2 * df.pi * xd[0]) ** 2
dx_d = df.dx(metadata={"quadrature_degree": Q})
Ad = df.as_backend_type(df.assemble(kd * df.inner(df.grad(ud), df.grad(vd)) * dx_d
                                    + ud * vd * dx_d)).mat()

# dolfinx: the same mesh, periodic in x through UniqueDofs
xm = mesh.create_rectangle(MPI.COMM_SELF, [[0, 0], [LX, LY]], [NX, NY])
V = fem.functionspace(xm, ("Lagrange", 2))
from feluga.dolfinx.unique import UniqueDofs  # noqa: E402

U = UniqueDofs(V, *periodic_box_x(V, periodic=(True, False)))
fx = fem.Function(V)
fx.interpolate(lambda X: np.cos(2 * np.pi * X[0]) * X[1] + 1)
c = U.restrict(fx, U.create_vector())
x = ufl.SpatialCoordinate(xm)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
k = 1 + 0.5 * ufl.sin(2 * ufl.pi * x[0]) ** 2
dx = ufl.dx(metadata={"quadrature_degree": Q})
a = fem.form(k * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + u * v * dx)
K = create_matrix(a, U)
assemble_matrix(K, a)

vd_ = df.as_backend_type(fd.vector()).vec()
res["dofs"] = [Vd.dim(), U.n_global]
res["norm"] = abs(c.norm() - vd_.norm()) / vd_.norm()
res["sum"] = abs(c.sum() - vd_.sum()) / abs(vd_.sum())
nd, nx = Ad.norm(PETSc.NormType.FROBENIUS), K.norm(PETSc.NormType.FROBENIUS)
res["operator"] = abs(nd - nx) / nd

with open(sys.argv[1], "w") as fh:
    json.dump(res, fh)
