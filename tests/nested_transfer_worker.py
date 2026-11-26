"""Worker for tests/test_nested_transfer.py: feluga.nested_transfer on one backend.

    [mpirun -n k] python3 tests/nested_transfer_worker.py dolfin|dolfinx out.json h5dir [read_from]

For a rectangle (crossed diagonals) and a box, periodic in x, refined uniformly once and
(the rectangle) twice: the prolongation of a P2 field reproduces it on the fine P2 and P1
spaces; the injection from a fine mixed (P1 or P2) field to the coarse P2 and P1 spaces
gives the coarse interpolant; R P is the identity. A fine mixed field is written by its
lattice to h5dir/<backend>_<ranks>.h5, and with `read_from` (another run's file) read
back and compared with this run's interpolant. The matrices and the lattice are the
same, entry for entry, when the fine cells are taken a few at a time.
"""
import json
import os
import sys

import numpy as np

CASES = {"rect1": (2, 6, 1), "rect2": (2, 6, 2), "box1": (3, 3, 1)}


def f(x):
    return x[1] ** 2 - 0.4 * x[1] + (0.7 * x[1] * x[2] if len(x) > 2 else 0.0)


F_CPP = {2: "x[1]*x[1] - 0.4*x[1]", 3: "x[1]*x[1] - 0.4*x[1] + 0.7*x[1]*x[2]"}


class Dolfin:
    def __init__(self):
        import dolfin as df
        self.df = df
        self.comm = df.MPI.comm_world

    def mesh(self, d, n):
        df = self.df
        if d == 2:
            return df.RectangleMesh(df.Point(0, 0), df.Point(1, 1), n, n, "crossed")
        return df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 1), n, n, n)

    def space(self, mesh, degree, mixed=False):
        from feluga.periodic import periodic_box
        df = self.df
        d = mesh.topology().dim()
        box = periodic_box([1.0] * d, periodic=[True] + [False] * (d - 1))
        if mixed:
            el = df.FiniteElement("CG", mesh.ufl_cell(), degree)
            return df.FunctionSpace(mesh, df.MixedElement([el, el]), constrained_domain=box)
        return df.FunctionSpace(mesh, "CG", degree, constrained_domain=box)

    def interp(self, V, mixed=False, scale=(1.0, 2.0)):
        df = self.df
        d = V.mesh().topology().dim()
        if mixed:
            e = df.Expression(tuple("{}*({})".format(s, F_CPP[d]) for s in scale), degree=2)
        else:
            e = df.Expression(F_CPP[d], degree=2)
        return df.as_backend_type(df.interpolate(e, V).vector()).vec().copy()

    def zero(self, V):
        return self.df.as_backend_type(self.df.Function(V).vector()).vec().copy()


class Dolfinx:
    def __init__(self):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD

    def mesh(self, d, n):
        import dolfinx.mesh as dm
        if d == 2:
            return dm.create_rectangle(self.comm, [[0, 0], [1, 1]], [n, n],
                                       diagonal=dm.DiagonalType.crossed)
        return dm.create_box(self.comm, [[0, 0, 0], [1, 1, 1]], [n, n, n])

    def space(self, mesh, degree, mixed=False):
        import basix.ufl
        from dolfinx import fem
        from feluga.dolfinx.periodic import periodic_box
        from feluga.dolfinx.unique import UniqueDofs
        d = mesh.topology.dim
        el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree)
        V = fem.functionspace(mesh, basix.ufl.mixed_element([el, el]) if mixed else el)
        U = UniqueDofs(V, *periodic_box(V, upper=[1.0] * d,
                                        periodic=[True] + [False] * (d - 1)), interleave=True)
        return V, U

    def interp(self, space, mixed=False, scale=(1.0, 2.0)):
        from dolfinx import fem
        V, U = space
        u = fem.Function(V)
        if mixed:
            for i, s in enumerate(scale):
                u.sub(i).interpolate(lambda x, s=s: s * f(x))
        else:
            u.interpolate(f)
        u.x.scatter_forward()
        return U.restrict(u, U.create_vector())

    def zero(self, space):
        return space[1].create_vector()


def run(backend, out, h5dir, read_from=None):
    from feluga.nested_transfer import Nested, refine_uniformly
    B = Dolfin() if backend == "dolfin" else Dolfinx()
    comm = B.comm
    res = {}
    for name, (d, n, passes) in CASES.items():
        coarse = B.mesh(d, n)
        fine, anc = refine_uniformly(coarse, passes, [1.0] * d, [0])
        nest = Nested(coarse, fine, anc, passes)
        Vc, Qc = B.space(coarse, 2), B.space(coarse, 1)
        xc = B.interp(Vc)
        r = {}
        for q in (1, 2):
            Vf = B.space(fine, q)
            P = nest.prolongation(Vc, Vf)
            y = P.createVecLeft()
            P.mult(xc, y)
            y.axpy(-1.0, B.interp(Vf))
            r["P_fine_p{}".format(q)] = y.norm()
            if q == 2:
                R = nest.injection(Vf, Vc)
                r["blocks"] = _blockwise_differs(B, nest, Vc, Vf, P, R)
                z = R.createVecLeft()
                R.mult(_mult(P, xc), z)
                z.axpy(-1.0, xc)
                r["RP"] = z.norm()
            Wf = B.space(fine, q, mixed=True)
            xw = B.interp(Wf, mixed=True)
            for comp, Sc, scale in ((0, Vc, 1.0), (1, Qc, 2.0)):
                R = nest.injection(Wf, Sc, fine_component=comp)
                z = R.createVecLeft()
                R.mult(xw, z)
                ref = B.interp(Sc)
                ref.scale(scale)
                z.axpy(-1.0, ref)
                r["R_w{}_c{}".format(q, comp)] = z.norm()
        # a fine mixed P2 field by its lattice
        Wf = B.space(fine, 2, mixed=True)
        lat = nest.lattice(Wf)
        path = os.path.join(h5dir, "{}_{}.h5".format(backend, comm.size))
        lat.write(path, name, B.interp(Wf, mixed=True), mode="a" if name != "rect1" else "w")
        if read_from:
            x = B.zero(Wf)
            lat.read(read_from, name, x)
            x.axpy(-1.0, B.interp(Wf, mixed=True))
            r["read"] = x.norm()
        res[name] = r
    if comm.rank == 0:
        with open(out, "w") as fh:
            json.dump(res, fh)


def _csr(A):
    return [np.asarray(a).copy() for a in A.getValuesCSR()]


def _blockwise_differs(B, nest, Vc, Vf, P, R):
    """1.0 when P, R or the lattice of Vf differ when built from blocks of 5 fine cells,
    on any rank; else 0.0."""
    from feluga import hierarchical_transfer
    chunk = hierarchical_transfer.EVAL_CHUNK
    lat = nest.lattice(Vf)
    hierarchical_transfer.EVAL_CHUNK = 5
    try:
        pairs = [(_csr(P), _csr(nest.prolongation(Vc, Vf))),
                 (_csr(R), _csr(nest.injection(Vf, Vc)))]
        small = nest.lattice(Vf)
    finally:
        hierarchical_transfer.EVAL_CHUNK = chunk
    pairs.append(([lat.cell_of, lat.slot, lat.dof], [small.cell_of, small.slot, small.dof]))
    differs = any(a.shape != b.shape or not np.array_equal(a, b)
                  for one, other in pairs for a, b in zip(one, other))
    return float(B.comm.allreduce(int(differs)))


def _mult(A, x):
    y = A.createVecLeft()
    A.mult(x, y)
    return y


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
