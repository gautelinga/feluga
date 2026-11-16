"""tests/test_dolfinx_sync.py's worker: a backend Function's two layouts kept consistent
(femtools.backend.dolfinx), whatever writes them; one JSON line a rank."""
import json

from femtools.backend.dolfinx import (MPI, FiniteElement, Function, FunctionSpace,
                                      MixedElement, PeriodicBox, UnitSquareMesh, assemble,
                                      dx, split)

out = {}
rank = MPI.comm_world.rank
# a write to the compact vector on one rank only: the next assembly is collective all
# the same (the prolongation a scatter every rank must take part in)
mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
u.vector()[:] = 1.0
assemble(u * dx)
if rank == 0:
    u.vector()[0] = 1.0
out["one_rank_write"] = assemble(u * dx)
# a partial write through DOLFINx's array after the compact vector changed: it starts from
# the vector's values, not the array's old ones
w = Function(FunctionSpace(UnitSquareMesh(4, 4), "CG", 1))
assemble(w * dx)
w.vector()[:] = 3.0
w.x.array[:2] = 3.0
w.x.scatter_forward()
out["partial_array_write"] = assemble(w * dx)
# one component interpolated: the other keeps its values (ghosts included)
m4 = UnitSquareMesh(4, 4)
W = FunctionSpace(m4, MixedElement([FiniteElement("CG", m4.ufl_cell(), 1)] * 2))
q = Function(W)
q1 = q.sub(1)
q.vector()[:] = 5.0
q1.interpolate(lambda x: 4 + 0 * x[0])
a, b = split(q)
out["component_kept"], out["component_set"] = assemble(a * dx), assemble(b * dx)
# a periodic space: interpolating a non-periodic field leaves the slaves their own values;
# assembly sees the compact vector's (the masters') before and after a no-op write
P = FunctionSpace(UnitSquareMesh(8, 8), "CG", 1, constrained_domain=PeriodicBox({0: (0, 1)}))
p = Function(P)
p.interpolate(lambda x: x[0])
first = assemble(p * dx)
p.vector()[:] = p.vector()[:]
out["periodic_same"] = first == assemble(p * dx)


# a subclass's own sync runs before each assembly: here a copy that follows another
# Function, as a solver's copy of a field on another mesh may
class Follower(Function):
    def __init__(self, V, source):
        Function.__init__(self, V)
        object.__setattr__(self, "source", source)

    def sync(self):
        self.vector().zero()
        self.vector().axpy(1.0, self.source.vector())
        return Function.sync(self)


source = Function(V)
follower = Follower(V, source)
source.vector()[:] = 2.0
assemble(follower * dx)
source.vector()[:] = 7.0
out["follower"] = assemble(follower * dx)
print("RESULT " + json.dumps(out), flush=True)
