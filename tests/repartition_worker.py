"""Run under `mpirun -n 2` by tests/test_repartition.py; prints one RESULT line per check.

A unit cube, periodic on all three axes, rebuilt (`rebuild`) with most of its cells on
rank 0, then rebalanced: `repartition_plan(...).build()` must bring the busiest rank
within ParMETIS's tolerance, keep every cell and vertex with its global id, give a
correctly paired periodic P2 space, and `carry_by_id` must bring arrays along (ids and
coordinates, bit for bit, and a bool)."""
import dolfin as df
import numpy as np
from mpi4py import MPI

from femtools import periodic_partition as pp
from femtools.periodic import periodic_box

comm = MPI.COMM_WORLD
L = np.array([1.0, 1.0, 1.0])
periodic = (True, True, True)


def result(name, value):
    if comm.rank == 0:
        print("RESULT", name, value, flush=True)


def counts(mesh):
    tdim = mesh.topology().dim()
    return mesh.num_entities_global(tdim), mesh.num_entities_global(0)


def ids(mesh, dim):
    own = pp._owned(mesh) if dim == mesh.topology().dim() else None
    g = np.asarray(mesh.topology().global_indices(dim), dtype=np.int64)[:own]
    return np.unique(np.concatenate(comm.allgather(g)))


mesh = df.UnitCubeMesh(comm, 6, 6, 6)
mid = np.array([c.midpoint().array() for c in df.cells(mesh)])
# rank 1 keeps only the cells with x above 0.8: rank 0 holds about 80 %
skewed = pp.rebuild(mesh, np.where(mid[:, 0] > 0.8, 1, 0).astype(np.int32))
own = pp._owned(skewed)
total = comm.allreduce(own)
result("skewed", comm.allreduce(own, op=MPI.MAX) > pp.IMBALANCE * total / comm.size)

tdim = skewed.topology().dim()
cell_ids = np.asarray(skewed.topology().global_indices(tdim), dtype=np.int64)
cell_mid = np.array([c.midpoint().array() for c in df.cells(skewed)])
vert_ids = np.asarray(skewed.topology().global_indices(0), dtype=np.int64)
vert_x = skewed.coordinates().copy()
flag = vert_x[:, 2] > 0.5

plan = pp.repartition_plan(skewed, L, periodic, comm, log=lambda *a: None)
result("planned", plan is not None)
new = plan.build()
n = pp._owned(new)
result("balanced", comm.allreduce(n, op=MPI.MAX) <= pp.IMBALANCE * total / comm.size + 1)
result("counts_kept", counts(new) == counts(skewed))
result("cell_ids_kept", np.array_equal(ids(new, tdim), ids(skewed, tdim)))
result("vertex_ids_kept", np.array_equal(ids(new, 0), ids(skewed, 0)))

(c_ids, c_mid), (v_ids, v_x, v_flag) = pp.carry_by_id(skewed, new, cells=(cell_ids, cell_mid),
                                                      vertices=(vert_ids, vert_x, flag))
new_mid = np.array([c.midpoint().array() for c in df.cells(new)])
ok = (np.array_equal(c_ids, np.asarray(new.topology().global_indices(tdim), dtype=np.int64))
      and np.allclose(c_mid, new_mid, rtol=0, atol=1e-14)
      and np.array_equal(v_ids, np.asarray(new.topology().global_indices(0), dtype=np.int64))
      and np.array_equal(v_x, new.coordinates())
      and v_flag.dtype == bool and np.array_equal(v_flag, new.coordinates()[:, 2] > 0.5))
result("carried", comm.allreduce(bool(ok), op=MPI.LAND))

# the plan's own carry (the mesh it was decided on may be gone) agrees
(c2,), _ = plan.carry(new, cells=(cell_ids,))
result("plan_carry", comm.allreduce(bool(np.array_equal(c2, c_ids)), op=MPI.LAND))

V = df.VectorFunctionSpace(new, "Lagrange", 2,
                           constrained_domain=periodic_box(L, 1e-8, periodic, comm=comm))
try:
    pp.check_space_pairing(V, L, periodic, comm)
    paired = True
except RuntimeError:
    paired = False
result("paired", paired)
# already balanced: nothing to plan
result("again_none", pp.repartition_plan(new, L, periodic, comm, log=lambda *a: None) is None)
