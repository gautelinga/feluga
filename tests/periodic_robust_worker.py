"""Worker for tests/test_periodic_robust.py.

    [mpirun -n k] python3 tests/periodic_robust_worker.py MODE out.json

MODE:
  far_dolfinx, far_dolfin  a box at x = 1e6 whose upper face's nodes lie one ulp off:
                           the pairs, and whether the refined faces are congruent
  ghost                    DOLFINx markers from locate_entities (ghosts included) refine
                           as the owned ones do; indices outside the local cells raise
                           on every rank
  jit                      a build failing on rank 0 raises on every rank, the generic
                           helper's and the DOLFINx extension's (FEMTOOLS_JIT_DIR fresh)
  pairing                  DOLFINx's periodic_box on faces that do not match raises on
                           every rank; on a periodic unit square every slave pairs
  taylor_hood              periodic_box on a Taylor-Hood space: every slave's master is
                           the dof of the same component at its image
  zero_cells_dolfin        plaza_periodic on dolfin with ranks holding no cells raises on
                           every rank (dolfin's refinement hangs there)
  zero_cells_dolfinx       femtools.dolfinx.refine there: refines, pairs, congruent
Each rank writes its part; rank 0 merges them into out.json.
"""
import json
import sys

import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD


def _raised(f):
    try:
        f()
    except Exception as e:
        return type(e).__name__
    return None


def far_dolfinx():
    import dolfinx.mesh as dm
    from femtools.dolfinx.refine import congruent, refine_periodic
    x0 = 1e6
    lo, hi = [x0, 0.0], [x0 + 0.3, 0.7]
    mesh = dm.create_rectangle(COMM, [lo, hi], [3, 7], dm.CellType.triangle)
    x = mesh.geometry.x
    up = np.isclose(x[:, 0], hi[0], rtol=0, atol=1e-6)
    x[up, 0] = np.nextafter(x[up, 0], np.inf)      # one ulp off, as a generator may leave it
    cells = np.flatnonzero(x[mesh.geometry.dofmaps[0]][:, :, 0].mean(1) > x0 + 0.25)
    new, npair, _ = refine_periodic(mesh, cells.astype(np.int32), hi, [0], lower=lo,
                                    redistribute=False)
    return {"npair": npair, "congruent": bool(congruent(new, hi, [0], lower=lo))}


def far_dolfin():
    import dolfin as df
    from femtools.periodic import faces_congruent
    from femtools.plaza_periodic import refine_periodic
    x0 = 1e6
    lo, hi = [x0, 0.0], [x0 + 0.3, 0.7]
    mesh = df.RectangleMesh(COMM, df.Point(*lo), df.Point(*hi), 3, 7)
    x = mesh.coordinates()
    up = np.isclose(x[:, 0], hi[0], rtol=0, atol=1e-6)
    x[up, 0] = np.nextafter(x[up, 0], np.inf)
    markers = df.MeshFunction("bool", mesh, 2, False)
    for c in df.cells(mesh):
        markers[c] = c.midpoint().x() > x0 + 0.25
    new, npair, _ = refine_periodic(mesh, markers, hi, [0], lower=lo, redistribute=False)
    return {"npair": npair,
            "congruent": bool(faces_congruent(new, hi, [0], edges=True, lower=lo)[0])}


def ghost():
    import dolfinx.mesh as dm
    from femtools.dolfinx.refine import refine_periodic
    mesh = dm.create_unit_square(COMM, 8, 8, ghost_mode=dm.GhostMode.shared_facet)
    im = mesh.topology.index_map(2)
    cells = dm.locate_entities(mesh, 2, lambda x: x[0] > 0.8)
    had_ghosts = bool(COMM.allreduce(int(np.any(cells >= im.size_local))))
    a = refine_periodic(mesh, cells, [1, 1], [0], redistribute=False)[0]
    b = refine_periodic(mesh, cells[cells < im.size_local], [1, 1], [0],
                        redistribute=False)[0]
    bad = np.array([im.size_local + im.num_ghosts + 5], np.int32) if COMM.rank == 0 \
        else np.zeros(0, np.int32)
    mesh.topology.create_entities(1)
    ie = mesh.topology.index_map(1)
    bad_edge = np.array([ie.size_local + ie.num_ghosts], np.int64) if COMM.rank == 1 \
        else np.zeros(0, np.int64)
    return {"had_ghosts": had_ghosts,
            "with_ghosts": int(a.topology.index_map(2).size_global),
            "owned_only": int(b.topology.index_map(2).size_global),
            "out_of_range": _raised(lambda: refine_periodic(mesh, bad, [1, 1], [0])),
            "frozen_out_of_range": _raised(lambda: refine_periodic(
                mesh, None, [1, 1], [0], frozen=bad_edge))}


def jit():
    import subprocess

    import femtools.dolfinx.refine as refine
    from femtools.jit import on_rank0

    def build():
        raise FileNotFoundError("no compiler here")

    generic = _raised(lambda: on_rank0(COMM, build, "test"))

    def missing(*args, **kwargs):
        raise FileNotFoundError("cmake")      # what subprocess.run raises without cmake
    subprocess.run = missing
    return {"generic": generic, "refine": _raised(lambda: refine._module(COMM))}


def pairing():
    import dolfinx.fem as fem
    import dolfinx.mesh as dm
    from femtools.dolfinx.periodic import periodic_box
    mesh = dm.create_unit_square(COMM, 6, 6)
    V = fem.functionspace(mesh, ("Lagrange", 2))
    slaves, masters = periodic_box(V, upper=(1, 1), periodic=(True, True))
    n_slaves = COMM.allreduce(len(slaves))
    # a box whose lower y face is no row of nodes: the slaves on y = 1 find no master
    wrong = _raised(lambda: periodic_box(V, upper=(1, 1), lower=(0, 0.05),
                                         periodic=(False, True)))
    return {"n_slaves": int(n_slaves), "wrong": wrong,
            "masters_known": bool(COMM.allreduce(int(np.all(masters >= 0)), op=MPI.MIN))}


def taylor_hood():
    import basix.ufl
    import dolfinx.fem as fem
    import dolfinx.mesh as dm
    from femtools.dolfinx.periodic import flat_collapse_map, periodic_box

    def leaves(V):
        n = V.element.num_sub_elements
        return [V] if n == 0 else [w for i in range(n) for w in leaves(V.sub(i))]
    mesh = dm.create_unit_square(COMM, 5, 5)
    P2 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(2,))
    P1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    W = fem.functionspace(mesh, basix.ufl.mixed_element([P2, P1]))
    slaves, masters = periodic_box(W, upper=(1, 1), periodic=(True, True))
    imap = W.dofmap.index_map
    n_loc = imap.size_local + imap.num_ghosts
    l2g = imap.local_to_global(np.arange(n_loc, dtype=np.int32)).astype(np.int64)
    # every local dof's position and leaf, by global index, over all ranks
    where, leaf_of = {}, {}
    for k, leaf in enumerate(leaves(W)):
        Vi, parent = leaf.collapse()
        parent = flat_collapse_map(parent)
        X = Vi.tabulate_dof_coordinates()
        for j, p in enumerate(parent):
            where[int(l2g[p])] = X[j]
            leaf_of[int(l2g[p])] = k
    allw, alll = {}, {}
    for w, lf in zip(COMM.allgather(where), COMM.allgather(leaf_of)):
        allw.update(w)
        alll.update(lf)
    worst, same_leaf = 0.0, True
    for s_, m in zip(slaves, masters):
        g = int(l2g[s_])
        image = np.array(allw[g])
        image[:2] = np.where(np.isclose(image[:2], 1.0), 0.0, image[:2])
        worst = max(worst, float(np.abs(image - allw[int(m)]).max()))
        same_leaf &= alll[g] == alll[int(m)]
    n_slaves = COMM.allreduce(len(slaves))
    # P2 x 2 and P1 nodes on the upper faces: 2*(2*10+1) + (2*5+1)
    return {"n_slaves_owned_or_ghost": int(n_slaves),
            "worst": COMM.allreduce(worst, op=MPI.MAX),
            "same_leaf": bool(COMM.allreduce(int(same_leaf), op=MPI.MIN)),
            "n_unique": len({int(l2g[s_]) for s_ in slaves})}


def zero_cells_dolfin():
    import dolfin as df
    from femtools.plaza_periodic import refine_periodic
    from femtools.refine_periodic import refine_congruent
    mesh = df.UnitSquareMesh(COMM, 1, 1)
    m = df.MeshFunction("bool", mesh, 2, True)
    return {"cells": mesh.num_cells(),
            "periodic": _raised(lambda: refine_periodic(mesh, m, [1.0, 1.0], [0, 1])),
            "plain": _raised(lambda: refine_congruent(mesh, m, [1.0, 1.0], []))}


def zero_cells_dolfinx():
    import dolfinx.mesh as dm
    from femtools.dolfinx.refine import congruent, refine_congruent, refine_periodic
    mesh = dm.create_rectangle(COMM, [[0, 0], [1, 1]], [1, 1], dm.CellType.triangle)
    out = {"cells": int(mesh.topology.index_map(2).size_local)}
    new, npair, parent = refine_periodic(mesh, None, [1.0, 1.0], [0, 1], redistribute=False)
    out["periodic"] = [int(new.topology.index_map(2).size_global), npair, len(parent)]
    out["congruent"] = bool(congruent(new, [1.0, 1.0], [0, 1]))
    new, _, _ = refine_congruent(mesh, None, [1.0, 1.0], [0, 1], max_cells=100,
                                 frozen=np.zeros(0, np.int32))
    out["capped"] = int(new.topology.index_map(2).size_global)
    new, _, _ = refine_periodic(mesh, None, [1.0, 1.0], [], redistribute=True)
    out["plain"] = int(new.topology.index_map(2).size_global)
    return out


def main():
    mode, out = sys.argv[1], sys.argv[2]
    result = globals()[mode]()
    gathered = COMM.gather(result, root=0)
    if COMM.rank == 0:
        with open(out, "w") as f:
            json.dump(gathered, f)


if __name__ == "__main__":
    main()
