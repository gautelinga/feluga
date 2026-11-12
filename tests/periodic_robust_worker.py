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


def main():
    mode, out = sys.argv[1], sys.argv[2]
    result = globals()[mode]()
    gathered = COMM.gather(result, root=0)
    if COMM.rank == 0:
        with open(out, "w") as f:
            json.dump(gathered, f)


if __name__ == "__main__":
    main()
