"""Worker for tests/test_dolfinx_refine.py: femtools.dolfinx.refine.refine_periodic.

    [mpirun -n k] python3 tests/dolfinx_refine_worker.py dolfinx out.json
    python3 tests/dolfinx_refine_worker.py dolfin out.json     # plaza_periodic, serial

For a rectangle (crossed diagonals) and a box, refined near the upper and near the lower x
face, it reports: the P2 nodes on the two periodic x faces (equal when congruent), whether
every slave dof finds its master, the cells made, and the sorted vertex positions (for the
comparison with dolfin's plaza_periodic, which numbers the same input alike); plain
DOLFINx refinement of the same marking for contrast; uniform refinement against
dolfinx.mesh.refine; frozen edges; max_cells; the parent map.
"""
import json
import sys

import numpy as np

CASES = {"rect": (2, 12), "right": (2, 12), "box": (3, 6)}


def signature(x, cells, comm):
    """Every cell as its sorted vertex positions, sorted, over all ranks: the same for the
    same cells however numbered and distributed."""
    import numpy as np
    sig = np.round(x[cells], 9)                       # (n, k, dim)
    sig = [sorted(map(tuple, c)) for c in sig.tolist()]
    sig = sorted(sum(comm.allgather(sig), []))
    return sig


def marked(mid, side):
    return mid[:, 0] > 0.8 if side == "upper" else mid[:, 0] < 0.2


def run_dolfinx(out):
    from mpi4py import MPI
    import dolfinx.mesh as dm
    from dolfinx import fem
    from femtools.dolfinx.periodic import periodic_box
    from femtools.dolfinx.refine import refine_periodic, congruent, TooManyCells
    comm = MPI.COMM_WORLD

    def make(dim, n, name=None):
        if dim == 2:
            diagonal = dm.DiagonalType.right if name == "right" else dm.DiagonalType.crossed
            return dm.create_rectangle(comm, [[0, 0], [1, 1]], [n, n], diagonal=diagonal)
        return dm.create_box(comm, [[0, 0, 0], [1, 1, 1]], [n, n, n])

    def faces(mesh):
        V = fem.functionspace(mesh, ("Lagrange", 2))
        X = V.tabulate_dof_coordinates()[:V.dofmap.index_map.size_local]
        lo = comm.allreduce(int(np.isclose(X[:, 0], 0.0).sum()))
        hi = comm.allreduce(int(np.isclose(X[:, 0], 1.0).sum()))
        try:
            periodic_box(V, periodic=(True,) + (False,) * (mesh.geometry.dim - 1))
            paired = True
        except RuntimeError:
            paired = False
        return lo, hi, paired

    def simplices(mesh):
        tdim = mesh.topology.dim
        n = mesh.topology.index_map(tdim).size_local
        nodes = dm.entities_to_geometry(mesh, tdim, np.arange(n, dtype=np.int32))
        return signature(mesh.geometry.x[:, :mesh.geometry.dim], nodes, comm)

    def vertices(mesh):
        x = mesh.geometry.x[:mesh.geometry.index_map().size_local, :mesh.geometry.dim]
        x = np.round(np.concatenate(comm.allgather(x)), 10)
        return x[np.lexsort(x.T[::-1])].tolist()

    res = {}
    for name, (dim, n) in CASES.items():
        for side in ("upper", "lower"):
            mesh = make(dim, n, name)
            tdim = mesh.topology.dim
            n_own = mesh.topology.index_map(tdim).size_local
            mid = dm.compute_midpoints(mesh, tdim, np.arange(n_own, dtype=np.int32))
            cells = np.flatnonzero(marked(mid, side)).astype(np.int32)
            fine, npair, parent = refine_periodic(mesh, cells, (1,) * dim, [0],
                                                  redistribute=False)
            lo, hi, paired = faces(fine)
            key = "{}_{}".format(name, side)
            res[key] = dict(lo=lo, hi=hi, paired=paired, npair=npair,
                            cells=fine.topology.index_map(tdim).size_global,
                            congruent=congruent(fine, (1,) * dim, [0]))
            if comm.size == 1:
                res[key]["vertices"] = vertices(fine)
            res[key]["simplices"] = simplices(fine)
            # the parent map: every child's centroid in its parent cell
            nf = fine.topology.index_map(tdim).size_local
            child = dm.compute_midpoints(fine, tdim, np.arange(nf, dtype=np.int32))
            G = mesh.geometry.dofmaps[0]
            ok = True
            for c, p in zip(range(nf), parent):
                v = mesh.geometry.x[G[p]][:, :dim]
                T = (v[1:] - v[0]).T
                lam = np.linalg.solve(T, child[c, :dim] - v[0])
                ok &= bool(lam.min() > -1e-12 and lam.sum() < 1 + 1e-12)
            res[key]["parents"] = bool(comm.allreduce(int(not ok)) == 0) and len(parent) == nf
            # plain DOLFINx refinement of the same cells' edges, for contrast
            mesh.topology.create_connectivity(tdim, 1)
            edges = np.unique(dm.compute_incident_entities(mesh.topology, cells, tdim, 1))
            plain, _, _ = dm.refine(mesh, edges)
            res[key]["plain"] = list(faces(plain)[:2])
            res[key]["plain_congruent"] = congruent(plain, (1,) * dim, [0])
        # uniform: dolfinx.mesh.refine's cells
        mesh = make(dim, n, name)
        mesh.topology.create_entities(1)
        fine, _, _ = refine_periodic(mesh, None, (1,) * dim, list(range(dim)), redistribute=False)
        ref, _, _ = dm.refine(mesh)
        res[name + "_uniform"] = [fine.topology.index_map(dim).size_global,
                                  ref.topology.index_map(dim).size_global]
        res[name + "_uniform_simplices"] = simplices(fine)
        # frozen: the edges on the upper x face (and so their partners) are never bisected
        mesh = make(dim, n, name)
        mesh.topology.create_entities(1)
        face = dm.locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[0], 1.0))
        report = {}
        fine, _, _ = refine_periodic(mesh, None, (1,) * dim, [0], frozen=face, report=report)
        lo, hi, paired = faces(fine)
        V0 = fem.functionspace(mesh, ("Lagrange", 2))
        X0 = V0.tabulate_dof_coordinates()[:V0.dofmap.index_map.size_local]
        before = comm.allreduce(int(np.isclose(X0[:, 0], 1.0).sum()))
        res[name + "_frozen"] = dict(lo=lo, hi=hi, before=before, paired=paired,
                                     frozen_cells=report["frozen_cells"])
        # max_cells: refused, nothing made
        try:
            refine_periodic(make(dim, n), None, (1,) * dim, list(range(dim)), max_cells=10)
            res[name + "_capped"] = False
        except TooManyCells as e:
            res[name + "_capped"] = e.n_cells > 10
    if comm.rank == 0:
        with open(out, "w") as f:
            json.dump(res, f)


def run_dolfin(out):
    import dolfin as df
    from mpi4py import MPI
    from femtools.plaza_periodic import refine_periodic
    comm = MPI.COMM_WORLD

    def simplices(mesh):
        return signature(mesh.coordinates(), mesh.cells(), comm)

    def make(name, dim, n):
        if dim == 2:
            return df.RectangleMesh(df.Point(0, 0), df.Point(1, 1), n, n,
                                    "right" if name == "right" else "crossed")
        return df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 1), n, n, n)

    res = {}
    for name, (dim, n) in CASES.items():
        for side in ("upper", "lower"):
            mesh = make(name, dim, n)
            mid = np.array([c.midpoint().array()[:dim] for c in df.cells(mesh)]).reshape(-1, dim)
            markers = df.MeshFunction("bool", mesh, dim, False)
            markers.array()[:] = marked(mid, side)
            fine, npair, _ = refine_periodic(mesh, markers, (1,) * dim, [0], redistribute=False)
            key = "{}_{}".format(name, side)
            res[key] = dict(cells=int(df.MPI.sum(comm, fine.num_cells())), npair=npair,
                            simplices=simplices(fine))
            if comm.size == 1:
                x = np.round(fine.coordinates(), 10)
                res[key]["vertices"] = x[np.lexsort(x.T[::-1])].tolist()
        mesh = make(name, dim, n)
        markers = df.MeshFunction("bool", mesh, dim, True)
        fine, _, _ = refine_periodic(mesh, markers, (1,) * dim, list(range(dim)),
                                     redistribute=False)
        res[name + "_uniform_simplices"] = simplices(fine)
    if comm.rank == 0:
        with open(out, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    {"dolfinx": run_dolfinx, "dolfin": run_dolfin}[sys.argv[1]](sys.argv[2])
