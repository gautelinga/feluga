"""Periodic box pairing on DOLFINx spaces, without dolfinx_mpc.

The DOLFINx counterpart of `femtools.periodic.periodic_box`, with the same rule: the box
runs from `lower` to `upper`, `periodic` says which axes are identified; a dof on an
upper periodic face is a slave, and its image takes every coordinate that sits on an
upper periodic face down by the box length, so an edge or corner dof maps to the lower
corner in one step. The dof at the image is its master; masters are never on an upper
periodic face, so there are no chains.

Matching is by coordinates, in parallel: each rank contributes its owned candidate
masters (dofs on a lower periodic face and on no upper one) to one allgather, and every
rank looks up its local (owned and ghost) slaves in a KD-tree of all candidates. The
gathered set is one face's worth of dofs.

    slaves, masters = periodic_box(V, upper=(Lx, Ly), periodic=(True, False))

`slaves` are local dof indices (owned and ghost, unrolled for blocked spaces), `masters`
the global dof index of each one's master. Scalar, blocked and mixed Lagrange spaces.
"""
import numpy as np


def box_extent(mesh):
    """The mesh's bounding box over all ranks, as two 3-vectors."""
    from mpi4py import MPI

    x = mesh.geometry.x
    lo_loc = x.min(axis=0) if len(x) else np.full(3, np.inf)
    hi_loc = x.max(axis=0) if len(x) else np.full(3, -np.inf)
    lo, hi = np.empty(3), np.empty(3)
    # Buffer Allreduce: the pickle-based allreduce compares whole arrays with MIN/MAX.
    mesh.comm.Allreduce(np.ascontiguousarray(lo_loc, dtype=float), lo, op=MPI.MIN)
    mesh.comm.Allreduce(np.ascontiguousarray(hi_loc, dtype=float), hi, op=MPI.MAX)
    return lo, hi


def flat_collapse_map(dofs):
    """A collapse dof map as a flat int32 array: DOLFINx 0.11 returns a list with one
    array per cell type, 0.10 a flat sequence."""
    if isinstance(dofs, (list, tuple)) and len(dofs) > 0 and not np.isscalar(dofs[0]):
        if len(dofs) != 1:
            raise NotImplementedError("collapse map over several cell types")
        dofs = dofs[0]
    return np.asarray(dofs, dtype=np.int32)


def _pad3(v, fill):
    v = np.asarray(v, dtype=float)
    return np.concatenate([v, np.full(3 - len(v), fill)])


def _pair(comm, x, n_owned, l2g, lower, upper, periodic, tol):
    """Slave local indices and master global indices for one set of coordinates `x`
    (owned first, then ghosts) with global indices `l2g`."""
    from scipy.spatial import cKDTree

    axes = np.flatnonzero(periodic)
    on_hi = np.zeros(len(x), dtype=bool)
    on_lo = np.zeros(len(x), dtype=bool)
    image = x.copy()
    for a in axes:
        hi_a = np.abs(x[:, a] - upper[a]) < tol
        on_hi |= hi_a
        on_lo |= np.abs(x[:, a] - lower[a]) < tol
        image[hi_a, a] -= upper[a] - lower[a]
    slaves = np.flatnonzero(on_hi)

    cand = np.flatnonzero(on_lo & ~on_hi)
    cand = cand[cand < n_owned]  # each master contributed once, by its owner
    pts = np.concatenate(comm.allgather(x[cand])) if comm.size > 1 else x[cand]
    gids = np.concatenate(comm.allgather(l2g[cand])) if comm.size > 1 else l2g[cand]

    if len(slaves) == 0:
        return slaves.astype(np.int32), np.zeros(0, dtype=np.int64)
    if len(pts) == 0:
        raise RuntimeError("periodic_box: slaves found but no candidate masters")
    dist, idx = cKDTree(pts).query(image[slaves], distance_upper_bound=tol)
    bad = ~np.isfinite(dist)
    if bad.any():
        raise RuntimeError(
            f"periodic_box: {bad.sum()} slave dofs have no master within {tol:g} "
            f"(first image {image[slaves][bad][0]}); are the periodic faces congruent?")
    return slaves.astype(np.int32), gids[idx].astype(np.int64)


def periodic_box(V, upper=None, periodic=None, lower=None, tol=1e-8):
    """Slave dofs of `V` and their masters on a periodic box.

    `upper`, `lower`: the box corners (default the mesh's bounding box); `periodic`:
    which axes are identified (default all geometric axes); `tol`: how near a face a
    dof must be, and how near its master, relative to the box size.
    Returns (slaves, masters): local dof indices (owned and ghost, unrolled) and the
    global dof index of each one's master.
    """
    mesh = V.mesh
    comm = mesh.comm
    gdim = mesh.geometry.dim
    lo_m, hi_m = box_extent(mesh)
    lower = lo_m if lower is None else _pad3(lower, 0.0)
    upper = hi_m if upper is None else _pad3(upper, 0.0)
    if periodic is None:
        periodic = [True] * gdim
    periodic = np.concatenate([np.asarray(periodic, bool), np.zeros(3 - len(periodic), bool)])
    atol = tol * float(np.max(upper - lower))

    imap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    n_loc = imap.size_local + imap.num_ghosts
    num_sub = V.element.num_sub_elements
    if num_sub > 0 and bs == 1:  # mixed space: pair each collapsed component
        l2g = imap.local_to_global(np.arange(n_loc, dtype=np.int32)).astype(np.int64)
        S, M = [], []
        for i in range(num_sub):
            Vi, parent = V.sub(i).collapse()
            parent = flat_collapse_map(parent)
            owned = parent < imap.size_local
            order = np.argsort(~owned, kind="stable")  # owned first
            s, m = _pair(comm, Vi.tabulate_dof_coordinates()[order], int(owned.sum()),
                         l2g[parent[order]], lower, upper, periodic, atol)
            S.append(parent[order][s])
            M.append(m)
        return np.concatenate(S).astype(np.int32), np.concatenate(M)

    l2g_blk = imap.local_to_global(np.arange(n_loc, dtype=np.int32)).astype(np.int64)
    s_blk, m_blk = _pair(comm, V.tabulate_dof_coordinates(), imap.size_local, l2g_blk,
                         lower, upper, periodic, atol)
    comp = np.arange(bs)
    return ((s_blk[:, None] * bs + comp).ravel().astype(np.int32),
            (m_blk[:, None] * bs + comp).ravel().astype(np.int64))
