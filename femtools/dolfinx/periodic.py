"""Periodic box pairing on DOLFINx spaces, without dolfinx_mpc.

The DOLFINx counterpart of `femtools.periodic.periodic_box`, with the same rule: the box
runs from `lower` to `upper`, `periodic` says which axes are identified; a dof on an
upper periodic face is a slave, and its image takes every coordinate that sits on an
upper periodic face down by the box length, so an edge or corner dof maps to the lower
corner in one step. The dof at the image is its master; masters are never on an upper
periodic face, so there are no chains.

Matching is by coordinates, in parallel, without gathering a face anywhere: each owned
candidate master (a dof on a lower periodic face and on no upper one) goes to the rank
its position's bin hashes to, each local (owned and ghost) slave to the ranks of the
bins a master within tol of its image may be in; they meet there in a KD-tree, and each
slave hears its master's global index back. `tol` is relative to the box
(`femtools.periodic_core.box_scale`).

    slaves, masters = periodic_box(V, upper=(Lx, Ly), periodic=(True, False))

`slaves` are local dof indices (owned and ghost, unrolled for blocked spaces), `masters`
the global dof index of each one's master. Scalar, blocked and mixed Lagrange spaces
(Taylor-Hood's too).
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
    (owned first, then ghosts) with global indices `l2g`; `tol` a length. Raises on
    every rank alike when some slave has no master. Collective."""
    from scipy.spatial import cKDTree

    from femtools.hdf5_rows import allsum, rank_of, reply, route

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

    # bins much wider than tol: a master goes to its own bin's rank, a slave to those of
    # its image's bin and, where within tol of a bin's edge, the neighbour's
    h = 1e3 * tol
    size = comm.Get_size()
    sent = route(comm, rank_of(np.floor(x[cand] / h).astype(np.int64), size),
                 x[cand], l2g[cand])
    pts, gids = sent.received
    pts = pts.reshape(-1, 3)
    y = image[slaves] / h
    keys, of = _near_bins(y, 2 * tol / h)
    asked = route(comm, rank_of(keys, size), image[slaves][of])
    (q,) = asked.received
    found = np.full(len(q), -1, np.int64)
    if len(q) and len(pts):
        dist, idx = cKDTree(pts).query(q.reshape(-1, 3), distance_upper_bound=tol)
        ok = np.isfinite(dist)
        found[ok] = gids[idx[ok]]
    (found,) = reply(comm, asked, found)
    master = np.full(len(slaves), -1, np.int64)
    np.maximum.at(master, of, found)
    bad = master < 0
    n_bad = int(allsum(comm, int(bad.sum())))
    if n_bad:
        first = image[slaves][bad][0] if bad.any() else None
        firsts = [f for f in comm.allgather(first) if f is not None]
        raise RuntimeError(
            f"periodic_box: {n_bad} slave dofs have no master within {tol:g} "
            f"(first image {firsts[0]}); are the periodic faces congruent?")
    return slaves.astype(np.int32), master


def _near_bins(y, near):
    """`(keys, index)`: each row's bin (`floor(y)`) and, in each coordinate within `near`
    of a bin's edge, the neighbour's too -- every bin a point within `near` of it has."""
    b = np.floor(y).astype(np.int64)
    frac = y - b
    keys, index = b, np.arange(len(y))
    for d in range(3):
        for step, m in ((-1, frac[:, d] <= near), (1, frac[:, d] >= 1 - near)):
            mm = m[index]
            extra = keys[mm].copy()
            extra[:, d] += step
            keys, index = np.vstack([keys, extra]), np.r_[index, index[mm]]
    return keys, index


def _scalar_leaves(V):
    """V's scalar sub-spaces, depth first: a mixed space's components, and a vector
    (blocked) component's own components."""
    n = V.element.num_sub_elements
    if n == 0:
        return [V]
    return [leaf for i in range(n) for leaf in _scalar_leaves(V.sub(i))]


def periodic_box(V, upper=None, periodic=None, lower=None, tol=1e-8):
    """Slave dofs of `V` and their masters on a periodic box.

    `upper`, `lower`: the box corners (default the mesh's bounding box); `periodic`:
    which axes are identified (default all geometric axes); `tol`: how near a face a
    dof must be, and how near its master, relative to the box
    (`femtools.periodic_core.box_scale`).
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
    from femtools.periodic_core import box_scale
    atol = tol * box_scale(lower, upper)

    imap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    n_loc = imap.size_local + imap.num_ghosts
    num_sub = V.element.num_sub_elements
    if num_sub > 0 and bs == 1:
        # a mixed space: pair each scalar leaf, collapsed (a vector sub-space, as
        # Taylor-Hood's velocity, by its components: its collapse is blocked)
        l2g = imap.local_to_global(np.arange(n_loc, dtype=np.int32)).astype(np.int64)
        S, M = [], []
        for leaf in _scalar_leaves(V):
            Vi, parent = leaf.collapse()
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
