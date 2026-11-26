"""Periodic box pairing on DOLFINx spaces, without dolfinx_mpc.

The DOLFINx counterpart of `feluga.periodic.periodic_box`, with the same rule: the box
runs from `lower` to `upper`, `periodic` says which axes are identified; a dof on an
upper periodic face is a slave, and its image takes every coordinate that sits on an
upper periodic face down by the box length, so an edge or corner dof maps to the lower
corner in one step. The dof at the image is its master; masters are never on an upper
periodic face, so there are no chains.

Matching is by coordinates, in parallel, without gathering a face anywhere: the owned
candidate masters (dofs on a lower periodic face and on no upper one) and the local
(owned and ghost) slaves' images meet by position (`feluga.hdf5_rows.match_nodes`), and
each slave hears its master's global index back. `tol` is relative to the box
(`feluga.periodic_core.box_scale`).

    slaves, masters = periodic_box(V, upper=(Lx, Ly), periodic=(True, False))

`slaves` are local dof indices (owned and ghost, unrolled for blocked spaces), `masters`
the global dof index of each one's master. Scalar, blocked and mixed Lagrange spaces
(Taylor-Hood's too).
"""
import numpy as np


def box_extent(mesh):
    """The mesh's bounding box over all ranks, as two 3-vectors."""
    from feluga.hdf5_rows import extent
    return extent(mesh.geometry.x, mesh.comm, 3)


def flat_collapse_map(dofs):
    """A collapse dof map as a flat int32 array: DOLFINx 0.11 returns a list with one
    array per cell type, 0.10 a flat sequence."""
    if isinstance(dofs, (list, tuple)) and len(dofs) > 0 and not np.isscalar(dofs[0]):
        if len(dofs) != 1:
            raise NotImplementedError("collapse map over several cell types")
        dofs = dofs[0]
    return np.asarray(dofs, dtype=np.int32)


def _pair(comm, x, n_owned, l2g, lower, upper, periodic, tol):
    """Slave local indices and master global indices for one set of coordinates `x`
    (owned first, then ghosts) with global indices `l2g`; `tol` a length. The owned
    candidate masters are the nodes and the slaves' images the points of
    `hdf5_rows.match_nodes`. Raises on every rank alike when some slave has no master.
    Collective."""
    from feluga.hdf5_rows import allsum, match_nodes

    from feluga.periodic_core import classify
    on_lo, on_hi, image = classify(x, lower, upper, periodic, tol)
    slaves = np.flatnonzero(on_hi)
    cand = np.flatnonzero(on_lo & ~on_hi)
    cand = cand[cand < n_owned]  # each master contributed once, by its owner
    found, distance = match_nodes(comm, x[cand], l2g[cand], image[slaves], tol, lower)
    master = np.where(distance < tol, found[:, 0], -1)
    bad = master < 0
    n_bad = int(allsum(comm, int(bad.sum())))
    if n_bad:
        first = image[slaves][bad][0] if bad.any() else None
        firsts = [f for f in comm.allgather(first) if f is not None]
        raise RuntimeError(
            f"periodic_box: {n_bad} slave dofs have no master within {tol:g} "
            f"(first image {firsts[0]}); are the periodic faces congruent?")
    return slaves.astype(np.int32), master.astype(np.int64)


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
    (`feluga.periodic_core.box_scale`).
    Returns (slaves, masters): local dof indices (owned and ghost, unrolled) and the
    global dof index of each one's master.
    """
    from feluga.periodic_core import box_scale, xyz
    mesh = V.mesh
    comm = mesh.comm
    gdim = mesh.geometry.dim
    lo_m, hi_m = box_extent(mesh)
    lower = lo_m if lower is None else xyz(lower)
    upper = hi_m if upper is None else xyz(upper)
    if periodic is None:
        periodic = [True] * gdim
    periodic = np.concatenate([np.asarray(periodic, bool), np.zeros(3 - len(periodic), bool)])
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
