"""The backend-free half of femtools' periodic boxes and their refinement, shared by the
dolfin modules (`femtools.periodic`, `femtools.plaza_periodic`) and the DOLFINx one
(`femtools.dolfinx.refine`).

    faces_congruent_arrays(comm, x, e, L, axes, tol, exempt, lower)
    lower_corner(extent, axes, lower)      the box's lower corner (`periodic.box_lower`)
    on_planes(x, lower, upper, axes, tol)  which points lie on an identified plane
    xyz(a)                                 padded to three components
    TooManyCells                           a refinement refused by max_cells

Every face node goes by its rounded position to the rank its key hashes to, where the two
sides are deduplicated and compared; only counts are reduced, so no rank holds a face.
"""
import numpy as np


def xyz(a):
    """Coordinates, or a corner, padded with zeros to three components."""
    a = np.asarray(a, dtype=float)
    pad = [(0, 0)] * (a.ndim - 1) + [(0, 3 - a.shape[-1])]
    return np.pad(a, pad) if a.shape[-1] < 3 else a


def lower_corner(extent, axes, lower=None, tol=1e-9):
    """The periodic box's lower corner, three components: `lower`, or the origin when
    it is None -- refused when the mesh (`extent`: its bounding box over all ranks,
    as (lo, hi)) does not reach the origin along a periodic axis, since a box elsewhere
    then needs its lower corner given (the upper faces would otherwise be compared with
    planes inside the mesh)."""
    if lower is not None:
        return xyz(lower)
    lo, hi = extent
    off = [a for a in axes if abs(lo[a]) > tol * max(1.0, abs(hi[a] - lo[a]))]
    if off:
        raise ValueError(
            "the mesh starts at {} along periodic axis {}, not at 0: give the box's "
            "lower corner (lower=)".format(lo[off[0]], off[0]))
    return np.zeros(3)


def faces_congruent_arrays(comm, x, e, L, axes, tol=1e-9, exempt=None, lower=None):
    """`femtools.periodic.faces_congruent` on arrays: `x` the local vertices'
    positions (ghosts included), `e` the local edges as vertex pairs (None: vertices
    only), `L` and `lower` the box's corners, `exempt` a bool per vertex then per
    edge or an (m, 3) array of positions. Returns `(ok, axis, n_lo, n_hi)`. Collective."""
    edges = e is not None
    # the box moved to the origin, in three components
    lower = xyz(lower)
    L = xyz(L) - lower
    x = xyz(x) - lower
    # only the nodes on an identified plane are compared, or looked up among the exempt
    # positions: those are taken first, the vertices then the edges in their own order,
    # each midpoint's coordinate along an axis computed as the whole midpoint would be,
    # so every comparison below sees the rows it saw when every node was built and
    # matched
    on_v = on_planes(x, np.zeros(3), L, axes, tol)
    pts, keep = x[on_v], [np.nonzero(on_v)[0]]
    n_nodes = len(x)
    if edges:
        k = np.nonzero(on_planes(0.5 * (x[e[:, 0]] + x[e[:, 1]]), np.zeros(3), L, axes,
                                 tol))[0]
        pts = np.vstack((pts, 0.5 * (x[e[k, 0]] + x[e[k, 1]])))
        keep.append(len(x) + k)
        n_nodes += len(e)
    if exempt is not None and np.asarray(exempt).ndim == 2:
        exempt = _at_positions(pts, xyz(exempt) - lower, tol)
    elif exempt is not None:
        exempt = np.asarray(exempt, bool)
        if len(exempt) != n_nodes:
            raise ValueError("faces_congruent: exempt has {} entries for {} nodes".format(
                len(exempt), n_nodes))
        exempt = exempt[np.concatenate(keep)]
    if exempt is not None:
        return _faces_congruent_exempt(comm, pts, exempt, L, axes, tol)
    for a in axes:
        # an edge lies in the plane exactly when its midpoint does
        on_lo = np.abs(pts[:, a]) < tol
        on_hi = np.abs(pts[:, a] - L[a]) < tol
        n_lo, n_hi, bad_lo, bad_hi = _compare_sides(
            comm, pts[on_lo], pts[on_hi], a, L, tol)
        if bad_lo or bad_hi or not n_lo:
            return False, a, n_lo, n_hi
    return True, None, 0, 0


def on_planes(x, lower, upper, axes, tol):
    """Which points of `x` (coordinates along the last axis) lie within `tol` of an
    identified plane, lower[a] or upper[a], of `axes`."""
    on = np.zeros(np.shape(x)[:-1], bool)
    for a in axes:
        on |= (np.abs(x[..., a] - lower[a]) < tol) | (np.abs(x[..., a] - upper[a]) < tol)
    return on


def _compare_sides(comm, lo, hi, a, L, tol, ex_lo=None, ex_hi=None):
    """`(n_lo, n_hi, bad_lo, bad_hi)` over every rank: the distinct nodes on each side of
    axis `a` (`hi` taken back by L[a]), and those of each with no partner on the other
    side and not exempt. Each node goes by its rounded position to the rank its key
    hashes to, which deduplicates and compares; the counts are summed. Collective."""
    from mpi4py import MPI
    from femtools.hdf5_rows import find_rows, rank_of, route
    hi = np.array(hi, float).reshape(-1, 3)
    hi[:, a] -= L[a]
    keys = np.round(np.vstack([np.reshape(lo, (-1, 3)), hi]) / tol).astype(np.int64)
    side = np.r_[np.zeros(len(lo), np.int64), np.ones(len(hi), np.int64)]
    ex = np.zeros(len(keys), np.int64) if ex_lo is None else \
        np.r_[np.asarray(ex_lo, np.int64), np.asarray(ex_hi, np.int64)]
    keys, side, ex = route(comm, rank_of(keys, comm.Get_size()), keys, side, ex).received
    out = np.zeros(4, np.int64)
    sides = []
    for j in (0, 1):
        m = side == j
        if m.any():
            u, inv = np.unique(keys[m], axis=0, return_inverse=True)
            ex_u = np.zeros(len(u), bool)
            ex_u[inv.reshape(-1)[ex[m] > 0]] = True     # a ghosted copy carries the flag
        else:
            u, ex_u = np.zeros((0, 3), np.int64), np.zeros(0, bool)
        sides.append((u, ex_u))
        out[j] = len(u)
    for j in (0, 1):
        (u, ex_u), (other, _) = sides[j], sides[1 - j]
        out[2 + j] = np.count_nonzero((find_rows(u, other) < 0) & ~ex_u)
    out = comm.allreduce(out, op=MPI.SUM)
    return tuple(int(v) for v in out)


def _faces_congruent_exempt(comm, x, exempt, L, axes, tol):
    """`faces_congruent` with the cut exception: a face node may lack its partner if
    exempt. On failure the counts are the unpartnered nodes that are not exempt."""
    for a in axes:
        on_lo = np.abs(x[:, a]) < tol
        on_hi = np.abs(x[:, a] - L[a]) < tol
        n_lo, n_hi, bad_lo, bad_hi = _compare_sides(
            comm, x[on_lo], x[on_hi], a, L, tol, exempt[on_lo], exempt[on_hi])
        if not n_lo and not n_hi:
            return False, a, 0, 0
        if bad_lo or bad_hi:
            return False, a, bad_lo, bad_hi
    return True, None, 0, 0


def _at_positions(pts, where, tol):
    """Which of `pts` lie at one of the positions `where`, to `tol`."""
    if not len(pts) or not len(where):
        return np.zeros(len(pts), bool)
    from femtools.hdf5_rows import find_rows, unique_rows
    return find_rows(np.round(pts / tol).astype(np.int64),
                     unique_rows(np.round(where / tol).astype(np.int64))) >= 0


class TooManyCells(Exception):
    """The refinement would have made more cells than `max_cells`, or more on one rank
    than `max_rank_cells`; nothing was refined.

    Raised on every rank alike (both counts are reduced), before any vertex or cell of the
    new mesh exists -- so a caller can stop for the price of the marking."""

    def __init__(self, n_cells, max_cells, n_rank=0, max_rank_cells=0):
        self.n_cells, self.max_cells = int(n_cells), int(max_cells or 0)
        self.n_rank, self.max_rank_cells = int(n_rank), int(max_rank_cells or 0)
        if self.max_cells and self.n_cells > self.max_cells:
            what = "{} cells, over the {} allowed".format(self.n_cells, self.max_cells)
        else:
            what = "{} cells on the most loaded rank, over the {} allowed a rank".format(
                self.n_rank, self.max_rank_cells)
        super().__init__("the refinement would make {}; nothing was refined".format(what))
