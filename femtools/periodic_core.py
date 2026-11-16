"""The backend-free half of femtools' periodic boxes and their refinement, shared by the
dolfin modules (`femtools.periodic`, `femtools.plaza_periodic`) and the DOLFINx one
(`femtools.dolfinx.refine`).

    faces_congruent_arrays(comm, x, e, L, axes, tol, exempt, lower)
    box_scale(lower, upper)                the length the tolerances are relative to
    lower_corner(extent, axes, lower)      the box's lower corner (`periodic.box_lower`)
    on_planes(x, lower, upper, axes, tol)  which points lie on an identified plane
    xyz(a)                                 padded to three components
    TooManyCells                           a refinement refused by max_cells
    dorfler_threshold_values(values, theta, comm)
                                           the bulk-chasing threshold of a cellwise indicator
    refine_congruent(mesh, markers, box, axes, *, comm, refine_periodic, ...)
                                           both backends' refine_congruent, their
                                           refinement and check given
    TOL                                    the congruence check's tolerance

Every face node goes by its rounded position to the rank its key hashes to, where the two
sides are deduplicated and compared; only counts are reduced, so no rank holds a face.

Tolerances. Every tolerance a periodic box is judged with is relative: a number times
`box_scale(lower, upper)`, the box's largest extent or its corners' largest coordinate,
whichever is larger -- so a unit box at the origin takes the number as it is, and a box
far from the origin, whose coordinates carry their own rounding (one ulp at 1e6 is
1.2e-10), keeps its pairs. The numbers, by what they judge:

    1e-10  the refinement's pairing of face edges and its ties (`refine_periodic`'s tol)
    1e-9   the congruence check (`faces_congruent`, `congruent`; TOL): partners within
           2 NEAR TOL (2e-11) of each other always match, up to TOL by rounding
    1e-8   DOLFINx's dof pairing (`femtools.dolfinx.periodic.periodic_box`)

dolfin's own SubDomains (`femtools.periodic.PeriodicBox`, `periodic_box`) keep dolfin's
absolute convention, which dolfin's `constrained_domain` applies.
"""
import numpy as np

#: the congruence check's tolerance (relative to the box, `box_scale`)
TOL = 1e-9


def box_scale(lower, upper):
    """The length a periodic box's tolerances are relative to (see the module's
    docstring): the largest of its extents and of its corners' coordinates' magnitudes;
    1 for a degenerate box."""
    lower, upper = xyz(lower), xyz(upper)
    scale = float(max(np.max(np.abs(upper - lower)), np.max(np.abs(lower)),
                      np.max(np.abs(upper))))
    return scale if scale > 0 else 1.0


def xyz(a):
    """Coordinates, or a corner, padded with zeros to three components."""
    a = np.asarray(a, dtype=float)
    pad = [(0, 0)] * (a.ndim - 1) + [(0, 3 - a.shape[-1])]
    return np.pad(a, pad) if a.shape[-1] < 3 else a


def check_box(box, gdim, lower=None):
    """Refuse a box (or a lower corner) of fewer components than the mesh's dimension:
    padded with zeros, a missing axis would read as flat (upper == lower) and pair the
    nodes of different heights across it. More (a 2D mesh's 3D extent) are flat indeed.
    The same on every rank (its arguments are)."""
    for name, corner in (("box", box), ("lower", lower)):
        if corner is not None and len(np.ravel(corner)) < gdim:
            raise ValueError("refine_periodic: a {} of {} components on a {}-D mesh".format(
                name, len(np.ravel(corner)), gdim))


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
    edge or an (m, 3) array of positions, `tol` relative to the box (`box_scale`).
    Returns `(ok, axis, n_lo, n_hi)`. Collective."""
    edges = e is not None
    # the box moved to the origin, in three components
    lower = xyz(lower)
    tol = tol * box_scale(lower, L)
    L = xyz(L) - lower
    x = xyz(x) - lower
    # only the nodes on an identified plane are compared, or looked up among the exempt
    # positions: those are taken first, the vertices then the edges in their own order,
    # each midpoint computed whole, so the comparisons see the same rows as they would
    # with every node built
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
        from femtools.hdf5_rows import agreed
        if agreed(comm, len(exempt) != n_nodes):     # every rank raises, or none
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


# how far (in units of tol) a node's position may lie from its partner's for the two
# to be found by the keys they share (`hdf5_rows.near_keys`)
NEAR = 0.01


def _compare_sides(comm, lo, hi, a, L, tol, ex_lo=None, ex_hi=None):
    """`(n_lo, n_hi, bad_lo, bad_hi)` over every rank: the distinct nodes on each side of
    axis `a` (`hi` taken back by L[a]), and those of each with no partner on the other
    side and not exempt. A node is its rounded position (ghosted copies agree); it goes
    under each of its `hdf5_rows.near_keys` to the rank the key hashes to, where the sides meet,
    then with what it met to the rank of its rounded position, which deduplicates and
    counts. Only counts are reduced. Collective."""
    from mpi4py import MPI
    from femtools.hdf5_rows import near_keys, rank_of, reply, route
    hi = np.array(hi, float).reshape(-1, 3)
    hi[:, a] -= L[a]
    y = np.vstack([np.reshape(lo, (-1, 3)), hi]) / tol
    node = np.round(y).astype(np.int64)
    side = np.r_[np.zeros(len(lo), np.int64), np.ones(len(hi), np.int64)]
    ex = np.zeros(len(node), np.int64) if ex_lo is None else \
        np.r_[np.asarray(ex_lo, np.int64), np.asarray(ex_hi, np.int64)]
    # 1. the sides meet under each near key: a key met by both matches its nodes
    keys, of = near_keys(y, NEAR)
    routed = route(comm, rank_of(keys, comm.Get_size()), keys, side[of])
    k, s = routed.received
    met = np.zeros(len(k), np.int64)
    if len(k):
        _, inv = np.unique(k, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        both = np.zeros((inv.max() + 1, 2), bool)
        both[inv, s] = True
        met = both[inv, 1 - s].astype(np.int64)
    (met,) = reply(comm, routed, met)
    matched = np.zeros(len(node), np.int64)
    np.maximum.at(matched, of, met)
    # 2. each node, with whether it matched, to the rank of its rounded position
    node, side, ex, matched = route(comm, rank_of(node, comm.Get_size()), node, side, ex,
                                    matched).received
    out = np.zeros(4, np.int64)
    for j in (0, 1):
        m = side == j
        if not m.any():
            continue
        u, inv = np.unique(node[m], axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        ok = np.zeros(len(u), bool)        # a copy that matched, or that is exempt
        ok[inv[(matched[m] > 0) | (ex[m] > 0)]] = True
        out[j] = len(u)
        out[2 + j] = np.count_nonzero(~ok)
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
    """Which of `pts` lie at one of the positions `where`, to `tol` (and never miss one
    on a rounding boundary: `hdf5_rows.near_keys`)."""
    if not len(pts) or not len(where):
        return np.zeros(len(pts), bool)
    from femtools.hdf5_rows import find_rows, near_keys, unique_rows
    keys, of = near_keys(np.asarray(pts) / tol, 2 * NEAR)
    hit = find_rows(keys, unique_rows(np.round(np.asarray(where) / tol).astype(np.int64))) >= 0
    out = np.zeros(len(pts), bool)
    out[of[hit]] = True
    return out


def dorfler_threshold_values(values, theta, comm=None, iterations=60):
    """The largest t with the sum of |values| over {|value| > t} at least theta of all,
    `values` this rank's share of a cellwise indicator (comm None for one rank).

    Bulk chasing (Doerfler 1996): refine the smallest set of cells carrying a fixed
    share of the total. Scale free, so it does not care what units the indicator is in,
    and it responds to how concentrated the error is rather than to how large. An
    indicator carrying a sign counts by |value|: a share of the total contribution, not
    of the estimate.

    Found by bisection on the threshold rather than by sorting, so nothing is gathered
    and the cost is a fixed number of small reductions, every rank leaving with the same
    answer. Collective."""
    from femtools.hdf5_rows import allreduce, allsum
    a = np.abs(np.asarray(values, dtype=float)).ravel()
    total = float(allsum(comm, a.sum()))
    lo, hi = 0.0, allreduce(comm, float(a.max()) if len(a) else 0.0, "max")
    if total <= 0.0:
        return 0.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        got = float(allsum(comm, a[a > mid].sum()))
        lo, hi = (mid, hi) if got >= theta * total else (lo, mid)
    return lo


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


def refine_congruent(mesh, cell_markers, box, axes, *, comm, refine_periodic,
                     check_congruent, refine_plain, plain_name, redistribute=False,
                     max_cells=None, exempt=None, max_rank_cells=None, frozen=None,
                     report=None, deferred=False, lower=None):
    """The periodic-safe refinement, with its output checked before it is trusted: the
    one body of `femtools.refine_periodic.refine_congruent` (dolfin) and
    `femtools.dolfinx.refine.refine_congruent`, which give their backend's
    `refine_periodic`, `check_congruent`, `refine_plain(mesh, cell_markers)` (the
    library's own refinement, for no periodic axes; `plain_name` names it) and `comm`.
    Returns `(new_mesh, npair, parent_cell)`; see the backends' docstrings."""
    def say(text):
        if comm.rank == 0:
            print(text, flush=True)

    if deferred and exempt is not None:          # before the refinement, not after it
        raise ValueError("refine_congruent: a deferred guard takes its exempt nodes at "
                         "check_congruent")
    say("refining")
    if axes:
        new_mesh, npair, parent_cell = refine_periodic(
            mesh, cell_markers, box, axes, redistribute=redistribute, max_cells=max_cells,
            max_rank_cells=max_rank_cells, frozen=frozen, report=report, lower=lower)
        say("periodic Plaza: {} identified edge pairs".format(npair))
        if not deferred:
            check_congruent(new_mesh, box, axes,
                            None if exempt is None else exempt(new_mesh, parent_cell),
                            lower)
        return new_mesh, npair, parent_cell
    if max_cells or max_rank_cells or frozen is not None:
        raise ValueError("max_cells and frozen need periodic axes: {} can neither count "
                         "nor freeze before it refines".format(plain_name))
    return refine_plain(mesh, cell_markers), 0, None
