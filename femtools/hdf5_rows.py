#!/usr/bin/env python3
"""Scattered rows of an HDF5 dataset, without any rank holding the whole of it.

Arrays keyed by a global index -- a parent-cell chain, a mesh's topology and
coordinates, a stored dofmap -- are read by ranks that each want a *scattered subset*:
the rows for their own local cells. dolfin scatters global cell ids across ranks, so
there is no slice smaller than the file, and h5py's fancy selection is far slower than
a whole read. Read whole on every rank, though, the array is held once a rank.
So: **each rank reads a contiguous block and the ranks shuffle rows to whoever asked
for them.**

    block read      rank r reads rows [n*r/size, n*(r+1)/size) and nothing else
    request         each rank sends the owner of each row it wants that row's index
    reply           each owner answers out of the block it holds

Two `Alltoallv` for the payloads and one `alltoall` for the counts. No rank holds more
than one block plus what it asked for (unless a caller keeps several, from `read_block`
and `invert_labels`, to serve many requests by `lookup`), and the block read is
contiguous, which is the access pattern HDF5 is fastest at -- so this is cheaper in time
than the fancy selection it replaces as well as in memory.

**mpi4py is imported lazily**, only on the parallel path, so a serial caller needs numpy
and h5py alone. With no comm, or on one rank, this is a plain read and nothing is imported.
"""
import contextlib
import os

import numpy as np


def block_edges(n, size):
    """Where each rank's contiguous block starts; `size + 1` entries."""
    return np.array([(n * r) // size for r in range(size + 1)], dtype=np.int64)


#: entries of one array a rank sends, or receives, in one Alltoallv: MPI's counts and
#: displacements are 32-bit (2^31 - 1), and a rank of a 2B-cell array on few ranks passes
#: that. More goes in rounds (`n_rounds`). FEMTOOLS_ROUND overrides it, for the
#: tests to force many rounds
ROUND = int(os.environ.get("FEMTOOLS_ROUND", 1 << 30))


def owner_of(edges, index, comm=None):
    """Which rank's block (`block_edges`) holds each global index; an index outside
    [0, edges[-1]) is refused on every rank together."""
    index = np.asarray(index, dtype=np.int64)
    bad = bool(index.size) and (index.min() < 0 or index.max() >= edges[-1])
    if agreed(comm, bad):
        raise ValueError("an index outside the {} rows the blocks cover{}".format(
            int(edges[-1]), ": {}..{} here".format(int(index.min()), int(index.max()))
            if bad else " (on another rank)"))
    return np.searchsorted(edges, index, side="right") - 1


def n_rounds(most_sent, most_received, limit=None):
    """Exchange rounds for the most entries a rank sends and a rank receives: enough that
    `round_slices` puts at most `limit` (default `ROUND`) plus one row a peer into any
    rank's round, sent or received."""
    limit = ROUND if limit is None else limit
    return int(max(-(-int(most_sent) // limit), -(-int(most_received) // limit), 1))


def round_slices(counts, r, rounds):
    """(lo, hi): round r's share of each peer's `counts` rows, as offsets into that
    peer's run, counts r // rounds .. counts (r + 1) // rounds. Sender and receiver
    compute the same split from the same count, so neither tells the other."""
    counts = np.asarray(counts, dtype=np.int64)
    return (counts * r) // rounds, (counts * (r + 1)) // rounds


#: row dimensions a layout signature (`_signature`) carries; an array of more is agreed
#: through `_gathered_layout`
_SIGNATURE_DIMS = 4
_BIG = np.iinfo(np.int64).max


def _signature(a, n):
    """`(fields, general)`: an array's dtype and row shape as int64 fields for a buffer
    reduction (the dtype's `str` packed into one int64, the number of row dimensions,
    each dimension), and whether they do not fit there (a structured or subarray dtype,
    a long dtype string, more than `_SIGNATURE_DIMS` row dimensions). A rank with no rows
    (n == 0) has no fields."""
    if not n:
        return None, False
    dt, shape = a.dtype, tuple(a.shape[1:])
    code = dt.str.encode()
    if dt.fields is not None or dt.subdtype is not None or len(code) > 8 \
            or len(shape) > _SIGNATURE_DIMS:
        return None, True
    packed = int(np.frombuffer(code.ljust(8, b"\0"), dtype="<i8")[0])
    dims = list(shape) + [0] * (_SIGNATURE_DIMS - len(shape))
    return np.array([packed, len(shape)] + dims, dtype=np.int64), False


def _gathered_layout(comm, a, n):
    """`_agreed_layout` through an allgather of every rank's dtype and row shape: for the
    layouts a signature does not carry, and for the message naming disagreeing ones."""
    mine = (a.dtype.str, tuple(a.shape[1:])) if n else None
    seen = {t for t in comm.allgather(mine) if t is not None}
    if len(seen) > 1:
        raise ValueError("ranks pass rows of different dtypes or shapes: {}".format(
            sorted(seen)))
    if not seen:        # no rank has rows: rank zero's, so that all still agree
        dt, shape = comm.bcast((a.dtype.str, tuple(a.shape[1:])), root=0)
        return np.dtype(dt), shape
    dt, shape = seen.pop()
    return np.dtype(dt), shape


def _agreed_layout(comm, a, n):
    """The dtype and row shape every rank with rows passed for an array (rank zero's
    when none has rows); a rank with none may pass any (an empty default). Disagreeing
    ranks are refused on every rank:
    received into its own dtype, a row would be another's bytes reinterpreted.

    One buffer Allreduce of each rank's `_signature`, its minimum and maximum: equal
    means every rank with rows agrees. What a signature does not carry, a disagreement
    (for its message) and no rank with rows go through `_gathered_layout`."""
    from mpi4py import MPI
    fields, general = _signature(a, n)
    width = 2 + _SIGNATURE_DIMS
    v = np.full(2 * width + 2, _BIG, dtype=np.int64)
    if fields is not None:
        v[:width], v[width:2 * width] = fields, -fields
    v[-2:] = -int(bool(n)), -int(general)
    out = np.empty_like(v)
    comm.Allreduce(v, out, op=MPI.MIN)
    lo, hi = out[:width], -out[width:2 * width]
    if out[-1] < 0 or out[-2] == 0 or (lo != hi).any():
        return _gathered_layout(comm, a, n)
    code = lo[0:1].astype("<i8").tobytes().rstrip(b"\0").decode()
    return np.dtype(code), tuple(int(d) for d in lo[2:2 + int(lo[1])])


def _alltoall_counts(comm, counts):
    """Every rank's `counts[r]` for this rank r, as int64: one buffer Alltoall."""
    counts = np.ascontiguousarray(counts, dtype=np.int64)
    recv = np.empty(comm.Get_size(), dtype=np.int64)
    comm.Alltoall(counts, recv)
    return recv


def _int_reduce(comm, values, op):
    """Integers reduced elementwise over the ranks by `op` ("max" or "sum"), as int64, in
    one buffer Allreduce."""
    from mpi4py import MPI
    v = np.ascontiguousarray(np.atleast_1d(values), dtype=np.int64)
    out = np.empty_like(v)
    comm.Allreduce(v, out, op={"max": MPI.MAX, "sum": MPI.SUM}[op])
    return out


def exchange(comm, counts, *arrays):
    """Rows grouped by destination -- `counts[r]` rows for rank r, in rank order -- sent;
    returns `(recv_counts, [received])`, each array's rows grouped by source rank in rank
    order. Every array has `counts.sum()` rows, of any row shape. Their dtypes and row
    shapes are agreed over the ranks (`_agreed_layout`), and an array a rank would send
    or receive more than `ROUND` entries of goes in rounds, so MPI's 32-bit counts never
    wrap. Collective. With no comm (one rank) the arrays come back as they are."""
    counts = np.asarray(counts, dtype=np.int64)
    if comm is None:
        return counts, [np.asarray(a) for a in arrays]
    recv_counts = _alltoall_counts(comm, counts)
    n_send = int(counts.sum())
    out = []
    for a in arrays:
        a = np.asarray(a)
        dt, shape = _agreed_layout(comm, a, n_send)
        flat = np.ascontiguousarray(a, dtype=dt).reshape((n_send,) + shape)
        out.append(_send_rows(comm, counts, recv_counts, flat)[0])
    return recv_counts, out


def _send_rows(comm, counts, recv_counts, send, rounds=None):
    """`(received, rounds)`: `send` (rows grouped by destination, `counts[r]` for rank r)
    exchanged for the rows this rank receives (`recv_counts[r]` from rank r), grouped by
    source rank; in `rounds` rounds when a rank would send or receive more than `ROUND`
    entries at once (worked out over the ranks when None). Collective."""
    from mpi4py import MPI
    shape = send.shape[1:]
    width = int(np.prod(shape)) if shape else 1
    dt = send.dtype
    flat = send.reshape(len(send), width)
    n_recv = int(recv_counts.sum())
    recv = np.empty((n_recv, width), dtype=dt)
    if rounds is None:
        most = int(_int_reduce(comm, max(len(send), n_recv) * width, "max")[0])
        rounds = n_rounds(most, most)
    s_start = np.concatenate(([0], np.cumsum(counts)[:-1]))
    r_start = np.concatenate(([0], np.cumsum(recv_counts)[:-1]))
    mpi_type = MPI._typedict[dt.char]
    for r in range(rounds):
        s_lo, s_hi = round_slices(counts, r, rounds)
        r_lo, r_hi = round_slices(recv_counts, r, rounds)
        if rounds == 1:
            out, got = flat, recv
        else:
            out = flat[np.concatenate([np.arange(b + lo, b + hi) for b, lo, hi
                                       in zip(s_start, s_lo, s_hi)])]
            got = np.empty((int((r_hi - r_lo).sum()), width), dtype=dt)
        sc = ((s_hi - s_lo) * width).astype(np.int32)
        rc = ((r_hi - r_lo) * width).astype(np.int32)
        sd = np.concatenate(([0], np.cumsum(sc)[:-1])).astype(np.int32)
        rd = np.concatenate(([0], np.cumsum(rc)[:-1])).astype(np.int32)
        comm.Alltoallv([out, sc, sd, mpi_type], [got, rc, rd, mpi_type])
        if rounds > 1:
            recv[np.concatenate([np.arange(b + lo, b + hi) for b, lo, hi
                                 in zip(r_start, r_lo, r_hi)])] = got
    return recv.reshape((n_recv,) + shape), rounds


class Routed:
    """What `route` received (`received`, one array per array sent) and how to answer it
    (`reply`): the sort that grouped the rows (`order`, None on one rank, where nothing
    moved) and both ways' counts."""

    def __init__(self, received, order, counts, recv_counts):
        self.received, self.order = received, order
        self.counts, self.recv_counts = counts, recv_counts


def route(comm, dest, *arrays):
    """Row i of every array to rank `dest[i]` (checked in range on every rank); returns a
    `Routed`. Collective; no comm is one rank, where the rows stay as they are (nothing
    sorted or copied)."""
    dest = np.asarray(dest, dtype=np.int64)
    size = 1 if comm is None else comm.Get_size()
    bad = bool(dest.size) and (dest.min() < 0 or dest.max() >= size)
    if agreed(comm, bad):
        raise ValueError("route: a destination outside the {} ranks".format(size))
    if size == 1:
        n = np.array([len(dest)], dtype=np.int64)
        return Routed([np.asarray(a) for a in arrays], None, n, n)
    order = rank_order(dest, size)
    counts = np.bincount(dest, minlength=size).astype(np.int64)
    recv_counts, got = exchange(comm, counts, *(np.asarray(a)[order] for a in arrays))
    return Routed(got, order, counts, recv_counts)


def rank_order(dest, size):
    """`np.argsort(dest, kind="stable")` for rank numbers 0..size-1, the same permutation:
    sorted as 16-bit integers when they fit, which numpy radix-sorts, cheaper than its
    stable sort of 64-bit ones (every `route` sorts its rows by rank)."""
    dest = np.asarray(dest)
    if size <= np.iinfo(np.int16).max + 1:
        dest = dest.astype(np.int16)
    return np.argsort(dest, kind="stable")


def reply(comm, routed, *answers):
    """Answers to what `routed` received (aligned with its `received`) sent back to the
    ranks that asked; returns them aligned with the rows those ranks routed. Collective."""
    if routed.order is None:                  # one rank: the rows never moved
        return [np.asarray(b) for b in answers]
    _, back = exchange(comm, routed.recv_counts, *answers)
    out = []
    for b in back:
        o = np.empty_like(b)
        o[routed.order] = b
        out.append(o)
    return out


def to_owners(comm, edges, index, *values):
    """Each `(index, values rows)` sent to the rank whose block holds `index`; returns what
    this rank received, `[index, values...]`, in no particular order. Collective."""
    index = np.asarray(index, dtype=np.int64)
    return route(comm, owner_of(edges, index, comm), index, *values).received


def as_rows(a):
    """`a` as rows, `(len(a), the rest flattened)`: `reshape(len(a), -1)`, also for a rank
    holding no rows, where numpy cannot infer the -1 and raises -- on that rank alone,
    before the collective the others then wait in."""
    a = np.asarray(a)
    return a.reshape(len(a), int(np.prod(a.shape[1:], dtype=np.int64)))


def rank_of(rows, size):
    """The rank each row of integers (any size) meets at, out of `size`: the same row on
    every rank to the same rank, rows spread evenly. The one routing hash for `route`'s
    destinations. Not an identity: two rows may meet at one rank, which compares the rows
    themselves. Rows of any width and value range are hashed whole (no bit packing to
    overflow)."""
    rows = as_rows(rows).astype(np.int64).view(np.uint64)
    h = np.zeros(len(rows), np.uint64)
    with np.errstate(over="ignore"):
        for c in rows.T:                    # splitmix64's finaliser over each column
            h = h ^ c
            h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            h = h ^ (h >> np.uint64(31))
        return ((h >> np.uint64(32)) * np.uint64(size) >> np.uint64(32)).astype(np.int64)


def world():
    """mpi4py's world, or None where there is no mpi4py (the venv, which runs serially:
    the same code path as one rank). Every helper here takes None for one rank."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.COMM_WORLD


def allsum(comm, x):
    """Elementwise sum over the ranks of a float array (or scalar), on every rank."""
    x = np.asarray(x, dtype=np.float64)
    if comm is None or comm.Get_size() == 1:
        return x
    from mpi4py import MPI
    out = np.empty_like(x)
    comm.Allreduce(x, out, op=MPI.SUM)
    return out


def allreduce(comm, value, op):
    """`value` reduced over the ranks by `op` ("min", "max" or "sum"), on every rank."""
    if comm is None or comm.Get_size() == 1:
        return value
    from mpi4py import MPI
    return comm.allreduce(value, op={"min": MPI.MIN, "max": MPI.MAX, "sum": MPI.SUM}[op])


def agreed(comm, flag):
    """Whether **any** rank set `flag` -- so that all of them raise, or none does.

    A rank-local `raise` just before a collective is the shape this project has been
    bitten by: the objecting rank leaves, the others sit in the shuffle, and the job
    dies by MPI abort rather than by the message the guard took the trouble to write.
    Every guard here goes through this.
    """
    if comm is None or comm.Get_size() == 1:
        return bool(flag)
    return bool(_int_reduce(comm, int(bool(flag)), "max")[0])


def check_range(path, dataset, wanted, n, comm=None):
    """Refuse indices the dataset does not hold, on every rank together."""
    bad = bool(wanted.size) and (wanted.min() < 0 or wanted.max() >= n)
    if agreed(comm, bad):
        where = ("{}..{}".format(int(wanted.min()), int(wanted.max()))
                 if bad else "nothing out of range here, but another rank disagreed")
        raise ValueError("{}:{} holds {} rows; asked for {}".format(
            path, dataset, n, where))


def fetch_rows(path, dataset, wanted, comm=None, n=None):
    """Rows `wanted` (global indices) of `path:dataset`, in that order.

    Duplicates and any ordering are allowed; the rows come back aligned with `wanted`.
    Serial, or with one rank, this is a whole read and an index -- far cheaper than an
    h5py fancy selection and, at one rank, no worse in memory since there is
    nobody to share the array with. The bounds check is the same either way: a negative
    index would otherwise wrap to a real row serially and raise under MPI, and an
    asymmetry like that outlives the refactor that introduced it.

    `n`: only rows [0, n) are read, and asked for -- a slab output's own rows, its buffer
    after them unread (`read_block`).
    """
    import h5py
    wanted = np.asarray(wanted, dtype=np.int64)
    size = 1 if comm is None else comm.Get_size()
    if size == 1:
        with h5py.File(path, "r") as h5:
            dset = h5[dataset]
            n = _rows_read(dset, n)
            check_range(path, dataset, wanted, n, comm)
            return np.asarray(dset[:n])[wanted]

    edges, block = read_block(path, dataset, comm, n=n)
    check_range(path, dataset, wanted, int(edges[-1]), comm)
    return lookup(comm, edges, block, wanted)


def read_block(path, dataset, comm, group=1, n=None):
    """`(edges, block)`: this rank's contiguous block of `path:dataset`, and where each
    rank's starts. What `fetch_rows` reads on every call, for a caller to keep and hand
    to `lookup` itself when it asks the same dataset many times. A block `read_ahead`
    holds is served from there instead (the same array), and dropped once served as
    often as it was listed.

    `group` > 1 reads a flat dataset as rows of `group` entries -- a dofmap's
    `cell_dofs` as one row a cell -- so blocks hold whole rows and `edges` count rows.
    `n` (group 1) blocks rows [0, n) of it alone."""
    key = _ahead_key(path, dataset, group, n)
    if key in _AHEAD:
        entry = _AHEAD[key]
        entry[2] -= 1
        if entry[2] <= 0:
            del _AHEAD[key]
        return entry[0], entry[1]
    import h5py
    with h5py.File(path, "r") as h5:
        return _read_block(h5[dataset], comm.Get_rank(), comm.Get_size(), int(group), n)


def _rows_read(dset, n):
    """The rows of `dset` a read with limit `n` covers: all of them, or the first `n`."""
    rows = int(dset.shape[0])
    if n is None:
        return rows
    if not 0 <= int(n) <= rows:
        raise ValueError("{}: rows [0, {}) asked of {}".format(dset.name, int(n), rows))
    return int(n)


def _read_block(dset, rank, size, group, n=None):
    if group == 1:
        edges = block_edges(_rows_read(dset, n), size)
        return edges, np.asarray(dset[edges[rank]:edges[rank + 1]])
    if n is not None:
        raise ValueError("{}: a row limit is for rows of one entry".format(dset.name))
    m = int(dset.shape[0])
    if m % group:
        raise ValueError("{} holds {} entries, not rows of {}".format(dset.name, m, group))
    edges = block_edges(m // group, size)
    block = np.asarray(dset[edges[rank] * group:edges[rank + 1] * group])
    return edges, block.reshape(-1, group)


def _ahead_key(path, dataset, group=1, n=None):
    return (os.path.realpath(path), dataset, int(group), None if n is None else int(n))


#: blocks `read_ahead` has read and `read_block` has yet to serve:
#: (realpath, dataset, group, row limit) -> [edges, block, times still to serve]
_AHEAD = {}


def read_ahead(comm, items, stagger=True):
    """Read this rank's `read_block` of every `(path, dataset)`, `(path, dataset,
    group)` or `(path, dataset, group, n)` in `items` now, in one pass, for `read_block`
    to serve later; an item listed k times is read once and served k times. Returns `{path: [seconds, bytes]}`, this rank's reading time and bytes per
    file. Nothing is read without a comm of more than one rank (`fetch_rows` reads whole
    files there). `drop_ahead` forgets what was not served.

    Why: the readers otherwise take their datasets in the same order on every rank, one
    collective step at a time, so at any moment every rank reads the same file. On
    Lustre a file lives on its own few storage targets, and a step then runs at one
    file's bandwidth while the others' targets idle. With `stagger` each rank starts at
    a different file -- rank r at the file holding the fraction (r + 1/2)/P of the
    listed bytes, so a big file starts with proportionally many readers -- and takes
    the files cyclically from there, so all of them are read at once. The blocks are the
    ones `read_block` reads, so the values are the same bits either way.
    """
    import time
    import h5py
    if comm is None or comm.Get_size() == 1:
        return {}
    rank, size = comm.Get_rank(), comm.Get_size()
    want = {}
    for item in items:
        item = tuple(item)
        key = _ahead_key(*item[:2], *item[2:4])
        want[key] = want.get(key, 0) + 1
    files = list(dict.fromkeys(k[0] for k in want))
    if stagger and files:
        from femtools.jit import on_rank0

        def weigh():                        # one rank asks the metadata server
            weights = []
            for path in files:
                with h5py.File(path, "r") as h5:
                    weights.append(sum(h5[d].id.get_storage_size()
                                       * (1.0 if n is None else n / max(h5[d].shape[0], 1))
                                       for p, d, _, n in want if p == path))
            return weights
        # rank 0's error (a missing file) raised on every rank, not rank 0 alone
        weights = np.asarray(on_rank0(comm, weigh, "read_ahead"), dtype=np.float64)
        cum = np.cumsum(weights) / max(weights.sum(), 1.0)
        first = min(int(np.searchsorted(cum, (rank + 0.5) / size, side="right")),
                    len(files) - 1)
        files = files[first:] + files[:first]
    stats = {}
    for path in files:
        t0, nbytes = time.time(), 0
        with h5py.File(path, "r") as h5:
            for (p, dataset, group, n), count in want.items():
                if p != path:
                    continue
                edges, block = _read_block(h5[dataset], rank, size, group, n)
                nbytes += block.nbytes
                _AHEAD[(p, dataset, group, n)] = [edges, block, count]
        stats[path] = [time.time() - t0, nbytes]
    return stats


def drop_ahead():
    """Forget every block `read_ahead` read and `read_block` has not served."""
    _AHEAD.clear()


def lookup(comm, edges, block, wanted):
    """Rows `wanted` of an array whose blocks the ranks hold between them.

    The core of `fetch_rows`, factored out because `rows_of_labels` below needs the
    same exchange over a block it computes rather than reads. On one rank (or no comm)
    it is an index into the block.
    """
    if comm is None or comm.Get_size() == 1:
        local = np.asarray(wanted, dtype=np.int64) - int(edges[0])
        if local.size and (local.min() < 0 or local.max() >= len(block)):
            raise ValueError("an index outside the {} rows the blocks cover".format(
                len(block)))
        return np.asarray(block)[local]
    rank = comm.Get_rank()
    block = np.asarray(block)

    # ask for each distinct row once, at the rank whose block holds it
    asked, inverse = unique_inverse(wanted)
    routed = route(comm, owner_of(edges, asked, comm), asked)
    local = routed.received[0] - edges[rank]
    if agreed(comm, local.size and (local.min() < 0 or local.max() >= len(block))):
        raise ValueError("a request reached the wrong rank; the block map disagrees")
    replies, = reply(comm, routed, block[local].reshape((len(local),) + block.shape[1:]))
    out = replies[inverse]                     # undo the deduplication
    return out.reshape((len(wanted),) + block.shape[1:])


def unique_rows(rows):
    """`np.unique(rows, axis=0)` for int rows, the same rows in the same order, many times
    faster than numpy's sort of the rows as one structured value each (the periodic
    repair's largest cost): the columns packed, each from its minimum, into one int64 that sorts in
    the rows' order when their ranges fit 63 bits (vertex numbers, ranks and the kind do),
    else a lexsort and a neighbour comparison."""
    if not len(rows):
        return rows.reshape(0, rows.shape[1])
    lo = rows.min(axis=0)
    bits = [int(w).bit_length() for w in rows.max(axis=0) - lo]
    if sum(bits) <= 63:
        key = np.zeros(len(rows), np.int64)
        for c, b in enumerate(bits):
            key = (key << b) | (rows[:, c] - lo[c])
        key = np.unique(key)
        out = np.empty((len(key), rows.shape[1]), rows.dtype)
        for c in range(rows.shape[1] - 1, -1, -1):
            out[:, c] = (key & ((1 << bits[c]) - 1)) + lo[c]
            key = key >> bits[c]
        return out
    s = rows[np.lexsort(rows.T[::-1])]
    return s[np.r_[True, np.any(s[1:] != s[:-1], axis=1)]]


def find_rows(a, b):
    """For each row of `a`, the index of the equal row of `b` (unique rows), or -1."""
    if not len(a) or not len(b):
        return np.full(len(a), -1, np.int64)
    both = np.concatenate([b, a])
    order = np.lexsort(both.T[::-1])
    s = both[order]
    first = np.r_[True, np.any(s[1:] != s[:-1], axis=1)]
    group = np.cumsum(first) - 1
    head = np.full(group[-1] + 1, -1, np.int64)
    from_b = order < len(b)
    head[group[from_b]] = order[from_b]
    out = np.empty(len(both), np.int64)
    out[order] = head[group]
    return out[len(b):]


def extent(x, comm=None, width=None):
    """`(lower, upper)`: the corners of the bounding box of every rank's rows `x` of
    coordinates, `width` components each (a rank may hold none). Collective."""
    x = np.asarray(x, float)
    width = x.shape[1] if width is None else int(width)
    lo = x.min(axis=0) if len(x) else np.full(width, np.inf)
    hi = x.max(axis=0) if len(x) else np.full(width, -np.inf)
    if comm is None or comm.Get_size() == 1:
        return lo, hi
    from mpi4py import MPI
    out_lo, out_hi = np.empty(width), np.empty(width)
    comm.Allreduce(np.ascontiguousarray(lo), out_lo, op=MPI.MIN)
    comm.Allreduce(np.ascontiguousarray(hi), out_hi, op=MPI.MAX)
    return out_lo, out_hi


def near_keys(y, delta):
    """`(keys, index)`: every integer rounding of the rows of `y` moved by up to `delta`
    (less than a half) in each coordinate -- one key a point, more where a coordinate lies
    within `delta` of a rounding boundary -- and the point each key is of. Two points
    within `2 delta` of each other in every coordinate share a key, and two within `delta`
    find each other's `np.round` among their keys."""
    y = as_rows(np.asarray(y, dtype=float))
    lo, hi = np.round(y - delta).astype(np.int64), np.round(y + delta).astype(np.int64)
    keys, index = lo, np.arange(len(y))
    for d in range(y.shape[1]):
        m = lo[index, d] != hi[index, d]
        extra = keys[m].copy()
        extra[:, d] = hi[index[m], d]
        keys, index = np.vstack([keys, extra]), np.r_[index, index[m]]
    return keys, index


def match_nodes(comm, coords, values, x, tol, lower):
    """The rows of `values` at the points `x`: each point matched to its nearest node of
    `coords` (`values` row for row), nodes and points spread over the ranks in any way.
    Returns `(found, distance)`, aligned with `x`: the row at each point's node and how
    far that node is. A node within `tol` of a point is always found; past `tol` the
    nearest may be missed, and a point with no node at all gets `inf`. Collective on comm
    (None for one rank).

    Under MPI nodes and points meet by position: space is cut into cubes of side
    `1000 tol` from `lower`, a point goes to the rank of its cube (`rank_of` of its
    rounded position) and a node to the ranks of every cube within `tol` of it
    (`near_keys`), so each is sent about once however the ranks' boxes overlap."""
    from scipy.spatial import cKDTree
    coords, values = as_rows(np.asarray(coords, dtype=float)), as_rows(values)
    x = as_rows(np.asarray(x, dtype=float))
    width = values.shape[1]
    asked, points = None, x
    if comm is not None and comm.Get_size() > 1:
        size = comm.Get_size()
        h = 1e3 * tol
        lower = np.asarray(lower, dtype=float)
        keys, of = near_keys((coords - lower) / h, tol / h)
        coords, values = route(comm, rank_of(keys, size), coords[of], values[of]).received
        asked = route(comm, rank_of(np.round((x - lower) / h), size), x)
        points = asked.received[0]
    if len(points) and len(coords):
        distance, index = cKDTree(coords).query(points)
        found = values[index]
    else:
        distance = np.full(len(points), np.inf)
        found = np.zeros((len(points), width), dtype=values.dtype)
    if asked is not None:
        found, distance = reply(comm, asked, found, distance)
    return found.reshape(len(x), width), distance

def unique_inverse(values):
    """`np.unique(values, return_inverse=True)`, the same two arrays, for the 1-D int64
    arrays of row and vertex ids the readers deduplicate: value and position packed into
    one int64 and sorted once, a plain sort being cheaper than the argsort `np.unique`
    makes. Anything else (another dtype or
    shape, negative values, or values too large to pack) goes to `np.unique`."""
    a = np.asarray(values)
    n = a.size
    if a.dtype != np.int64 or a.ndim != 1 or n < 2:
        return np.unique(a, return_inverse=True)
    shift = int(n - 1).bit_length()
    if a.min() < 0 or int(a.max()) >= 1 << (63 - shift):
        return np.unique(a, return_inverse=True)
    key = (a << shift) | np.arange(n, dtype=np.int64)
    key.sort()
    first = np.empty(n, dtype=bool)
    first[0] = True
    sorted_values = key >> shift
    np.not_equal(sorted_values[1:], sorted_values[:-1], out=first[1:])
    inverse = np.empty(n, dtype=np.intp)
    inverse[key & ((1 << shift) - 1)] = np.cumsum(first) - 1
    return sorted_values[first], inverse


def _check_labels(path, dataset, labels, n, comm):
    """Every label names a row of the dataset. Agreed across ranks, never rank-local.

    Both of these guards sit in front of a collective, and a lone rank raising there
    turns a written message into a hang -- which is how this was found, by a test that
    deadlocked instead of failing. They are shared with the serial branch so the two
    refuse the same files: `np.empty` plus a duplicated label returns *uninitialised
    memory* rather than raising, which is the worst of the three outcomes.
    """
    bad = bool(labels.size) and (labels.min() < 0 or labels.max() >= n)
    if agreed(comm, bad):
        raise ValueError("{}:{} holds a label outside the {} rows it labels".format(
            path, dataset, n))


def _check_covered(path, dataset, inverse, comm):
    """Every row was claimed by exactly one label, i.e. the labelling is a permutation.

    A duplicate leaves some other label unclaimed, so this catches both.
    """
    if agreed(comm, bool((inverse < 0).any())):
        raise ValueError("{}:{} does not label every row exactly once, so it is not a "
                         "permutation and cannot be inverted".format(path, dataset))


def rows_of_labels(path, dataset, wanted, comm=None):
    """Which row carries each wanted label, where `dataset` labels the rows.

    dolfin writes `mesh/cell_indices` and a function's `cells` to say which global cell
    each stored row belongs to, and a reader addressing rows by global id has to invert
    that. **It is not always the identity**: a refined mesh comes out in global order, but
    a Function written under MPI does not -- a `delta.h5` written on 3 ranks has a
    scrambled `delta/cells`, which is the normal case for a ladder rung and was found
    by a test failing rather than by reading the format.

    Inverting it without anyone holding it: each rank reads its block of labels, so it
    holds (label, row) pairs for those rows; it ships each pair to the rank that owns
    that *label*, so every rank ends up holding the inverse over its own label block;
    then the wanted labels are looked up there exactly as `fetch_rows` looks up rows.
    One extra exchange, once per source, and no identity to assume.
    """
    import h5py
    wanted = np.asarray(wanted, dtype=np.int64)
    size = 1 if comm is None else comm.Get_size()
    with h5py.File(path, "r") as h5:
        dset = h5[dataset]
        n = int(dset.shape[0])
        if size == 1:
            labels = np.asarray(dset, dtype=np.int64)
            check_range(path, dataset, wanted, n, comm)
            _check_labels(path, dataset, labels, n, comm)
            inverse = np.full(n, -1, dtype=np.int64)
            inverse[labels] = np.arange(n, dtype=np.int64)
            _check_covered(path, dataset, inverse, comm)
            return inverse[wanted]
    check_range(path, dataset, wanted, n, comm)
    edges, block = invert_labels(path, dataset, comm)
    return lookup(comm, edges, block, wanted)


def invert_labels(path, dataset, comm):
    """`(edges, block)`: the inverse of the labelling over this rank's block of labels.

    The collective half of `rows_of_labels` -- a block read, the agreed guards and one
    exchange -- returned in the form `lookup` takes, so that a caller asking the same
    labelling many times can invert it once and keep the block.
    """
    rank = comm.Get_rank()
    edges, mine = read_block(path, dataset, comm)
    mine = mine.astype(np.int64, copy=False)
    n = int(edges[-1])

    rows = np.arange(edges[rank], edges[rank + 1], dtype=np.int64)
    _check_labels(path, dataset, mine, n, comm)
    got_labels, got_rows = to_owners(comm, edges, mine, rows)

    block = np.full(int(edges[rank + 1] - edges[rank]), -1, dtype=np.int64)
    block[got_labels - edges[rank]] = got_rows
    _check_covered(path, dataset, block, comm)
    return edges, block


def constant_diff(path, dataset, step, comm=None):
    """Whether consecutive entries differ by `step` throughout, without gathering them.

    `x_cell_dofs` is an offsets array, and a reader that takes `offsets[1] - offsets[0]`
    as the stride is assuming it is uniform. A reader that holds the array can check that
    itself; a reader that declines to hold it needs the same
    assurance for the same reason -- a ragged offsets array with the right total would
    reshape into the right shape and mix cells together. Each rank checks its own block
    and the seam to the next, so the cost is a block read and a reduction.
    """
    import h5py
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    with h5py.File(path, "r") as h5:
        dset = h5[dataset]
        n = int(dset.shape[0])
        edges = block_edges(n, size)
        lo, hi = int(edges[rank]), int(edges[rank + 1])
        # one entry of overlap, so the join between two blocks is checked by the left one
        mine = np.asarray(dset[lo:min(hi + 1, n)])
    ok = bool(len(mine) < 2 or (np.diff(mine) == step).all())
    return not agreed(comm, not ok)


def move_rows(comm, n, index, values, wanted):
    """Rows of a distributed array, by global index, to wherever they are wanted now: each
    rank gives the global indices it holds (`index`, any order, repeats allowed and
    taken to agree) with their rows (`values`), and gets the rows of `wanted`. For data
    that follows a mesh through `periodic_partition.rebuild`: cells and vertices keep
    their global ids, not their ranks. Rows go through the rank whose block of [0, n)
    holds the index (`to_owners`, then `lookup`), so no rank holds the array. Raises
    on every rank if a wanted index is held by none. Collective."""
    values = np.asarray(values)
    if comm is None or comm.Get_size() == 1:
        at = np.full(n, -1, np.int64)
        at[np.asarray(index, np.int64)] = np.arange(len(index))
        if (at[np.asarray(wanted, np.int64)] < 0).any():
            raise ValueError("move_rows: a wanted index is held by no rank")
        return values[at[np.asarray(wanted, np.int64)]]
    edges = block_edges(n, comm.Get_size())
    got_index, got = to_owners(comm, edges, index, values)
    lo, hi = int(edges[comm.Get_rank()]), int(edges[comm.Get_rank() + 1])
    block = np.zeros((hi - lo,) + values.shape[1:], values.dtype)
    held = np.zeros(hi - lo, bool)
    block[got_index - lo], held[got_index - lo] = got, True
    # one request a distinct wanted row, answered with the row and whether it is held
    asked, inverse = unique_inverse(np.asarray(wanted, np.int64))
    routed = route(comm, owner_of(edges, asked, comm), asked)
    local = routed.received[0] - lo
    is_held, rows = reply(comm, routed, held[local], block[local])
    if agreed(comm, not np.all(is_held)):
        raise ValueError("move_rows: a wanted index is held by no rank")
    return rows[inverse]


def flag_first(comm, gid, flag, n):
    """New numbers for the global ids `gid` (0..n-1, repeats allowed, every id present on
    some rank): the ids flagged on any rank first, then the rest, each in id order.
    Returns `(new, n_flagged)`, `new` aligned with `gid`. A slab's final write puts its own
    cells and their vertices first with it."""
    gid = np.asarray(gid, dtype=np.int64)
    flag = np.asarray(flag, dtype=bool)
    size = 1 if comm is None else comm.Get_size()
    if size == 1:
        f = np.zeros(n, bool)
        seen = np.zeros(n, bool)
        f[gid[flag]] = True
        seen[gid] = True
        if not seen.all():
            raise ValueError("{} of {} ids are on no rank".format(int((~seen).sum()), n))
        new = np.empty(n, np.int64)
        k = int(f.sum())
        new[f] = np.arange(k)
        new[~f] = k + np.arange(n - k)
        return new[gid], k
    from mpi4py import MPI
    rank = comm.Get_rank()
    edges = block_edges(n, size)
    lo, hi = int(edges[rank]), int(edges[rank + 1])
    got, got_flag = to_owners(comm, edges, gid, flag.astype(np.int8))
    f = np.zeros(hi - lo, bool)
    seen = np.zeros(hi - lo, bool)
    f[got[got_flag > 0] - lo] = True
    seen[got - lo] = True
    if agreed(comm, not seen.all()):
        raise ValueError("some of {} ids are on no rank".format(n))
    mine = np.array([f.sum(), (~f).sum()], dtype=np.int64)
    before = np.zeros(2, np.int64)
    comm.Exscan(mine, before, op=MPI.SUM)
    if rank == 0:
        before[:] = 0
    total = comm.allreduce(int(mine[0]), op=MPI.SUM)
    block = np.empty(hi - lo, np.int64)
    block[f] = before[0] + np.arange(int(mine[0]))
    block[~f] = total + before[1] + np.arange(int(mine[1]))
    return lookup(comm, edges, block, gid), total


_SERIAL_WARNED = [False]


def mpio(comm):
    """Whether a write on `comm` goes through MPI-IO: more than one rank, and h5py's MPI
    build loaded (Debian's h5py loads it when `OMPI_COMM_WORLD_SIZE` or `MPI_LOCALNRANKS`
    is set, unless `H5PY_NEVER_USE_MPI` is; `H5PY_ALWAYS_USE_MPI` forces it). Without
    it, rank zero writes every rank's rows, in turn, which does not scale: said once, on
    rank 0, with what it costs and how to have MPI-IO."""
    if comm is None or comm.Get_size() == 1:
        return False
    import h5py
    parallel = bool(h5py.get_config().mpi)
    if not parallel and not _SERIAL_WARNED[0]:
        _SERIAL_WARNED[0] = True
        if comm.Get_rank() == 0:
            import sys
            print("femtools: h5py has no MPI support, so rank 0 writes every HDF5 dataset "
                  "for all {} ranks, receiving their compressed rows one rank after "
                  "another while the others wait: each write then takes as long as one "
                  "process writing the whole file, at any number of ranks, and the run's "
                  "output does not scale. Install h5py built with MPI (Debian/Ubuntu: "
                  "python3-h5py-mpi; on a cluster, h5py built against its MPI and a "
                  "parallel HDF5) for every rank to write its own block through MPI-IO."
                  .format(comm.Get_size()), file=sys.stderr, flush=True)
    return parallel


@contextlib.contextmanager
def opened(path, mode, comm=None):
    """`path` opened for what every rank writes alike (groups, attributes, links): by
    every rank, collectively, under MPI-IO (`mpio`); otherwise by rank zero alone, and
    None on the others. Collective."""
    import h5py
    rank = 0 if comm is None else comm.Get_rank()
    if mpio(comm):
        with h5py.File(path, mode, driver="mpio", comm=comm) as h5:
            yield h5    # closed collectively: no rank goes on before it is written
        return
    if rank == 0:
        with h5py.File(path, mode) as h5:
            yield h5
    else:
        yield None
    if comm is not None and comm.Get_size() > 1:
        comm.Barrier()


def collective_write(node, lo, values):
    """One rank's part of a collective dataset write, rows lo.. of `node`, empty parts
    included.

    Every rank must take part in a collective write -- but h5py's `__setitem__`
    returns without calling H5Dwrite when the selection is empty, so a rank with
    nothing to write silently skips the collective the other ranks are inside.
    Rank zero then spins in `H5D__write`'s Allreduce while the rest wait in the
    file-close barrier, all at full CPU, forever. The empty rank instead makes the
    low-level write with a none-selection, which participates in the collective and
    writes nothing.
    """
    if len(values):
        with node.collective:
            node[lo:lo + len(values)] = values
        return
    import h5py
    fspace = node.id.get_space()
    fspace.select_none()
    mspace = h5py.h5s.create_simple((0,) + tuple(node.shape[1:]))
    mspace.select_none()
    dxpl = h5py.h5p.create(h5py.h5p.DATASET_XFER)
    dxpl.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)
    node.id.write(mspace, fspace, np.zeros((0,) + tuple(node.shape[1:]), dtype=node.dtype),
                  dxpl=dxpl)


def write_ordered(path, name, rows, values, n, comm=None, mode="a", attrs=None):
    """Dataset `name` of `path`: row `rows[i]` is `values[i]`, compressed (`compress`).
    Rows may come from several ranks, and the same row more than once (a vertex every
    rank holding it writes); the copies must agree to the bit, and every row 0..n-1 must
    be given. A rank with no rows may pass any empty array: the dtype and row shape are
    those of the ranks with rows (`_agreed_layout`). Under MPI the rows go to blocks of
    whole chunks (one exchange), and every rank writes its own block: collectively
    through MPI-IO (`mpio`), with HDF5 compressing each rank's chunks. Rows that are
    already one contiguous run a rank, in rank order, covering 0..n-1 (a mesh's topology,
    a vector's local range) go to the blocks as they are, with no index sent and nothing
    sorted (`_runs`).

    Without h5py's MPI build each rank compresses its own chunks (`compress.encode`)
    and rank zero writes the stored bytes a rank at a time, in messages of at most
    `MESSAGE` bytes (MPI counts are 32-bit), so no rank ever holds the dataset.
    Collective; the file is opened in `mode` (`opened`), after the rows are checked. A
    failure while writing aborts the job (`abort.aborting`): the other ranks would wait
    for the failed one."""
    from femtools.abort import aborting
    blocks = _blocks(path, name, rows, values, n, comm)
    # the abort inside the file: an MPI-IO file's close is collective, and a rank
    # raising through it would wait there for ranks waiting in the write
    with opened(path, mode, comm) as h5, aborting(comm):
        _store(h5, name, blocks, comm, attrs)


def write_rows(h5, name, rows, values, n, comm=None, attrs=None, plans=None):
    """`write_ordered` into a file already open: `h5` opened by every rank under MPI-IO,
    by rank zero alone otherwise (None on the others), as `opened` gives it -- a writer
    that keeps its file open across many datasets (a time series). Collective.

    `plans`: a dict kept beside `rows` by a caller writing the same rows again and again
    (a time series' nodes, every step), in which the routing of the rows to the ranks
    writing them is kept (`RowPlan`, by the row count and the chunk rows): a later write
    then sends its values alone. `rows` must not change while the dict is kept."""
    from femtools.abort import aborting
    blocks = _blocks(getattr(h5, "filename", ""), name, rows, values, n, comm, plans)
    with aborting(comm):
        _store(h5, name, blocks, comm, attrs)


class RowPlan:
    """How one rank's rows of a dataset reach the blocks of whole chunks their ranks write
    (`compress.aligned_edges`), worked out once for rows written again and again: the
    order that groups the rows by destination (`order`, None when they are already one
    run in rank order, `_runs`), the counts each way, and where each received row goes
    in this rank's block -- `place` when every row of the block arrives once, else `keep`
    and `at` (the first copy of each row, and its place) with `rep_a`, `rep_b` (copies
    to compare with the copy before them, which must agree to the bit). `repeats` (agreed
    over the ranks) says whether any rank has copies to compare. Built by `plan_rows`,
    used by `RowPlan.block`."""

    def __init__(self, lo, hi, order, counts, recv_counts):
        self.lo, self.hi = lo, hi
        self.order, self.counts, self.recv_counts = order, counts, recv_counts
        self.place = self.keep = self.at = self.rep_a = self.rep_b = None
        self.repeats = False
        self.rounds = {}            # row width -> the exchange's rounds (`_send_rows`)

    def block(self, comm, values):
        """`(block, differ)`: this rank's block of rows from every rank's `values` (aligned
        with the rows the plan was made for), and whether two copies of a row differ
        (on any rank). Collective."""
        send = values if self.order is None else values[self.order]
        width = int(np.prod(send.shape[1:])) if send.ndim > 1 else 1
        got, self.rounds[width] = _send_rows(comm, self.counts, self.recv_counts,
                                             np.ascontiguousarray(send),
                                             self.rounds.get(width))
        n = self.hi - self.lo
        differ = False
        if self.order is None:
            block = got
        elif self.place is not None:
            block = np.empty((n,) + got.shape[1:], dtype=got.dtype)
            block[self.place] = got
        else:
            block = np.zeros((n,) + got.shape[1:], dtype=got.dtype)
            if len(self.rep_a):
                flat = as_rows(got).view(np.uint8).reshape(len(got), -1)
                differ = bool((flat[self.rep_a] != flat[self.rep_b]).any())
        # every rank, whatever its own block held: `repeats` is the ranks' agreed flag, the
        # copies are some rank's (a rank whose rows all arrive once took no part before)
        if self.repeats:
            differ = agreed(comm, differ)
        if self.order is not None and self.place is None and not differ:
            block[self.at] = got[self.keep]
        return block, differ


def _local_index(k):
    """The narrowest integer type indexing k entries: a plan's index arrays live as long
    as the rows they route."""
    return np.int32 if k < np.iinfo(np.int32).max else np.int64


def plan_rows(comm, rows, n, step):
    """`(plan, bad)`: the `RowPlan` of this rank's `rows` of an `n`-row dataset of chunks
    of `step` rows, and whether (on any rank) rows are missing or out of range. One
    exchange of the row indices, unless the rows are runs (`_runs`). Collective."""
    from femtools import compress
    rows = np.asarray(rows, dtype=np.int64)
    size, rank = comm.Get_size(), comm.Get_rank()
    edges = compress.aligned_edges(n, size, step)
    lo, hi = int(edges[rank]), int(edges[rank + 1])
    start = _runs(comm, rows, n)
    if start is not None:
        counts = np.clip(np.minimum(start + len(rows), edges[1:])
                         - np.maximum(start, edges[:-1]), 0, None)
        return RowPlan(lo, hi, None, counts, _alltoall_counts(comm, counts)), False
    dest = owner_of(edges, rows, comm)
    order = rank_order(dest, size).astype(_local_index(len(rows)))
    counts = np.bincount(dest, minlength=size).astype(np.int64)
    plan = RowPlan(lo, hi, order, counts, _alltoall_counts(comm, counts))
    r = _send_rows(comm, counts, plan.recv_counts, rows[order])[0] - lo
    m = hi - lo
    index = _local_index(max(len(r), 1))
    if m and len(r) == m and r.min() >= 0 and r.max() < m:
        seen = np.zeros(m, bool)
        seen[r] = True
        if seen.all():
            plan.place = r.astype(index)
    bad = False
    if plan.place is None:
        srt = np.argsort(r, kind="stable")
        rs = r[srt]
        first = np.ones(len(rs), bool)
        first[1:] = rs[1:] != rs[:-1]
        rep = np.nonzero(~first)[0]
        plan.keep, plan.at = srt[first].astype(index), rs[first].astype(index)
        plan.rep_a, plan.rep_b = srt[rep].astype(index), srt[rep - 1].astype(index)
        bad = int(first.sum()) != m or bool(len(rs) and (rs[0] < 0 or rs[-1] >= m))
    plan.repeats = agreed(comm, plan.rep_a is not None and len(plan.rep_a) > 0)
    return plan, agreed(comm, bad)


def _blocks(path, name, rows, values, n, comm, plans=None):
    """The rows moved to the ranks writing them: `(n, shape, dtype, lo, block, step)`,
    this rank's block of whole chunks (`step` rows each) from row `lo`. Refuses, on every
    rank, rows missing or two copies that differ. A plan from `plans` (`write_rows`) is
    used when every rank has one, and one made is kept there. Collective."""
    from femtools import compress
    rows = np.asarray(rows, dtype=np.int64)
    values = np.asarray(values)
    size = 1 if comm is None else comm.Get_size()
    if size > 1:
        dt, shape = _agreed_layout(comm, values, len(rows))
        values = values.astype(dt, copy=False).reshape((len(rows),) + shape)
    shape = values.shape[1:]
    step = compress.chunk_rows((n,) + shape) if n else 1
    if size == 1:
        lo, block, bad = 0, *_place(rows, values, 0, n)
    else:
        # all ranks agree whether a plan is used: any rank without one makes it anew
        key = (int(n), int(step))
        plan = None if plans is None else plans.get(key)
        if plans is not None and agreed(comm, plan is None):
            plan = None
        bad = False
        if plan is None:
            plan, bad = plan_rows(comm, rows, n, step)
            if plans is not None and not bad:
                plans[key] = plan
        lo = plan.lo
        block = None
        if not bad:
            block, bad = plan.block(comm, values)
    if bad:                     # agreed over the ranks (`plan_rows`, `RowPlan.block`)
        raise ValueError("{}:{}: rows missing, or two copies of a row that differ".format(
            path, name))
    return n, shape, values.dtype, lo, block, step


def _store(h5, name, blocks, comm, attrs):
    """`_blocks`' blocks into dataset `name` of `h5` (as `write_rows` takes it)."""
    from femtools import compress
    n, shape, dtype, lo, block, step = blocks
    size = 1 if comm is None else comm.Get_size()
    opts = compress.options((n,) + shape)

    def create():
        d = h5.create_dataset(name, shape=(n,) + shape, dtype=dtype, **opts)
        for k, v in (attrs or {}).items():
            d.attrs[k] = v
        return d

    if mpio(comm):
        d = create()
        if n:               # a zero-size dataset has no storage to write to
            collective_write(d, lo, block)
        return
    if size == 1:
        d = create()
        if n:
            d[...] = block
        return
    stored = compress.encode(block, step) if opts else []
    block = None
    if comm.Get_rank() != 0:
        lengths = [len(b) for b in stored]
        comm.send((lo, lengths), dest=0, tag=81)
        for i, j in _batches(lengths):
            comm.Send(np.frombuffer(b"".join(stored[i:j]), np.uint8), dest=0, tag=82)
        return
    d = create()
    put(d, lo, stored, step)
    for r in range(1, size):
        at, lengths = comm.recv(source=r, tag=81)
        for i, j in _batches(lengths):
            buf = np.empty(sum(lengths[i:j]), np.uint8)
            comm.Recv(buf, source=r, tag=82)
            ends = np.cumsum(lengths[i:j])
            put(d, at + i * step, [buf[e - m:e] for e, m in zip(ends, lengths[i:j])], step)


#: the most bytes one message carries: MPI-3 counts are 32-bit, and an incompressible
#: column (a slab's 128-bit identities) reaches 2 GiB at 134M rows on one rank
MESSAGE = 1 << 30


def _batches(lengths, limit=None):
    """`(i, j)` runs of consecutive chunks each at most `limit` bytes (a larger chunk
    alone), covering them all; sender and receiver compute the same runs."""
    limit = MESSAGE if limit is None else limit
    out, i, total = [], 0, 0
    for k, m in enumerate(lengths):
        if k > i and total + m > limit:
            out.append((i, k))
            i, total = k, 0
        total += m
    if i < len(lengths):
        out.append((i, len(lengths)))
    return out


def put(d, lo, stored, step):
    """Stored chunks (`compress.encode`) into dataset `d` from row `lo`, a chunk boundary."""
    zeros = (0,) * (d.ndim - 1)
    for k, b in enumerate(stored):
        d.id.write_direct_chunk((lo + k * step,) + zeros, b)


def _runs(comm, rows, n):
    """This rank's first row when every rank's `rows` are one contiguous run (`first +
    arange(k)`) and the runs, in rank order, are 0..n-1 exactly; else None, on every rank.
    One Exscan of the run lengths (where each run must start) and one Allreduce (whether
    any rank's does not, and the total)."""
    from mpi4py import MPI
    rows = np.asarray(rows)
    k = len(rows)
    run = k == 0 or bool(rows[0] >= 0 and rows[-1] - rows[0] == k - 1
                         and (np.diff(rows) == 1).all())
    before = np.zeros(1, dtype=np.int64)
    comm.Exscan(np.array([k], dtype=np.int64), before, op=MPI.SUM)
    at = 0 if comm.Get_rank() == 0 else int(before[0])
    ok = run and (k == 0 or int(rows[0]) == at)
    bad, total = _int_reduce(comm, [int(not ok), k], "sum")
    return at if bad == 0 and total == n else None


def _place(rows, values, lo, hi):
    """`(block, bad)`: rows lo..hi-1 filled from `(rows, values)`; bad when a row is
    missing or two copies differ. Rows that are each row once, the usual case, are
    placed without sorting."""
    n = hi - lo
    if n and len(rows) == n:
        r = rows - lo
        if r.min() >= 0 and r.max() < n:
            seen = np.zeros(n, bool)
            seen[r] = True
            if seen.all():
                block = np.empty((n,) + values.shape[1:], dtype=values.dtype)
                block[r] = values
                return block, False
    block = np.zeros((n,) + values.shape[1:], dtype=values.dtype)
    if n == 0:
        return block, len(rows) > 0
    order = np.argsort(rows, kind="stable")
    r, v = rows[order] - lo, values[order]
    first = np.ones(len(r), bool)
    first[1:] = r[1:] != r[:-1]
    # a repeat must be its row's first copy to the bit: compared where repeats are
    rep = np.nonzero(~first)[0]
    differ = False
    if len(rep):
        flat = as_rows(v).view(np.uint8).reshape(len(r), -1)
        differ = bool((flat[rep] != flat[rep - 1]).any())
    bad = differ or int(first.sum()) != hi - lo \
        or bool(len(r) and (r[0] < 0 or r[-1] >= hi - lo))
    if not bad:
        block[r[first]] = v[first]
    return block, bad


def _mesh_datasets(mesh_file):
    """`(h5 path, topology, coordinates, cell ids or None)` of a mesh file: dolfin's HDF5
    layout (`mesh/...`, cell ids `mesh/cell_indices`), or an XDMF mesh's two DataItems
    (the meshers', through meshio: the HDF5 beside it; a cell's id is its row). Both
    backends' mesh readers take either (`femtools.periodic_partition.read_partitioned`,
    `femtools.dolfinx.mesh.read_mesh`)."""
    if not mesh_file.endswith('.xdmf'):
        return mesh_file, 'mesh/topology', 'mesh/coordinates', 'mesh/cell_indices'
    import xml.etree.ElementTree as ET
    root = ET.parse(mesh_file).getroot()
    where = {}
    for kind in ('Geometry', 'Topology'):
        item = root.find('.//{}/DataItem'.format(kind))
        if item is None or item.get('Format') != 'HDF':
            raise ValueError('{}: no HDF {} DataItem'.format(mesh_file, kind))
        where[kind] = item.text.strip().split(':', 1)
    (h5a, topo), (h5b, coords) = where['Topology'], where['Geometry']
    if h5a != h5b:
        raise ValueError('{}: topology and geometry in different files'.format(mesh_file))
    return os.path.join(os.path.dirname(mesh_file), h5a), topo, coords, None


def _xdmf_celltype(mesh_file):
    """The cell type of an XDMF mesh's Topology ('tetrahedron', 'triangle', ...)."""
    import xml.etree.ElementTree as ET
    topo = ET.parse(mesh_file).getroot().find('.//Topology')
    if topo is None or not topo.get('TopologyType'):
        raise ValueError('{}: no TopologyType'.format(mesh_file))
    return topo.get('TopologyType').lower()
