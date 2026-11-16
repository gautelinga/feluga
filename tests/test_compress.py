"""`femtools/compress.py`: every dataset written compressed, and existing files rewritten so.

    python3 -m pytest tests/test_compress.py

numpy and h5py only:

- a chunk `encode` makes, written as it is, reads back as the array, and is the very
  bytes HDF5's own filters store;
- `aligned_edges` cuts on chunk boundaries and covers every row;
- `rewrite` keeps every dataset, attribute (a NaN too) and the mtime, compresses what was
  plain, leaves what was compressed, skips links and hard-linked files, and a second run
  does nothing;
- `hdf5_rows.write_ordered` writes compressed.
"""
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from femtools import compress  # noqa: E402
from femtools.hdf5_rows import write_ordered  # noqa: E402


def arrays():
    rng = np.random.default_rng(0)
    return [rng.integers(0, 1 << 40, (300_001, 4)), rng.random((250_003, 3)),
            rng.integers(-5, 5, 2_100_000).astype(np.int16), rng.random(1_234_567) > .5]


@pytest.mark.parametrize("k", range(4))
def test_an_encoded_chunk_is_what_hdf5_stores(tmp_path, k):
    a = arrays()[k]
    rows = compress.chunk_rows(a.shape)
    with h5py.File(tmp_path / "x.h5", "w") as h:
        d = h.create_dataset("direct", shape=a.shape, dtype=a.dtype, **compress.options(a.shape))
        for i, b in enumerate(compress.encode(a, rows)):
            d.id.write_direct_chunk((i * rows,) + (0,) * (a.ndim - 1), b)
        compress.put(h, "filtered", a)
    with h5py.File(tmp_path / "x.h5", "r") as h:
        assert np.array_equal(h["direct"][()], a)
        d, f = h["direct"], h["filtered"]
        for i in range(-(-len(a) // rows)):
            at = (i * rows,) + (0,) * (a.ndim - 1)
            assert d.id.read_direct_chunk(at) == f.id.read_direct_chunk(at)


@pytest.mark.parametrize("n, size, rows", [(10, 3, 4), (100, 3, 7), (3, 4, 8), (0, 2, 1),
                                           (1 << 20, 7, 1 << 16)])
def test_aligned_edges(n, size, rows):
    e = compress.aligned_edges(n, size, rows)
    assert len(e) == size + 1 and e[0] == 0 and e[-1] == n and (np.diff(e) >= 0).all()
    assert all(x % rows == 0 for x in e[:-1])


def test_rewrite_keeps_everything_and_compresses_the_plain(tmp_path):
    p = str(tmp_path / "f.h5")
    a = arrays()
    with h5py.File(p, "w") as h:
        h.attrs["eps"] = np.nan
        h.attrs["name"] = np.bytes_("tetrahedron")
        g = h.create_group("mesh")
        g.attrs["partition"] = np.array([0, 5], np.uint64)
        g.create_group("empty").attrs["s"] = "text"
        g["topology"] = a[0]
        g["topology"].attrs["celltype"] = np.bytes_("tetrahedron")
        g["coordinates"] = a[1]
        compress.put(h, "already", a[2])
        h["scalar"] = 3.5
        h["none"] = np.zeros(0)
    os.utime(p, (1e9, 1e9))
    code, what = compress.rewrite(p)
    assert code == 0, what
    assert os.path.getmtime(p) == 1e9 and not os.path.exists(p + compress.SUFFIX)
    with h5py.File(p, "r") as h:
        assert np.isnan(h.attrs["eps"]) and h.attrs["name"] == b"tetrahedron"
        assert list(h["mesh"].attrs["partition"]) == [0, 5]
        assert h["mesh/empty"].attrs["s"] == "text"
        assert np.array_equal(h["mesh/topology"][()], a[0])
        assert h["mesh/topology"].attrs["celltype"] == b"tetrahedron"
        assert np.array_equal(h["mesh/coordinates"][()], a[1])
        assert np.array_equal(h["already"][()], a[2])
        assert h["scalar"][()] == 3.5 and h["none"].shape == (0,)
        for n in ("mesh/topology", "mesh/coordinates", "already"):
            assert h[n].compression == "gzip" and h[n].shuffle
    assert compress.rewrite(p) == (compress.SKIPPED, "nothing to compress")


def test_rewrite_skips_links(tmp_path):
    p = str(tmp_path / "f.h5")
    with h5py.File(p, "w") as h:
        h["x"] = np.arange(10)
    os.symlink(p, str(tmp_path / "link.h5"))
    os.link(p, str(tmp_path / "hard.h5"))
    assert compress.rewrite(str(tmp_path / "link.h5"))[0] == compress.SKIPPED
    assert compress.rewrite(p)[0] == compress.SKIPPED


def test_write_ordered_compresses(tmp_path):
    p = str(tmp_path / "o.h5")
    a = arrays()[0]
    rows = np.random.default_rng(1).permutation(len(a))
    write_ordered(p, "t", rows, a[rows], len(a), mode="w")
    with h5py.File(p, "r") as h:
        assert np.array_equal(h["t"][()], a) and h["t"].compression == "gzip"


def test_rewrite_keeps_maxshape_fill_and_stored_chunks(tmp_path):
    p = str(tmp_path / "r.h5")
    a = arrays()
    with h5py.File(p, "w") as h:
        h.create_dataset("growing", data=a[1], maxshape=(None, 3), chunks=(4096, 3),
                         fillvalue=-7.0)
        h.create_dataset("lzf", data=a[0], chunks=(8192, 4), compression="lzf")
        h.create_dataset("empty", shape=(0,), maxshape=(None,), dtype="f8", chunks=(16,))
    with h5py.File(p, "r") as h:
        before = [(m, b) for _, m, b in compress._chunks(h["lzf"])]
    assert compress.rewrite(p)[0] == 0
    with h5py.File(p, "r") as h:
        g = h["growing"]
        assert g.maxshape == (None, 3) and g.fillvalue == -7.0 and g.compression == "gzip"
        assert np.array_equal(g[()], a[1])
        assert h["lzf"].compression == "lzf" and np.array_equal(h["lzf"][()], a[0])
        assert [(m, b) for _, m, b in compress._chunks(h["lzf"])] == before
        assert h["empty"].maxshape == (None,) and h["empty"].shape == (0,)
    with h5py.File(p, "r+") as h:
        h["growing"].resize((len(a[1]) + 10, 3))        # still resizable
        assert (h["growing"][-10:] == -7.0).all()       # the new rows take the fill value


def test_rewrite_skips_a_file_with_little_plain(tmp_path):
    p = str(tmp_path / "s.h5")
    with h5py.File(p, "w") as h:
        compress.put(h, "big", arrays()[1])
        h["offsets"] = np.arange(1000)
    code, what = compress.rewrite(p)
    assert code == compress.SKIPPED and what.startswith("nothing worth compressing"), what


@pytest.mark.mpi
@pytest.mark.parametrize("h5py_build", ["mpio", "gather"])
def test_write_ordered_under_mpi(tmp_path, h5py_build):
    """3 ranks, rows scattered and repeated: the file is the array, compressed
    (tests/compress_worker.py) -- every rank writing its block through MPI-IO, or, with
    h5py's serial build, rank zero writing them all in messages capped at 64 KiB."""
    import sys
    pytest.importorskip("mpi4py")
    from tests import support
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compress_worker.py")
    # the tree's femtools, not an installed one (the worker's own folder is tests/)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {"mpio": {"H5PY_ALWAYS_USE_MPI": "1", "PYTHONPATH": root},
           "gather": {"H5PY_NEVER_USE_MPI": "1", "PYTHONPATH": root}}
    proc = support.run([sys.executable, worker, str(tmp_path)], 3, timeout=300,
                       env=env[h5py_build])
    assert proc.stdout.strip().splitlines() == [h5py_build, "ok"], proc.stdout + proc.stderr


@pytest.mark.mpi
@pytest.mark.parametrize("h5py_build", ["mpio", "gather"])
def test_repeats_in_one_block_only(tmp_path, h5py_build):
    """2 ranks, a repeated row in rank 0's block alone: every rank takes the same
    collectives, and the file is the array (tests/repeats_worker.py)."""
    import sys
    pytest.importorskip("mpi4py")
    from tests import support
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repeats_worker.py")
    # the tree's femtools, not an installed one (the worker's own folder is tests/)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {"mpio": {"H5PY_ALWAYS_USE_MPI": "1", "PYTHONPATH": root},
           "gather": {"H5PY_NEVER_USE_MPI": "1", "PYTHONPATH": root}}
    proc = support.run([sys.executable, worker, str(tmp_path)], 2, timeout=120,
                       env=env[h5py_build])
    assert proc.stdout.strip().splitlines()[-1:] == ["ok"], proc.stdout + proc.stderr


@pytest.mark.mpi
@pytest.mark.parametrize("h5py_build,vlen", [("mpio", False), ("mpio", True),
                                             ("gather", False)])
def test_rewrite_under_mpi(tmp_path, h5py_build, vlen):
    """3 ranks rewrite a file: through MPI-IO every rank its own blocks, or rank zero
    alone (h5py's serial build, or a variable-length dataset, which parallel HDF5 cannot
    write); everything kept, the plain compressed (tests/rewrite_worker.py)."""
    import sys
    pytest.importorskip("mpi4py")
    from tests import support
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rewrite_worker.py")
    # the tree's femtools, not an installed one (the worker's own folder is tests/)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {"mpio": {"H5PY_ALWAYS_USE_MPI": "1", "PYTHONPATH": root},
           "gather": {"H5PY_NEVER_USE_MPI": "1", "PYTHONPATH": root}}
    proc = support.run([sys.executable, worker, str(tmp_path)] + (["vlen"] if vlen else []),
                       3, timeout=300, env=env[h5py_build])
    assert proc.stdout.strip().splitlines() == [h5py_build, "0", "ok"], proc.stdout + proc.stderr


class _World:
    """Rank `rank` of ranks holding `rows` each, for `_runs` without MPI: its Exscan and
    Allreduce (sums), computed from what every rank's `_runs` contributes."""

    def __init__(self, rows, rank, n, capture=False):
        self.rows, self.rank, self.n, self.capture = rows, rank, n, capture
        self.sent = None

    def Get_rank(self):
        return self.rank

    def Exscan(self, send, recv, op):
        recv[...] = sum(len(r) for r in self.rows[:self.rank])

    def Allreduce(self, send, recv, op):
        from femtools.hdf5_rows import _runs
        if self.capture:
            self.sent = send.copy()
            recv[...] = send
            return
        total = send.copy()
        for q, rows in enumerate(self.rows):
            if q != self.rank:
                other = _World(self.rows, q, self.n, capture=True)
                _runs(other, rows, self.n)
                total += other.sent
        recv[...] = total


def test_runs_are_recognised_only_when_they_tile_the_rows():
    """`_runs`: every rank one contiguous run, in rank order, 0..n-1 exactly (a rank may
    hold none) -- anything else is the general path, on every rank."""
    from femtools.hdf5_rows import _runs

    def runs(rows, n):
        rows = [np.asarray(r, dtype=np.int64) for r in rows]
        return [_runs(_World(rows, q, n), rows[q], n) for q in range(len(rows))]

    a = np.arange
    assert runs([a(0, 5), a(5, 9), a(0), a(9, 12)], 12) == [0, 5, 9, 9]
    assert runs([a(0, 5), a(5, 9)], 10) == [None, None]                  # short of n
    assert runs([a(5, 9), a(0, 5)], 9) == [None, None]                   # out of order
    assert runs([a(0, 5), a(6, 10)], 10) == [None, None]                 # a gap
    assert runs([a(0, 5), [5, 7, 6, 8]], 9) == [None, None]              # not a run
    assert runs([a(0, 3), [3, 5, 4]], 6) == [None, None]
    assert runs([a(0, 3), a(3, 6)], 6) == [0, 3]


def test_place_is_what_it_was():
    """`_place` compares only the repeats and counts distinct rows from the sort, and
    places rows given once each without one: the same block and verdict as comparing
    every neighbour and `np.unique`, on rows given once, shuffled, repeated, missing,
    repeated with other bits, and out of range."""
    from femtools.hdf5_rows import _place, as_rows

    def before(rows, values, lo, hi):
        block = np.zeros((hi - lo,) + values.shape[1:], dtype=values.dtype)
        if hi - lo == 0:
            return block, len(rows) > 0
        order = np.argsort(rows, kind="stable")
        r, v = rows[order] - lo, values[order]
        first = np.ones(len(r), bool)
        first[1:] = r[1:] != r[:-1]
        flat = as_rows(v)
        same = np.ones(len(r), bool)
        same[1:] = (flat[1:].view(np.uint8).reshape(len(r) - 1, -1)
                    == flat[:-1].view(np.uint8).reshape(len(r) - 1, -1)).all(axis=1) \
            if len(r) > 1 else same[1:]
        bad = bool((~first & ~same).any()) or len(np.unique(r)) != hi - lo \
            or (len(r) and (r.min() < 0 or r.max() >= hi - lo))
        if not bad:
            block[r[first]] = v[first]
        return block, bad

    rng = np.random.default_rng(4)
    a = rng.random((50, 3))
    cases = [np.arange(10, 30), np.r_[np.arange(10, 30), [12, 12, 29]], np.arange(11, 30),
             np.r_[np.arange(10, 30), [40]], np.zeros(0, np.int64),
             # as many rows as the block: each once (placed without a sort), one
             # repeated in place of another, one out of range
             rng.permutation(np.arange(10, 30)), np.r_[np.arange(10, 29), [12]],
             np.r_[np.arange(11, 30), [40]]]
    for rows in cases:
        for lo, hi in ((10, 30), (0, 0)):
            got, want = _place(rows, a[rows], lo, hi), before(rows, a[rows], lo, hi)
            assert got[1] == want[1] and np.array_equal(got[0], want[0]), (rows, lo, hi)
    rows = np.r_[np.arange(10, 30), [15]]
    values = a[rows].copy()
    values[-1, 0] += 1e-16 if values[-1, 0] < 1 else -1e-16
    assert _place(rows, values, 10, 30)[1] and before(rows, values, 10, 30)[1]
