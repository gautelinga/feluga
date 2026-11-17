"""`hdf5_rows.rank_of`, the one hash that picks the rank a row of integers meets at,
`rank_order`, the sort that groups rows by rank, and `unique_rows` and `sorted_unique`,
np.unique by sorts (NumPy 2.3 and later hash a plain np.unique of ints).

    python3 -m pytest tests/test_routing.py

numpy only.
"""
import numpy as np

from femtools.hdf5_rows import rank_of


def test_rows_of_any_size_meet_at_one_rank_each_spread_evenly():
    """Rows of ids past 2^40 (a face's three vertex ids at h0.012 broke the join's check when
    it borrowed the slab identities' 40- and 21-bit packing, 2026-10-16): the same row to the
    same rank, every rank in range and used about equally, the columns' order mattering."""
    rng = np.random.default_rng(0)
    rows = rng.integers(1 << 40, 1 << 62, size=(64000, 3))
    dest = rank_of(rows, 64)
    assert dest.dtype == np.int64 and dest.min() >= 0 and dest.max() < 64
    assert np.array_equal(dest, rank_of(rows.copy(), 64))
    counts = np.bincount(dest, minlength=64)
    assert counts.min() > 0.85 * counts.mean() and counts.max() < 1.15 * counts.mean()
    assert not np.array_equal(dest, rank_of(rows[:, ::-1], 64))
    small = np.arange(3000).reshape(1000, 3)            # consecutive small ids spread too
    assert len(np.unique(rank_of(small, 7))) == 7
    assert rank_of(np.zeros((0, 3), np.int64), 4).shape == (0,)
    assert np.array_equal(rank_of(rows, 1), np.zeros(len(rows), np.int64))


def test_rank_order_is_the_stable_argsort():
    """`rank_order` sorts by 16-bit ranks where they fit: the permutation of the 64-bit
    stable argsort, at one rank, a few, a node's and the 16-bit edge, and past it."""
    from femtools.hdf5_rows import rank_order
    rng = np.random.default_rng(1)
    for size in (1, 3, 256, 32768, 40000):
        dest = rng.integers(0, size, 20000)
        assert np.array_equal(rank_order(dest, size), np.argsort(dest, kind="stable"))
    assert rank_order(np.zeros(0, np.int64), 4).shape == (0,)


def test_unique_rows_is_numpys():
    """`hdf5_rows.unique_rows` is `np.unique(axis=0)`, the same rows in the same order,
    on rows that pack into one key (vertex numbers, -1 for no second end, ranks) and on
    rows whose ranges do not."""
    from femtools.hdf5_rows import unique_rows as _unique_rows
    rng = np.random.default_rng(2)
    for top in (5, 1 << 20, 1 << 40):
        rows = np.stack([rng.integers(0, 2, 5000), rng.integers(0, top, 5000),
                         rng.integers(-1, top, 5000), rng.integers(0, 300, 5000)], 1)
        rows = np.concatenate([rows, rows[::3]]).astype(np.int64)
        assert np.array_equal(_unique_rows(rows), np.unique(rows, axis=0))
    assert _unique_rows(np.zeros((0, 4), np.int64)).shape == (0, 4)


def test_sorted_unique_is_numpys():
    """`hdf5_rows.sorted_unique` is a plain `np.unique` of ints: the same values, sorted,
    of the same type, for few and many distinct values, int32 and int64, a 2-D array
    (flattened), one value and none."""
    from femtools.hdf5_rows import sorted_unique
    rng = np.random.default_rng(3)
    for top in (3, 1 << 20, 1 << 40):
        for dtype in (np.int32, np.int64):
            a = rng.integers(-top // 2, top, 20000).astype(dtype)
            got = sorted_unique(a)
            assert got.dtype == a.dtype and np.array_equal(got, np.unique(a))
    a = rng.integers(0, 50, (40, 7))
    assert np.array_equal(sorted_unique(a), np.unique(a))
    for a in (np.array([4]), np.zeros(0, np.int64)):
        assert np.array_equal(sorted_unique(a), np.unique(a))
