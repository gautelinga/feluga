"""`femtools.xdmfread`: datasets read with their gzip chunks inflated on a thread pool,
whole and a chunk-row at a time, give what meshio wrote."""
import numpy as np
import pytest


def test_xdmfread_reads_what_meshio_reads(tmp_path):
    meshio = pytest.importorskip("meshio")

    from femtools import xdmfread
    rng = np.random.default_rng(4)
    pts, cells = rng.random((5000, 3)), rng.integers(0, 5000, (20000, 4)).astype(np.int32)
    meshio.write(tmp_path / 'm.xdmf', meshio.Mesh(pts, [('tetra', cells)]))
    it = xdmfread.items(tmp_path / 'm.xdmf')
    pool = xdmfread.pool(4)
    tets = xdmfread.open_item(it, 'topology')
    assert np.array_equal(xdmfread.read(xdmfread.open_item(it, 'geometry'), pool), pts)
    assert np.array_equal(xdmfread.read(tets, pool), cells)
    blocks = [b for _, b in xdmfread.row_blocks(tets, pool, ahead=3)]
    assert np.array_equal(np.concatenate(blocks), cells)


@pytest.mark.parametrize("layout", ["femtools", "meshio"])
@pytest.mark.parametrize("shape,dtype", [((100_003,), "f8"), ((70_001, 3), "f8"),
                                         ((50_000, 4), "i8"), ((9_999, 10), "i4")])
def test_the_fast_path_reads_what_h5py_reads(tmp_path, layout, shape, dtype):
    """Shuffled and gzipped by `femtools.compress` (whole rows a chunk), or gzipped a
    column a chunk as meshio writes: the chunks inflated here are h5py's arrays, read
    whole or a chunk-row at a time however far ahead -- 0 included."""
    import h5py
    from femtools import compress, xdmfread
    rng = np.random.default_rng(7)
    a = (rng.random(shape) * 1000).astype(dtype)
    path = str(tmp_path / "f.h5")
    with h5py.File(path, "w") as h:
        if layout == "femtools":
            compress.put(h, "x", a)
        else:
            chunks = (4096,) if a.ndim == 1 else (4096, 1)
            h.create_dataset("x", data=a, chunks=chunks, compression="gzip")
    pool = xdmfread.pool(2)
    with h5py.File(path, "r") as h:
        d = h["x"]
        assert xdmfread._raw_ok(d)
        assert np.array_equal(xdmfread.read(d, pool), a)
        if a.ndim == 2:
            for ahead in (0, 2):
                blocks = list(xdmfread.row_blocks(d, pool, ahead))
                assert np.array_equal(np.vstack([b for _, b in blocks]), a)
                assert [s for s, _ in blocks] == list(range(0, len(a), d.chunks[0]))


def test_merge_series_keeps_a_grid_before_the_series(tmp_path):
    """`xdmf.merge_series`: the first file's mesh grid, written before its series, stays;
    its steps from the second file's first time on are the second file's; the second
    .xdmf is gone and the merged file keeps its DOCTYPE."""
    import numpy as np
    from femtools.xdmf import DOCTYPE, XdmfDocument, grids, merge_series

    def series(path, times, static):
        doc = XdmfDocument(str(path))
        if static:
            g = doc.grid('mesh')
            doc.topology(g, None, 'Triangle', ref=doc.dataset(
                'Mesh/mesh/topology', np.zeros((1, 3), np.int32)), shape=(1, 3))
            doc.geometry(g, None, ref=doc.dataset('Mesh/mesh/geometry', np.zeros((3, 2))),
                         shape=(3, 2))
        for t in times:
            g = doc.grid('mesh', doc.collection())
            doc.topology(g, None, 'Triangle', ref='x.h5:/t', shape=(1, 3))
            doc.geometry(g, None, ref='x.h5:/g', shape=(3, 2))
            doc.time(g, t)
            doc.attribute(g, 'u', np.zeros((3, 1)), ref='x.h5:/u%g' % t, shape=(3, 1))
        doc.close()
    a, b = tmp_path / 'a.xdmf', tmp_path / 'b.xdmf'
    series(a, [0, 1, 2], True)
    series(b, [1.5, 2.5], False)
    merge_series([str(a), str(b)])
    assert [g['time'] for g in grids(str(a))] == [None, 0.0, 1.0, 1.5, 2.5]
    assert not b.exists() and DOCTYPE in a.read_text()
