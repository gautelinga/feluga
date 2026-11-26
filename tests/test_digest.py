"""`feluga.digest`: what a stored mesh or Function is, as a SHA-256.

    python3 -m pytest tests/test_digest.py

The mesh digest is the arrays, not the file (another attribute, the same value; another
coordinate, another value), whatever the block, and a file that is no mesh is refused;
the Function digest is each dataset's row blocks and then their digests in order, so
every rank of any rank count gets the serial value having hashed only its own blocks,
and another vector is another value (the block size is part of the value: a stored digest
names it). numpy and h5py.
"""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from feluga.digest import function_digest, mesh_digest  # noqa: E402


def write_mesh(path, shift=0.0):
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    x[4, 0] += shift
    with h5py.File(path, "w") as h5:
        h5["mesh/coordinates"] = x
        h5["mesh/topology"] = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], np.int64)
        h5["mesh/cell_indices"] = np.array([0, 1], np.int64)


def test_the_mesh_digest_is_the_mesh_not_the_file(tmp_path):
    a, b, c = (str(tmp_path / n) for n in ("a.h5", "b.h5", "c.h5"))
    write_mesh(a)
    write_mesh(b)
    write_mesh(c, shift=1e-12)
    with h5py.File(b, "a") as h5:
        h5.attrs["written_by"] = "another run"   # different bytes, same mesh
    assert mesh_digest(a) == mesh_digest(b) == mesh_digest(a, block=1)
    assert mesh_digest(c) != mesh_digest(a)
    with h5py.File(str(tmp_path / "x.h5"), "w") as h5:
        h5["something"] = np.zeros(3)
    with pytest.raises(ValueError, match="not a dolfin mesh: no mesh/coordinates or "
                                         "mesh/topology"):
        mesh_digest(str(tmp_path / "x.h5"))


class _Gathered(Exception):
    pass


class _Rank:
    """Rank `rank` of `size` in one process. Without `parts` it keeps the blocks this rank
    hashed and stops at the gather; with them, the gather hands back every rank's, as
    `allgather` would."""
    def __init__(self, size, rank, parts=None):
        self.size, self.rank, self.parts, self.mine = size, rank, parts, None

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

    def allgather(self, mine):
        if self.parts is None:
            self.mine = mine
            raise _Gathered
        return [self.parts[q] for q in range(self.size)]


def test_the_function_digest_is_its_blocks_in_order_on_any_rank_count(tmp_path):
    rng = np.random.default_rng(0)
    path = str(tmp_path / "u.h5")
    with h5py.File(path, "w") as h5:
        h5["u/vector_0"] = rng.random(300)
        h5["u/cell_dofs"] = rng.integers(0, 300, 30 * 20).astype(np.int32)
        h5["u/x_cell_dofs"] = (30 * np.arange(21)).astype(np.uint64)
        h5["u/cells"] = rng.permutation(20).astype(np.uint64)
        h5["v/vector_0"] = h5["u/vector_0"][:] * (1 + 1e-15)
        for name in ("cell_dofs", "x_cell_dofs", "cells"):
            h5["v/" + name] = h5["u/" + name][:]
    with h5py.File(path, "r") as f:
        serial = function_digest(f["u"], block=7)
        assert len(serial) == 64
        for P in (2, 3, 5):
            parts = {}
            for r in range(P):
                comm = _Rank(P, r)
                with pytest.raises(_Gathered):
                    function_digest(f["u"], comm, block=7)
                assert all(j % P == r for j in comm.mine)
                parts[r] = comm.mine
            for r in range(P):
                assert function_digest(f["u"], _Rank(P, r, parts), block=7) == serial
        assert function_digest(f["v"], block=7) != serial


def _golden(path):
    with h5py.File(path, "w") as h5:
        h5["u/vector_0"] = np.arange(30, dtype=np.float64) / 7.0
        h5["u/cell_dofs"] = (np.arange(60, dtype=np.int32) * 7) % 30
        h5["u/x_cell_dofs"] = (30 * np.arange(3)).astype(np.uint64)
        h5["u/cells"] = np.array([1, 0], dtype=np.uint64)
        h5["mesh/coordinates"] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                           [1, 1, 1]], float)
        h5["mesh/topology"] = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], np.int64)
        h5["mesh/cell_indices"] = np.array([0, 1], np.int64)


def test_the_digests_are_the_ones_files_hold(tmp_path):
    """Literal values, computed by the code these replaced (stokes-beadpack a7cc6c0's
    `slabs.order.velocity_digest` and `io.rung.mesh_digest`, 2026-11-15): digests already
    stored in slab and ladder files keep verifying."""
    path = str(tmp_path / "g.h5")
    _golden(path)
    with h5py.File(path, "r") as f:
        assert function_digest(f["u"]) == \
            "28ec522e5cff7f6b5159a89d39083fcdeda343f8188852e3f13ee86aacaf3510"
        assert function_digest(f["u"], block=4) == \
            "5074664617820a495a1e3949a93446085c845d82f93ad01c155c92889312397e"
    assert mesh_digest(path) == \
        "8039a5754b935e3c7278ba6c6b597867277e8d3bac38e1002a691d564e0dbf70"
