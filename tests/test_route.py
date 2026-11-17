"""`hdf5_rows.route`, `reply`, `exchange`: the one rank-to-rank row exchange.

    python3 -m pytest tests/test_route.py

Under `mpirun -n 3` (needs mpi4py and mpirun): rows of several arrays
and dtypes reach the ranks they name, grouped by source, and answers come back aligned
with the rows asked -- in one round and forced into many (FEMTOOLS_ROUND), the
same bits; a rank with no rows may pass an empty array of another dtype; ranks passing
rows of different dtypes, and a destination outside the ranks, are refused on every rank
(the job ends with the message rather than hanging).
"""
import shutil
import subprocess
import sys

import pytest

from tests import support


WORKER = r"""
import sys
import numpy as np
from mpi4py import MPI
from femtools.hdf5_rows import reply, route, to_owners, block_edges

def main(argv):
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    kind = sys.argv[1]
    rng = np.random.default_rng(rank)
    if kind == 'ok':
        n = [7, 0, 2000][rank]                       # rank 1 sends nothing
        dest = rng.integers(0, size, n)
        ids = np.arange(n, dtype=np.int64) + 10000 * rank
        vec = np.stack([ids * 0.5, ids * 0.25, -ids * 1.0], axis=1)    # (n, 3) float64
        tag = (ids % 7).astype(np.int8)
        if rank == 1:
            vec = np.zeros(0)                         # an empty default of another shape
            tag = np.zeros(0, np.float64)             # ... and dtype
        r = route(comm, dest, ids, vec, tag)
        got_ids, got_vec, got_tag = r.received
        assert got_vec.dtype == np.float64 and got_vec.shape[1:] == (3,)
        assert got_tag.dtype == np.int8
        assert np.array_equal(got_vec[:, 0], got_ids * 0.5)
        assert np.array_equal(got_tag, (got_ids % 7).astype(np.int8))
        src = np.repeat(np.arange(size), r.recv_counts)
        assert np.array_equal(src, got_ids // 10000)                  # grouped by source
        # every row this rank was sent, and no other
        everything = comm.allgather((ids, dest))
        mine = np.sort(np.concatenate([i[d == rank] for i, d in everything]))
        assert np.array_equal(np.sort(got_ids), mine)
        back, = reply(comm, r, got_ids * 3)
        assert np.array_equal(back, ids * 3)                           # aligned with ours
        edges = block_edges(30000, size)
        got_i, got_v = to_owners(comm, edges, ids, vec if rank != 1 else np.zeros((0, 3)))
        lo, hi = edges[rank], edges[rank + 1]
        assert ((got_i >= lo) & (got_i < hi)).all()
        print('RESULT', rank, int(np.sum(got_ids)), float(np.sum(got_vec)), flush=True)
    elif kind == 'dtype':
        vals = np.zeros(3, np.float64 if rank == 0 else np.int64)
        route(comm, np.zeros(3, np.int64), vals)
    elif kind == 'dest':
        route(comm, np.array([size if rank == 2 else 0]), np.zeros(1))
    return 0

try:
    sys.exit(main(sys.argv))
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    MPI.COMM_WORLD.Abort(1)
"""


def mpirun(tmp_path, kind, env_extra=None):
    pytest.importorskip("mpi4py")
    if not shutil.which("mpirun"):
        pytest.skip("no mpirun on this machine")
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    env = support.clean_env()
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = support.pythonpath()
    env.update(env_extra or {})
    return subprocess.run(["mpirun", "-n", "3", sys.executable, str(script), kind],
                          capture_output=True, text=True, timeout=120, env=env)


def results(proc):
    return sorted(line for line in proc.stdout.splitlines() if line.startswith("RESULT"))


@pytest.mark.mpi
def test_rows_reach_their_ranks_and_answers_come_back(tmp_path):
    one = mpirun(tmp_path, "ok")
    assert one.returncode == 0, one.stdout + one.stderr
    many = mpirun(tmp_path, "ok", {"FEMTOOLS_ROUND": "7"})   # hundreds of rounds
    assert many.returncode == 0, many.stdout + many.stderr
    assert results(one) == results(many) and len(results(one)) == 3


@pytest.mark.mpi
@pytest.mark.parametrize("kind, text", [("dtype", "different dtypes"),
                                        ("dest", "a destination outside the 3 ranks")])
def test_a_bad_exchange_is_refused_on_every_rank(tmp_path, kind, text):
    proc = mpirun(tmp_path, kind)
    assert proc.returncode != 0 and text in proc.stdout + proc.stderr, proc.stderr[-2000:]


def test_no_communicator_is_one_rank():
    """route, reply and lookup with comm None: the rows stay, answers come back aligned,
    a lookup is an index into the block (and refuses a row outside it)."""
    import numpy as np
    from femtools.hdf5_rows import lookup, reply, route
    a = np.arange(12.0).reshape(6, 2)
    r = route(None, np.zeros(6, np.int64), a)
    assert np.array_equal(r.received[0], a)
    back, = reply(None, r, r.received[0] * 2)
    assert np.array_equal(back, a * 2)
    with pytest.raises(ValueError, match="outside the 1 ranks"):
        route(None, np.array([1]), a[:1])
    edges = np.array([0, 6])
    assert np.array_equal(lookup(None, edges, a, [5, 0, 5]), a[[5, 0, 5]])
    with pytest.raises(ValueError, match="outside the 6 rows"):
        lookup(None, edges, a, [6])
