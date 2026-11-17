"""One periodic Taylor-Hood Stokes solve written once against femtools.backend, run on both
backends, one rank and two: plane Poiseuille (periodic in x and z, no-slip on y = 0 and 1,
unit force along x), which P2 holds exactly, so k = 1/12, with femtools.saddle's block
preconditioner of either kind, on either Taylor-Hood layout ([P2^3, P1] and
[P2, P2, P2, P1]); and the counts the solve reports
(global_dim, global_num_cells, global_num_vertices, marker_counts, fixed_dofs) by hand; the
solution split into velocity and pressure, and its file read across the backends.
The conformance test for code meant for both backends: a backend that differs (a stale
ghost, a lost pairing, a component misnumbered) fails here.

    python3 -m pytest tests/test_backend_stokes.py [-m mpi]
"""
import json
import os
import sys

import pytest

from tests import support

HERE = os.path.dirname(os.path.abspath(__file__))
N = 4
P2_NODES = (2 * N) * (2 * N + 1) * (2 * N)      # periodic in x and z
P1_NODES = N * (N + 1) * N
WALL_NODES = 2 * (2 * N) ** 2                   # y = 0 and y = 1, folded in x and z


@pytest.mark.parametrize("layout", ["vector", "scalars"])
@pytest.mark.parametrize("kind", ["fieldsplit", "shared"])
@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
@pytest.mark.parametrize("ranks", [1, pytest.param(2, marks=pytest.mark.mpi)])
def test_poiseuille(backend, ranks, kind, layout):
    pytest.importorskip(backend)
    proc = support.run([sys.executable, os.path.join(HERE, "backend_stokes_worker.py"),
                        backend, kind, layout], ranks, timeout=600)
    out = [json.loads(line[7:]) for line in proc.stdout.splitlines()
           if line.startswith("RESULT ")]
    assert len(out) == 1, proc.stdout[-2000:]
    r = out[0]
    assert r["reason"] > 0, r
    assert r["k"] == pytest.approx(1.0 / 12.0, rel=1e-8), r
    assert r["split_k"] == pytest.approx(r["k"], rel=1e-12), r     # the split velocity
    assert r["cross"] < 1e-10 and r["divergence"] < 1e-8, r
    assert r["dim"] == 3 * P2_NODES + P1_NODES and r["pressure_dim"] == P1_NODES, r
    assert r["uy_dim"] == P2_NODES, r                                # W.sub(0).sub(1) too
    assert (r["cells"], r["vertices"]) == (6 * N ** 3, (N + 1) ** 3), r
    assert r["markers"][:2] == [4 * 2 * N * N, 2 * 2 * N * N], r     # 4 other faces; y faces
    assert r["fixed"] == 3 * WALL_NODES, r


@pytest.mark.parametrize("kind", ["fieldsplit", "shared"])
@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
@pytest.mark.parametrize("ranks", [1, pytest.param(2, marks=pytest.mark.mpi)])
def test_prefix_and_operator_change(backend, ranks, kind):
    """A KSP with an options prefix gets the AMG on each velocity split (under its prefix),
    and after A changes the preconditioner set up anew is the one before: the same
    iterations and k (the pressure split keeps diag(M), not A's empty pressure block)."""
    pytest.importorskip(backend)
    proc = support.run([sys.executable, os.path.join(HERE, "backend_stokes_worker.py"),
                        backend, kind, "vector", "resetup"], ranks, timeout=600)
    out = [json.loads(line[7:]) for line in proc.stdout.splitlines()
           if line.startswith("RESULT ")]
    assert len(out) == 1, proc.stdout[-2000:]
    r = out[0]
    again = r["resetup"]
    assert r["k"] == pytest.approx(1.0 / 12.0, rel=1e-8), r
    assert again["iterations"] == r["iterations"], r
    assert again["k"] == pytest.approx(r["k"], rel=1e-12), r
    if kind == "fieldsplit":
        assert again["splits"] == [["stokes_fieldsplit_u{}_".format(c), "hypre"]
                                   for c in range(3)] + [["stokes_fieldsplit_p_", "python"]], r


@pytest.mark.parametrize("writer,reader", [("dolfin", "dolfinx"), ("dolfinx", "dolfin")])
def test_split_solution_crosses_backends(tmp_path, writer, reader):
    """The velocity of a [P2^3, P1] solve, split as solve-stokes splits it and written in
    dolfin's layout by one backend, is read by the other and equals its own solve's (both
    to the solver's tolerance): the file layout is the same."""
    for b in (writer, reader):
        pytest.importorskip(b)
    path = str(tmp_path / "up.h5")
    worker = os.path.join(HERE, "backend_stokes_worker.py")
    support.run([sys.executable, worker, writer, "fieldsplit", "vector", "write", path], 1,
                timeout=600)
    proc = support.run([sys.executable, worker, reader, "fieldsplit", "vector", "read", path],
                       1, timeout=600)
    out = [json.loads(line[7:]) for line in proc.stdout.splitlines()
           if line.startswith("RESULT ")]
    assert len(out) == 1, proc.stdout[-2000:]
    assert out[0]["other"] < 1e-8, out[0]
