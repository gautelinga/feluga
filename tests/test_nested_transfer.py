"""`femtools.nested_transfer`: transfers between a mesh and its uniform refinement on
dolfin and DOLFINx (tests/nested_transfer_worker.py).

On a rectangle (crossed diagonals, once and twice refined) and a box, periodic in x: the
prolongation of a P2 field is its fine P2 and P1 interpolant, the injection of a fine
mixed field is its coarse P2 and P1 interpolant, R P is the identity -- all to rounding,
on 1 and 3 ranks. A fine field written by its lattice on one backend and rank count is
read back on the other backend and another rank count. The matrices and the lattice
are the same when built a few fine cells at a time.
"""
import json
import os
import sys

import pytest

from tests import support

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKER = os.path.join(HERE, "nested_transfer_worker.py")
TOL = 1e-12
pytestmark = [pytest.mark.dolfin, pytest.mark.dolfinx]


def _run(backend, ranks, h5dir, tag, read_from=None):
    out = h5dir / "{}_{}_{}.json".format(backend, ranks, tag)
    cmd = [sys.executable, WORKER, backend, str(out), str(h5dir)]
    if read_from:
        cmd.append(str(read_from))
    support.run(cmd, ranks, env={"PYTHONPATH": ROOT})
    return json.loads(out.read_text())


def _check(r, read=False):
    for case, values in r.items():
        for key, value in values.items():
            assert value < TOL, (case, key, value)
        assert ("read" in values) == read, case


@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
def test_serial(backend, tmp_path):
    pytest.importorskip(backend)
    _check(_run(backend, 1, tmp_path, "w"))


@pytest.mark.mpi
def test_across_ranks_and_backends(tmp_path):
    pytest.importorskip("dolfin")
    pytest.importorskip("dolfinx")
    _check(_run("dolfin", 1, tmp_path, "w"))
    _check(_run("dolfinx", 3, tmp_path, "w"))
    _check(_run("dolfinx", 3, tmp_path, "r", tmp_path / "dolfin_1.h5"), read=True)
    _check(_run("dolfin", 2, tmp_path, "r", tmp_path / "dolfinx_3.h5"), read=True)

