"""femtools.garbage: petsc4py keeps a dropped PETSc object on a parallel communicator
until PETSc.garbage_cleanup; the DOLFINx backend destroys dropped matrices before it
makes another, and collect_garbage destroys the rest. On one rank and on two.

    python3 -m pytest tests/test_garbage.py [-m mpi]
"""
import json
import os
import sys

import pytest

pytest.importorskip("dolfinx")
pytestmark = [pytest.mark.dolfinx]

from tests import support  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _run(tmp_path, ranks):
    out = tmp_path / "garbage.{}.json".format(ranks)
    support.run([sys.executable, os.path.join(HERE, "garbage_worker.py"), str(out)], ranks,
                env={"PYTHONPATH": ROOT, "PETSC_OPTIONS": "-malloc_debug"})
    return json.loads(out.read_text())


def _check(r):
    assert r["matrix_mb"] > 1.0
    # one dropped matrix at most is alive: the last one, until the next is made
    assert r["five_dropped_matrices_mb"] < 1.5 * r["matrix_mb"], r
    assert r["dropped_vectors_kept_mb"] > 0.9 * r["vectors_mb"], r   # petsc4py keeps them...
    assert r["after_collect_garbage_mb"] < 0.05, r                    # ...until collect_garbage


def test_garbage_serial(tmp_path):
    _check(_run(tmp_path, 1))


@pytest.mark.mpi
def test_garbage_two_ranks(tmp_path):
    _check(_run(tmp_path, 2))
