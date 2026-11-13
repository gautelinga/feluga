"""femtools.dolfinx.assembly's C++ extension packs coefficients a block of cells at a
time: matrices equal to all cells in one block (exactly on one rank; to round-off on
three, where PETSc sums the values sent between ranks in arrival order), vectors to
DOLFINx's assembly exactly, for cell, exterior and interior facet integrals with
coefficients.

    python3 -m pytest tests/test_dolfinx_blocks.py [-m mpi]
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
    out = tmp_path / "blocks.{}.json".format(ranks)
    support.run([sys.executable, os.path.join(HERE, "dolfinx_blocks_worker.py"), str(out)],
                ranks, env={"PYTHONPATH": ROOT})
    return json.loads(out.read_text())


def _check(r, matrix_tol):
    assert r["matrix_blocks_vs_whole"] <= matrix_tol
    assert r["vector_blocks_vs_dolfinx"] == 0.0


def test_blocks_one_rank(tmp_path):
    _check(_run(tmp_path, 1), 0.0)


@pytest.mark.mpi
def test_blocks_three_ranks(tmp_path):
    # PETSc adds the values sent between ranks in the order the messages arrive, which
    # differs from run to run: round-off, whatever the blocks
    _check(_run(tmp_path, 3), 1e-15)
