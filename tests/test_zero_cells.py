"""femtools' collective data paths on more ranks than cells (tests/zero_cells_worker.py):
the dof tables, a checkpoint written and read, an XDMF series written and read, the
nested transfer's prolongation, injection and lattice -- on a mesh of two cells over four
ranks, two of which hold none. Every step must finish on every rank, with the right
values (a rank-local exception or a hang fails it)."""
import json
import os
import sys

import pytest

from tests import support

pytestmark = pytest.mark.mpi

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "zero_cells_worker.py")


def _run(backend, tmp_path):
    out = tmp_path / "{}.json".format(backend)
    support.run([sys.executable, WORKER, backend, str(tmp_path), str(out)], 4,
                env={"ZC_TIMEOUT": "120"}, timeout=180)
    return json.loads(out.read_text())


@pytest.mark.dolfinx
def test_dolfinx(tmp_path):
    pytest.importorskip("dolfinx")
    r = _run("dolfinx", tmp_path)
    assert 0 in r["cells"] and r["tables_cells"] == 2
    for key in ("checkpoint", "xdmf", "prolongation", "injection", "lattice"):
        assert r[key] < 1e-12, key


@pytest.mark.dolfin
def test_dolfin(tmp_path):
    pytest.importorskip("dolfin")
    r = _run("dolfin", tmp_path)
    assert 0 in r["cells"] and r["tables_cells"] == 2
    assert r["checkpoint"] < 1e-12


@pytest.mark.dolfin
def test_dolfin_refinement_refused(tmp_path):
    """dolfin cannot refine with ranks that hold no cells: refused on every rank (the
    worker finishing on all four is the check that none hangs)."""
    pytest.importorskip("dolfin")
    r = _run("dolfin-nested", tmp_path)
    assert 0 in r["cells"] and "refused" in r
