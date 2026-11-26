"""The same calls on both backends do the same thing (feluga.backend): MixedElement one by
one or as a list, Function.split(deepcopy=True) (copies), norm(..., mesh=), the Krylov
solver's solve(A, x, b), set_operator and set_operators with solve(x, b), write_mesh_h5's
name and mode, and XDMFFile keeping a P2 field's midside nodes (on dolfin feluga'
XDMFFile2: dolfin's own interpolates them away) and writing facet markers. On one rank and
two.

    python3 -m pytest tests/test_backend_calls.py [-m mpi]
"""
import json
import math
import os
import sys

import pytest

from tests import support

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
@pytest.mark.parametrize("ranks", [1, pytest.param(2, marks=pytest.mark.mpi)])
def test_same_calls_same_results(tmp_path, backend, ranks):
    pytest.importorskip(backend)
    proc = support.run([sys.executable, os.path.join(HERE, "backend_calls_worker.py"), backend,
                        str(tmp_path)], ranks, timeout=600)
    out = [json.loads(line[7:]) for line in proc.stdout.splitlines()
           if line.startswith("RESULT ")]
    assert len(out) == 1, proc.stdout[-2000:]
    r = out[0]
    assert r["mixed_dims"] == [81 + 25, 81 + 25], r
    assert r["split_values"] == pytest.approx([1.0, 1.0], rel=1e-12), r
    assert r["split_independent"] == pytest.approx(1.0, rel=1e-12), r
    assert r["norm"] == pytest.approx(math.sqrt(1.0 / 3.0), rel=1e-12), r
    assert r["krylov_mean"] == pytest.approx([1.0] * 3, rel=1e-10), r
    assert r["reuse_setups"] == [1, 1, 2], r    # kept across a new operator state, then not
    assert r["krylov_same"] < 1e-10, r
    assert r["mesh_groups"] == ["first", "second"], r
    assert r["p2_kept"] and r["markers_written"], r
