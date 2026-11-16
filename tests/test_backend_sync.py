"""`sync(f)` on either backend after a Function's vector was written through PETSc (owned
entries only, as a KSP solve writes it): assembly then sees the written values, ghosts
included. On legacy dolfin sync() updates the ghosts (it did nothing before 2026-11-16, and
a solve's integrals through the backend came out wrong on more than one rank: the
backend's assemble syncs its coefficients, which then did nothing); DOLFINx's backend
tracks the vector itself.

    python3 -m pytest tests/test_backend_sync.py -m mpi
"""
import json
import os
import sys

import pytest

from tests import support

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.mpi
@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
def test_sync_after_a_petsc_write(backend):
    pytest.importorskip(backend)
    proc = support.run([sys.executable, os.path.join(HERE, "backend_sync_worker.py"), backend],
                       2, timeout=300)
    out = [json.loads(line[7:]) for line in proc.stdout.splitlines()
           if line.startswith("RESULT ")]
    assert len(out) == 1, proc.stdout[-2000:]
    assert out[0]["after_sync"] == pytest.approx(1.0, rel=1e-12), out[0]
    if backend == "dolfin":                  # the test bites: without sync, ghosts are stale
        assert out[0]["stale"] < 1.0 - 1e-3, out[0]
        assert out[0]["raw_after_sync"] == pytest.approx(1.0, rel=1e-12), out[0]
