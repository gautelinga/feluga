"""DG0 Functions in dolfin's HDF5 layout on both backends: written on two ranks by dolfin
or DOLFINx, each file reads back on three ranks on either, to round-off (the two
backends' fields are their own assemblies of the same cell means), and both carry
dolfin's signature for the element.

    python3 -m pytest tests/test_dg0_h5.py
"""
import os
import sys

import pytest

from tests import support

pytestmark = [pytest.mark.dolfin, pytest.mark.dolfinx, pytest.mark.mpi]
HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "dg0_h5_worker.py")
BACKENDS = ("dolfin", "dolfinx")


def test_dg0_written_and_read_across_backends(tmp_path):
    import h5py
    for b in BACKENDS:
        pytest.importorskip(b)
    for w in BACKENDS:
        support.run([sys.executable, WORKER, w, str(tmp_path / (w + ".h5")), "w"], 2)
        with h5py.File(tmp_path / (w + ".h5"), "r") as h5:
            assert h5["f"].attrs["signature"] == \
                b"FiniteElement('Discontinuous Lagrange', tetrahedron, 0)", w
    for r in BACKENDS:
        for w in BACKENDS:
            out = support.run([sys.executable, WORKER, r, str(tmp_path / (w + ".h5")), "r"], 3)
            err = float(support.lines(out.stdout)[-1].split()[-1])
            assert err < (1e-15 if r == w else 1e-14), (r, w, err)
