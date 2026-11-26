"""`feluga.jit`: dolfin's compile_cpp_code on a communicator smaller than MPI's world,
built on its rank 0 and loaded on every rank of it, and a build that fails raising on
every rank of it (dolfin's own waits for the whole world, and hangs there).

    python3 -m pytest tests/test_jit.py
"""
import sys

import pytest

from tests import support

pytestmark = [pytest.mark.dolfin]

SUBSET = r"""
import os
from mpi4py import MPI
from feluga.jit import dolfin_module
comm = MPI.COMM_WORLD
sub = comm.Split(int(comm.rank < 2), comm.rank)
CODE = '''
#include <pybind11/pybind11.h>
int answer() { return %d; }
PYBIND11_MODULE(SIGNATURE, m) { m.def("answer", &answer); }
'''
if comm.rank < 2:
    print("ANSWER", comm.rank, dolfin_module(CODE % 42, sub, "test").answer(), flush=True)
    try:
        dolfin_module("this is not C++ SIGNATURE", sub, "a bad build")
    except RuntimeError as e:
        print("FAILED", comm.rank, "a bad build" in str(e), flush=True)
else:
    print("ANSWER", comm.rank, dolfin_module(CODE % 7, MPI.COMM_SELF, "test").answer(),
          flush=True)
comm.Barrier()
"""


@pytest.mark.mpi
def test_a_dolfin_module_built_on_part_of_the_ranks(tmp_path):
    """On three ranks, two compile one module on their own communicator and the third
    another alone, from an empty cache: each gets its module; a build that fails raises
    on both ranks of the two, naming what was built."""
    pytest.importorskip("dolfin")
    # in a folder of its own: dijitso leaves a failed build's sources in the working one
    proc = support.run([sys.executable, "-c", SUBSET], 3, timeout=600,
                       env={"DIJITSO_CACHE_DIR": str(tmp_path / "dijitso")},
                       cwd=str(tmp_path))
    lines = sorted(l for l in proc.stdout.splitlines() if l.startswith(("ANSWER", "FAILED")))
    assert lines == ["ANSWER 0 42", "ANSWER 1 42", "ANSWER 2 7", "FAILED 0 True",
                     "FAILED 1 True"], proc.stdout[-3000:] + proc.stderr[-3000:]
