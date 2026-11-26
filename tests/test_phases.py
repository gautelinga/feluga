"""`feluga.phases`: a line a phase, on rank zero only.

    python3 -m pytest tests/test_phases.py

Serially (no communicator: the world, or one rank without mpi4py) and under `mpirun -n 2`,
where rank zero alone prints and the summed sizes are at least the largest.
"""
import re
import sys

import pytest

from tests import support

LINE = re.compile(r"phase (\w+): [\d.]+ s \(total [\d.]+ s\), peak RSS ([\d.]+) GB max a rank, "
                  r"([\d.]+) GB summed; now ([\d.]+) GB max a rank, ([\d.]+) GB summed")
SCRIPT = r"""
from feluga.phases import Phases, mpi_print
phase = Phases()
x = bytearray(50 << 20)
phase("alloc")
mpi_print("done", 1)
"""


def run(ranks):
    proc = support.run([sys.executable, "-c", SCRIPT], ranks=ranks)
    return proc.stdout.splitlines()


def check(lines):
    phases = [LINE.match(line) for line in lines if line.startswith("phase")]
    assert len(phases) == 1 and phases[0].group(1) == "alloc", lines
    peak, total, now, now_total = (float(g) for g in phases[0].groups()[1:])
    assert total >= peak > 0.04 and now_total >= now
    assert lines.count("done 1") == 1


def test_one_line_a_phase():
    check(run(1))


@pytest.mark.mpi
def test_rank_zero_alone_prints():
    pytest.importorskip("mpi4py")
    check(run(2))
