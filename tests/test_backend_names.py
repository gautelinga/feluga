"""femtools.backend: both backends export exactly femtools.backend.NAMES, each name
defined. Each backend is imported in a process of its own (dolfin and DOLFINx do not
share one).

    python3 -m pytest tests/test_backend_names.py
"""
import os
import sys

import pytest

from femtools.backend import NAMES
from tests import support

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECK = """
import importlib, sys
m = importlib.import_module("femtools.backend." + sys.argv[1])
missing = [n for n in m.__all__ if not hasattr(m, n)]
print(sorted(m.__all__) == sorted(m.NAMES), m.name == sys.argv[1], missing)
"""


def test_names_unique():
    assert len(NAMES) == len(set(NAMES))


@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
def test_backend_exports_the_names(backend):
    pytest.importorskip(backend)
    proc = support.run([sys.executable, "-c", CHECK, backend], env={"PYTHONPATH": ROOT})
    assert proc.stdout.strip().splitlines()[-1] == "True True []"
