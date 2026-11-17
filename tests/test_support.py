"""tests/support.py: a child's PYTHONPATH is the tree first, then the caller's and the
inherited one, joined (a caller's never drops the inherited one, where DOLFINx or petsc4py
may be installed).

    python3 -m pytest tests/test_support.py
"""
import os
import sys

from tests import support


def test_pythonpath_joins(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/inherited", support.ROOT]))
    assert support.pythonpath({"PYTHONPATH": "/caller"}).split(os.pathsep) == \
        [support.ROOT, "/caller", "/inherited"]
    assert support.pythonpath().split(os.pathsep) == [support.ROOT, "/inherited"]


def test_child_keeps_the_inherited_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/inherited")
    proc = support.run([sys.executable, "-c", "import os; print(os.environ['PYTHONPATH'])"],
                       env={"PYTHONPATH": "/caller"})
    assert proc.stdout.split()[0].split(os.pathsep) == [support.ROOT, "/caller", "/inherited"]
