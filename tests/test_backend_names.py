"""femtools.backend: both backends export exactly femtools.backend.NAMES, each name
defined, every one femtools defines documented; `of`, `is_dolfinx` and `load` find each
backend's meshes, spaces and Functions. Each backend is imported in a process of its own
(dolfin and DOLFINx do not share one).

    python3 -m pytest tests/test_backend_names.py
"""
import sys

import pytest

from femtools.backend import NAMES
from tests import support


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
    proc = support.run([sys.executable, "-c", CHECK, backend])
    assert proc.stdout.strip().splitlines()[-1] == "True True []"


OF = """
import importlib, inspect, sys
L = importlib.import_module("femtools.backend." + sys.argv[1])
from femtools.backend import is_dolfinx, load, of
mesh = L.UnitSquareMesh(2, 2)
V = L.FunctionSpace(mesh, "CG", 1)
W = L.FunctionSpace(mesh, L.MixedElement([L.FiniteElement("CG", mesh.ufl_cell(), 2)] * 2))
objs = (mesh, V, W, W.sub(0), L.Function(V), L.Function(W))
undocumented = [n for n in L.NAMES
                if (inspect.isfunction(getattr(L, n)) or inspect.isclass(getattr(L, n)))
                and getattr(getattr(L, n), "__module__", "").startswith("femtools")
                and not (getattr(L, n).__doc__ or "").strip()]
print(all(of(o) is L for o in objs), all(is_dolfinx(o) == (sys.argv[1] == "dolfinx")
                                         for o in objs), load(sys.argv[1]) is L, undocumented)
"""


@pytest.mark.parametrize("backend", ["dolfin", "dolfinx"])
def test_of_finds_the_backend_and_names_are_documented(backend):
    pytest.importorskip(backend)
    proc = support.run([sys.executable, "-c", OF, backend])
    assert proc.stdout.strip().splitlines()[-1] == "True True True []"


def test_load_refuses_other_names():
    from femtools.backend import load
    with pytest.raises(ValueError):
        load("fenics")
