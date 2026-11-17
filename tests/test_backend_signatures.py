"""femtools.backend: every name in NAMES has the same kind (callable or not) and the same
signature (parameter names, kinds and defaults) on dolfin and on DOLFINx, but for the
differences listed in DIFFERENCES, each with its reason; a new difference fails, and so
does an entry that no longer differs. A name whose signature one side cannot report
(pybind11's) is compared by kind alone. Each backend in a process of its own.

    python3 -m pytest tests/test_backend_signatures.py
"""
import importlib.util
import json
import sys

import pytest

from tests import support

DUMP = """
import importlib, inspect, json, sys
m = importlib.import_module("femtools.backend." + sys.argv[1])
def sig(o):
    try:
        s = inspect.signature(o)
    except (TypeError, ValueError):
        return None
    return [[p.name, p.kind.name, None if p.default is p.empty else repr(p.default)]
            for p in s.parameters.values()]
out = {}
for n in m.NAMES:
    o = getattr(m, n)
    fn = callable(o) and not inspect.ismodule(o)
    out[n] = [fn, sig(o) if fn else None]
print("SIGNATURES " + json.dumps(out))
"""

UFL = "UFL's own, whose parameter names (and versions' extra keywords) differ; called alike"
GENERIC = "dolfin's is generic, (*args, **kwargs); the calls documented are the same"
DIFFERENCES = {
    "Coefficient": "DOLFINx: a tuple (ufl.Coefficient, ufl.Constant) for isinstance checks "
                   "alone; dolfin's is a class",
    "dx": UFL, "Measure": UFL, "FacetNormal": UFL, "SpatialCoordinate": UFL,
    "TrialFunction": UFL, "TrialFunctions": UFL, "TestFunction": UFL, "TestFunctions": UFL,
    "FiniteElement": "dolfin's UFL element also takes form_degree, quad_scheme, variant; "
                     "(family, cell, degree) on both",
    "VectorElement": "as FiniteElement; the default dim is the cell's topological dimension "
                     "on DOLFINx, its geometric one on dolfin",
    "MixedElement": "dolfin's UFL element takes keywords too; the elements alike",
    "PeriodicBox": "map_tol unused on DOLFINx (pairing by position within tol): default None",
    "Constant": "the second argument is a cell on dolfin, the mesh on DOLFINx (constant_on "
                "is the call both take)",
    "FunctionSpace": GENERIC, "Function": GENERIC, "Expression": GENERIC,
    "DirichletBC": GENERIC, "solve": GENERIC,
    "XDMFFile": "dolfin's: femtools.xdmf.XDMFFile2(comm_or_name, name=None), reading and "
                "checkpoints too; DOLFINx's: XDMFFile(comm, path), writing alone",
}


def _signatures(backend):
    proc = support.run([sys.executable, "-c", DUMP, backend])
    line = [x for x in proc.stdout.splitlines() if x.startswith("SIGNATURES ")]
    assert len(line) == 1, proc.stdout[-2000:]
    return json.loads(line[0][len("SIGNATURES "):])


def test_signatures_agree_but_as_listed():
    for backend in ("dolfin", "dolfinx"):
        if importlib.util.find_spec(backend) is None:
            pytest.skip("needs both backends; no {}".format(backend))
    a, b = _signatures("dolfin"), _signatures("dolfinx")
    assert sorted(a) == sorted(b)
    differ = {n for n in a
              if a[n][0] != b[n][0]
              or (a[n][1] is not None and b[n][1] is not None and a[n][1] != b[n][1])}
    assert sorted(differ - set(DIFFERENCES)) == [], {n: (a[n], b[n]) for n in differ}
    assert sorted(set(DIFFERENCES) - differ) == []      # an entry that no longer differs
