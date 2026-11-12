"""`femtools.dolfinx.expression.CompiledExpression`: dolfin's Expression compiled from C++.

What dolfin's Expressions mean is what these mean: <cmath>, the conditional operator,
integer division, near(), std:: names; parameters by name (also ones named like the
generated code's own variables), changed in place; vectors; a code that does not compile
reported with the compiler's message; one compilation per code; under MPI rank 0
compiles and every rank evaluates.
"""
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("mpi4py")
pytestmark = [pytest.mark.dolfinx]

from tests import support  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


@pytest.fixture(autouse=True)
def jit_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FEMTOOLS_JIT_DIR", str(tmp_path / "jit"))
    return tmp_path / "jit"


def points(n=9):
    rng = np.random.default_rng(1)
    return rng.random((3, n))


def test_values_and_parameters():
    from femtools.dolfinx.expression import CompiledExpression
    x = points()
    e = CompiledExpression("tanh((sqrt(pow(x[0]-c, 2) + pow(x[1]-c, 2)) - R)/(sqrt(2)*eps))",
                           c=0.5, R=0.3, eps=0.05)
    ref = np.tanh((np.hypot(x[0] - 0.5, x[1] - 0.5) - 0.3) / (np.sqrt(2) * 0.05))
    assert np.abs(e(x) - ref).max() < 1e-15
    e.R = 0.2
    assert e.R == 0.2
    ref = np.tanh((np.hypot(x[0] - 0.5, x[1] - 0.5) - 0.2) / (np.sqrt(2) * 0.05))
    assert np.abs(e(x) - ref).max() < 1e-15
    with pytest.raises(AttributeError):
        e.nonsense = 1.0


def test_cpp_semantics():
    from femtools.dolfinx.expression import CompiledExpression
    x = points()
    assert np.array_equal(CompiledExpression("x[0] > 0.5 ? 1.0 : -1.0")(x),
                          np.where(x[0] > 0.5, 1.0, -1.0))
    assert np.all(CompiledExpression("1/2 + 0*x[0]")(x) == 0.0)     # integer division
    assert np.all(CompiledExpression("near(x[0], x[0]) && x[1] < 2 ? 3.0 : 0.0")(x) == 3.0)
    assert np.allclose(CompiledExpression("std::max(0.0, std::abs(x[2] - 0.5))")(x),
                       np.maximum(0.0, np.abs(x[2] - 0.5)))
    assert np.allclose(CompiledExpression("sin(pi*x[0]) + DOLFIN_PI*0")(x), np.sin(np.pi * x[0]))


def test_parameters_named_like_the_generated_code():
    from femtools.dolfinx.expression import CompiledExpression
    x = points()
    e = CompiledExpression("n*x[0] + i + p + X + out", n=2.0, i=1.0, p=1.0, X=1.0, out=1.0)
    assert np.allclose(e(x), 2 * x[0] + 4)
    with pytest.raises(ValueError):
        CompiledExpression("x[0]", x=1.0)


def test_vector():
    from femtools.dolfinx.expression import CompiledExpression
    x = points()
    v = CompiledExpression(("x[1]", "-x[0]", "a"), a=2.0)
    assert v.value_size == 3
    assert np.allclose(v(x), np.vstack([x[1], -x[0], np.full(x.shape[1], 2.0)]))


def test_compile_error_and_cache(jit_dir):
    from femtools.dolfinx.expression import CompiledExpression
    with pytest.raises(ValueError, match="does not compile"):
        CompiledExpression("x[0] +* 2")
    CompiledExpression("x[0] + a", a=1.0)
    CompiledExpression("x[0] + a", a=5.0)        # same code: the same library
    libraries = [f for f in os.listdir(jit_dir) if f.endswith(".so")]
    assert len(libraries) == 1


@pytest.mark.mpi
def test_three_ranks(tmp_path, jit_dir):
    out = tmp_path / "res.json"
    support.run([sys.executable, os.path.join(HERE, "dolfinx_expression_worker.py"), str(out)],
                3, env={"PYTHONPATH": ROOT, "FEMTOOLS_JIT_DIR": str(jit_dir)})
    res = json.loads(out.read_text())
    assert res["ok"] and res["size"] == 3
