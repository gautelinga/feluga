"""`femtools.dolfinx.mixed`: components of mixed Functions, and vector Functions from
scalar ones, on DOLFINx.

    python3 -m pytest tests/test_dolfinx_mixed.py
"""
import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytestmark = [pytest.mark.dolfinx]

import basix.ufl  # noqa: E402
from dolfinx import fem, mesh  # noqa: E402
from mpi4py import MPI  # noqa: E402

from femtools.dolfinx.mixed import (AssignedVectorFunction, set_subfunction,  # noqa: E402
                                    subfunction, subspace)


def _mixed():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    el = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    return msh, fem.functionspace(msh, basix.ufl.mixed_element([el, el]))


def test_set_and_get_components():
    msh, W = _mixed()
    q = fem.Function(W)
    set_subfunction(q, 0, lambda x: x[0] + 2 * x[1])
    set_subfunction(q, 1, lambda x: np.sin(x[0]))
    V0 = subspace(W, 0)
    assert subspace(W, 0) is V0  # built once
    ref0, ref1 = fem.Function(V0), fem.Function(subspace(W, 1))
    ref0.interpolate(lambda x: x[0] + 2 * x[1])
    ref1.interpolate(lambda x: np.sin(x[0]))
    assert np.allclose(subfunction(q, 0).x.array, ref0.x.array, atol=1e-14)
    assert np.allclose(subfunction(q, 1).x.array, ref1.x.array, atol=1e-14)
    # a Function on the collapsed space goes in as it is
    ref0.x.array[:] *= 3.0
    set_subfunction(q, 0, ref0)
    assert np.allclose(subfunction(q, 0).x.array, ref0.x.array, atol=1e-14)
    assert np.allclose(subfunction(q, 1).x.array, ref1.x.array, atol=1e-14)


def test_assigned_vector_function():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = fem.functionspace(msh, ("Lagrange", 2))
    u0, u1 = fem.Function(V), fem.Function(V)
    u0.interpolate(lambda x: x[0] ** 2)
    u1.interpolate(lambda x: x[1] - x[0])
    uv = AssignedVectorFunction([u0, u1])()
    ref = fem.Function(uv.function_space)
    ref.interpolate(lambda x: np.vstack([x[0] ** 2, x[1] - x[0]]))
    assert np.allclose(uv.x.array, ref.x.array, atol=1e-14)
