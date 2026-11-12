"""`femtools.dolfinx.mixed`: components of mixed Functions on DOLFINx; and the backend's
vector Functions put together from scalar ones.

    python3 -m pytest tests/test_dolfinx_mixed.py
"""
import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytestmark = [pytest.mark.dolfinx]

import basix.ufl  # noqa: E402
from dolfinx import fem, mesh  # noqa: E402
from mpi4py import MPI  # noqa: E402

from femtools.dolfinx.mixed import set_subfunction, subfunction, subspace  # noqa: E402


def _mixed():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    el = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    return msh, fem.functionspace(msh, basix.ufl.mixed_element([el, el]))


def test_set_and_get_components():
    msh, W = _mixed()
    V0 = subspace(W, 0)
    assert subspace(W, 0) is V0  # built once
    ref0, ref1 = fem.Function(V0), fem.Function(subspace(W, 1))
    ref0.interpolate(lambda x: x[0] + 2 * x[1])
    ref1.interpolate(lambda x: np.sin(x[0]))
    q = fem.Function(W)
    set_subfunction(q, 0, ref0)
    set_subfunction(q, 1, ref1)
    out0, out1 = fem.Function(V0), fem.Function(subspace(W, 1))
    assert subfunction(q, 0, out0) is out0
    assert np.array_equal(out0.x.array, ref0.x.array)
    assert np.array_equal(subfunction(q, 1, out1).x.array, ref1.x.array)
    # the component as the mixed space's own interpolation gives it
    whole = fem.Function(W)
    whole.sub(0).interpolate(lambda x: x[0] + 2 * x[1])
    whole.sub(1).interpolate(lambda x: np.sin(x[0]))
    assert np.allclose(q.x.array, whole.x.array, atol=1e-14)


def test_other_spaces_refused():
    msh, W = _mixed()
    q = fem.Function(W)
    P2 = fem.functionspace(msh, ("Lagrange", 2))
    with pytest.raises(ValueError):
        subfunction(q, 0, fem.Function(P2))
    with pytest.raises(ValueError):
        set_subfunction(q, 0, fem.Function(P2))


def test_assigned_vector_function():
    from femtools.backend import dolfinx as B
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = B.FunctionSpace(msh, "CG", 2)
    u0, u1 = B.Function(V), B.Function(V)
    u0.interpolate(lambda x: x[0] ** 2)
    u1.interpolate(lambda x: x[1] - x[0])
    uv = B.AssignedVectorFunction([u0, u1])()
    ref = fem.Function(uv.function_space)
    ref.interpolate(lambda x: np.vstack([x[0] ** 2, x[1] - x[0]]))
    assert np.allclose(uv.x.array, ref.x.array, atol=1e-14)
