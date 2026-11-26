"""`feluga.box`: faces of a box, the walls of the obstacles inside it, and
`periodic.periodic_box_from_bounds`.

    python3 -m pytest tests/test_box.py
"""
import numpy as np
import pytest

df = pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

from feluga.box import Faces, mark_walls  # noqa: E402
from feluga.periodic import periodic_box, periodic_box_from_bounds  # noqa: E402


def marked(mf, mesh, marker):
    """Midpoints of the facets marked `marker`."""
    return np.array([df.Facet(mesh, i).midpoint().array()[:2]
                     for i in np.flatnonzero(mf.array() == marker)])


def test_faces_and_walls():
    """A square with a square hole: the hole's edges are walls, the box's faces not,
    and Faces picks out the faces asked for."""
    mesh = df.RectangleMesh(df.Point(-1, -1), df.Point(1, 1), 8, 8)
    hole = df.CompiledSubDomain("max(fabs(x[0]), fabs(x[1])) < 0.25 + 1e-9")
    cells = df.MeshFunction("size_t", mesh, 2, 0)
    hole.mark(cells, 1)
    mesh = df.SubMesh(mesh, cells, 0)
    mf = df.MeshFunction("size_t", mesh, 1, 9)
    mark_walls(mf, mesh)
    walls = marked(mf, mesh, 1)
    assert len(walls) == 8 and np.allclose(np.abs(walls).max(axis=1), 0.25)
    Faces(1, -1, 1).mark(mf, 2)
    faces = marked(mf, mesh, 2)
    assert len(faces) == 16 and np.allclose(np.abs(faces[:, 1]), 1)
    assert len(marked(mf, mesh, 0)) == 16 and np.allclose(np.abs(marked(mf, mesh, 0)[:, 0]), 1)


def test_periodic_box_from_bounds():
    """{1: (-1, 2)}: periodic in y alone, the same pairing as periodic_box's."""
    mesh = df.RectangleMesh(df.Point(0, -1), df.Point(2, 2), 4, 6)
    a = periodic_box_from_bounds({1: (-1, 2)})
    b = periodic_box([0, 2, 0], 1e-10, [False, True, False], [0, -1, 0])
    assert list(a.periodic) == [False, True, False] and np.allclose(a.lower, [0, -1, 0])
    dims = [df.FunctionSpace(mesh, "P", 1, constrained_domain=box).dim() for box in (a, b)]
    assert dims[0] == dims[1] == 5 * 7 - 5
