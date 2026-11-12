"""UFC's numbering of a triangle's and a tetrahedron's entities, which dolfin's dof tables
follow and femtools puts DOLFINx's in (femtools.dolfinx.cells): one copy for every module
that needs it. numpy only.

    UFC_EDGES[d]     a cell's edges as pairs of its local vertices: edge i of a triangle
                     opposite vertex i, a tetrahedron's in UFC's order
    CELL_NAMES[d]    "triangle", "tetrahedron"
    edge_index(d)    (d+1, d+1): the edge between local vertices i and j, -1 on the diagonal
"""
import numpy as np

UFC_EDGES = {2: ((1, 2), (0, 2), (0, 1)),
             3: ((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))}
CELL_NAMES = {2: "triangle", 3: "tetrahedron"}


def edge_index(tdim):
    """The reference cell's edge between local vertices i and j, (d+1, d+1); UFC's and
    basix's edges are the same."""
    E = np.full((tdim + 1, tdim + 1), -1, dtype=np.int64)
    for k, (a, b) in enumerate(UFC_EDGES[tdim]):
        E[a, b] = E[b, a] = k
    return E
