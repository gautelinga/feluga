"""UFC's numbering of a triangle's and a tetrahedron's entities, which dolfin's dof tables
follow and feluga puts DOLFINx's in (feluga.dolfinx.cells), and the Lagrange nodes
and basis in that order: one copy for every module that needs them. numpy only.

    UFC_EDGES[d]     a cell's edges as pairs of its local vertices: edge i of a triangle
                     opposite vertex i, a tetrahedron's in UFC's order
    CELL_NAMES[d]    "triangle", "tetrahedron"
    edge_index(d)    (d+1, d+1): the edge between local vertices i and j, -1 on the diagonal
    barycentric_simplex(vertices, points)   barycentric coordinates in given cells
    ufc_nodes(lam_vertices, degree)         a Lagrange element's nodes, UFC's order
    ufc_basis(lam, degree)                  the Lagrange basis at points, UFC's order
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


def barycentric_from_vertices(v, points):
    """Barycentric coordinates of each point in the tetrahedron whose four vertex
    coordinates are given as `v`, shape (n, 4, 3), no search. Each 3x3 system
    [e1 e2 e3] lam = p - v0 is solved by Cramer's rule: the rows of the inverse are
    the cross products of the other two edges over the volume, e1 . (e2 x e3)."""
    v0 = v[:, 0]
    e1, e2, e3 = (v[:, k] - v0 for k in (1, 2, 3))
    r = np.asarray(points) - v0
    c23, c31, c12 = _cross(e2, e3), _cross(e3, e1), _cross(e1, e2)
    det = np.einsum("ij,ij->i", e1, c23)
    if not det.all():
        raise np.linalg.LinAlgError("barycentric_from_vertices: a degenerate tetrahedron")
    lam = np.empty((len(v0), 4))
    lam[:, 1] = np.einsum("ij,ij->i", c23, r) / det
    lam[:, 2] = np.einsum("ij,ij->i", c31, r) / det
    lam[:, 3] = np.einsum("ij,ij->i", c12, r) / det
    lam[:, 0] = 1.0 - lam[:, 1] - lam[:, 2] - lam[:, 3]
    return lam


def _cross(a, b):
    """Row-wise cross product of two (n, 3) arrays, without np.cross's overhead."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def barycentric_simplex(vertices, points):
    """Barycentric coordinates of each point in its triangle or tetrahedron: `vertices`
    (n, d+1, d), `points` (n, d), d = 2 or 3. Returns (n, d+1)."""
    vertices, points = np.asarray(vertices, float), np.asarray(points, float)
    d = vertices.shape[2]
    if d == 3:
        return barycentric_from_vertices(vertices, points)
    if d != 2 or vertices.shape[1] != 3:
        raise ValueError("barycentric_simplex: triangles in 2D or tetrahedra in 3D")
    v0 = vertices[:, 0]
    e1, e2 = vertices[:, 1] - v0, vertices[:, 2] - v0
    r = points - v0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    if not det.all():
        raise np.linalg.LinAlgError("barycentric_simplex: a degenerate triangle")
    lam = np.empty((len(v0), 3))
    lam[:, 1] = (r[:, 0] * e2[:, 1] - r[:, 1] * e2[:, 0]) / det
    lam[:, 2] = (e1[:, 0] * r[:, 1] - e1[:, 1] * r[:, 0]) / det
    lam[:, 0] = 1.0 - lam[:, 1] - lam[:, 2]
    return lam


def ufc_nodes(lam_vertices, degree):
    """The barycentric coordinates of a Lagrange element's nodes, in UFC's order (the
    vertices, then for degree 2 the edges' midpoints in the order of UFC_EDGES), from
    those of its cell's vertices: `lam_vertices` (n, d+1, m), m coordinates each (in
    another simplex, say). Returns (n, n_nodes, m)."""
    lam_vertices = np.asarray(lam_vertices)
    if degree == 1:
        return lam_vertices
    if degree != 2:
        raise NotImplementedError("ufc_nodes: degree 1 or 2")
    d = lam_vertices.shape[1] - 1
    edges = [0.5 * (lam_vertices[:, a] + lam_vertices[:, b]) for a, b in UFC_EDGES[d]]
    return np.concatenate([lam_vertices, np.stack(edges, axis=1)], axis=1)


def ufc_basis(lam, degree):
    """The Lagrange basis of degree 1 or 2 at barycentric `lam` (..., d+1), in UFC's
    order: shape (..., n_nodes)."""
    lam = np.asarray(lam)
    if degree == 1:
        return lam
    if degree != 2:
        raise NotImplementedError("ufc_basis: degree 1 or 2")
    d = lam.shape[-1] - 1
    vertex = lam * (2.0 * lam - 1.0)
    edge = [4.0 * lam[..., a] * lam[..., b] for a, b in UFC_EDGES[d]]
    return np.concatenate([vertex, np.stack(edge, axis=-1)], axis=-1)
