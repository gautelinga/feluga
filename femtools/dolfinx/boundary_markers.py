"""Facet markers on a DOLFINx mesh, as dolfin's facet MeshFunction, written and read as a
boundary facet file (`femtools.boundaries`) -- the file femtools.boundary_markers writes
and reads on dolfin, so either backend reads the other's.

    markers = BoundaryMarkers(mesh)
    markers.mark(lambda x: np.isclose(x[0], 0.0), 3)
    markers.write("boundaries.xdmf")
    markers = BoundaryMarkers.read("boundaries.xdmf", mesh)
    facets = compiled_facets(mesh, "on_boundary && near(x[0], 0)")

femtools.backend.dolfinx gives both under the same names.
"""
import numpy as np

import dolfinx.mesh


def local_facets(mesh):
    """`(n, exterior)`: the number of local facets (owned and ghost) and the exterior
    ones, the facets and their cells' connectivity made first."""
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    imap = mesh.topology.index_map(tdim - 1)
    return (imap.size_local + imap.num_ghosts,
            dolfinx.mesh.exterior_facet_indices(mesh.topology))


class BoundaryMarkers:
    """Integer markers on the facets of a mesh, as dolfin's facet MeshFunction:
    set_all(value), mark(where, value) with `where` a vectorized predicate of the
    coordinates (an array of shape (3, n)) or an object with such an `inside_points`;
    a boundary facet is marked when its vertices and its midpoint satisfy it.
    `meshtags()` gives the boundary facets' markers for DOLFINx (Measure takes the
    BoundaryMarkers)."""

    def __init__(self, mesh, value=0):
        self.mesh = mesh
        n, self.boundary = local_facets(mesh)
        self.values = np.full(n, value, dtype=np.int32)
        self._tags = None
        self.version = 0      # counts the changes, for the forms compiled with them

    def set_all(self, value):
        self.values[:] = value
        self._changed()

    def _changed(self):
        self._tags = None
        self.version += 1

    def mark(self, where, value):
        """Mark `value` on the boundary facets where `where` holds at every vertex and
        at the midpoint (as dolfin's SubDomain.mark checks them)."""
        predicate = getattr(where, "inside_points", where)
        fdim = self.mesh.topology.dim - 1
        facets = dolfinx.mesh.locate_entities_boundary(self.mesh, fdim, predicate)
        if len(facets):
            mid = dolfinx.mesh.compute_midpoints(self.mesh, fdim, facets)
            facets = facets[np.asarray(predicate(mid.T), dtype=bool)]
        self.values[facets] = value
        self._changed()

    def array(self):
        """The local facets' markers (owned and ghost), as dolfin's MeshFunction.array();
        written through set_values."""
        return self.values

    def set_values(self, values):
        self.values[:] = values
        self._changed()

    def where(self, value):
        """The local facets (owned and ghost) marked `value`."""
        return np.flatnonzero(self.values == value).astype(np.int32)

    def write(self, path):
        """`path` (.xdmf) and its .h5: the exterior facets with a nonzero marker, as
        femtools.boundary_markers.BoundaryMarkers.write writes them. Collective; returns
        `(facets, points)` counts."""
        from femtools.dolfinx.mesh import write_boundary_facets
        return write_boundary_facets(path, self.mesh, self.values)

    @classmethod
    def read(cls, path, mesh):
        """The markers of `path`'s facets on `mesh`'s exterior facets, matched by vertex
        coordinates (`femtools.boundaries.match_exterior`); every other facet 0. Refuses a
        facet that matches no exterior facet. Collective."""
        from femtools.boundaries import match_exterior
        self = cls(mesh)
        exterior = self.boundary.astype(np.int32)
        fdim = mesh.topology.dim - 1
        nodes = dolfinx.mesh.entities_to_geometry(mesh, fdim, exterior).reshape(
            len(exterior), fdim + 1)
        owned = exterior < mesh.topology.index_map(fdim).size_local
        at, marks = match_exterior(path, mesh.geometry.x[:, :mesh.geometry.dim], nodes,
                                   owned, mesh.comm if mesh.comm.size > 1 else None)
        self.values[exterior[at]] = marks
        self._changed()
        return self

    def meshtags(self):
        if self._tags is None:
            facets = np.sort(self.boundary).astype(np.int32)
            self._tags = dolfinx.mesh.meshtags(self.mesh, self.mesh.topology.dim - 1,
                                               facets, self.values[facets])
        return self._tags


def compiled_facets(mesh, code):
    """The local facets where `code` holds, as dolfin's CompiledSubDomain marks them: C++
    in x[0], x[1], x[2] and `on_boundary` (true on exterior facets), compiled as an
    Expression is (femtools.dolfinx.expression); a facet is in when the code holds at its
    vertices and its midpoint."""
    from femtools.dolfinx.expression import CompiledExpression
    n, exterior = local_facets(mesh)
    facets = np.arange(n, dtype=np.int32)
    nodes = dolfinx.mesh.entities_to_geometry(mesh, mesh.topology.dim - 1, facets)
    x = mesh.geometry.x[nodes]                                          # (n, tdim, 3)
    points = np.concatenate([x, x.mean(axis=1, keepdims=True)], axis=1)
    boundary = np.zeros(n, bool)
    boundary[exterior] = True
    on = "on_boundary" in code
    expr = CompiledExpression(code, mesh.comm, **({"on_boundary": 0.0} if on else {}))
    inside = np.zeros(points.shape[:2], bool)
    for flag in ((True, False) if on else (None,)):
        rows = slice(None) if flag is None else (boundary if flag else ~boundary)
        if on:
            expr.on_boundary = 1.0 if flag else 0.0
        pts = points[rows]
        if len(pts):
            inside[rows] = (np.asarray(expr(pts.reshape(-1, 3).T)).reshape(pts.shape[:2])
                            != 0.0)
    return facets[inside.all(axis=1)]
