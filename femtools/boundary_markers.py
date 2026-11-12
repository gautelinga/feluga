"""A facet MeshFunction that is written and read as a boundary facet file.

    markers = BoundaryMarkers(mesh)
    markers.mark(subdomain, 3)
    bc = DirichletBC(V, g, markers, 3)
    ds = Measure("ds", domain=mesh, subdomain_data=markers)
    markers.write("boundaries.xdmf")
    markers = BoundaryMarkers.read("boundaries.xdmf", mesh)

`BoundaryMarkers` is a `dolfin.cpp.mesh.MeshFunctionSizet` on the facets, so it goes
wherever one does. The file (`femtools.boundaries`) holds the exterior facets with a
nonzero marker, each written by the rank whose cell holds it, on their vertices, one row
a global vertex, in global order. Reading matches its facets to the mesh's exterior
facets by vertex coordinates, on any number of ranks. Imports dolfin.
"""
import numpy as np

import dolfin as df

from femtools import boundaries


def _comm(mesh):
    """The mesh's communicator, None on one rank."""
    comm = mesh.mpi_comm()
    return None if comm.Get_size() == 1 else comm


def _exterior(mesh):
    """`(facets, vertices, owned)`: the local exterior facets, ghosts included, their
    local vertices (a row each) and whether a regular cell here holds each."""
    D = mesh.topology().dim()
    mesh.init(D - 1, D)
    mesh.init(D - 1, 0)
    on = df.MeshFunction("size_t", mesh, D - 1, 0)
    df.DomainBoundary().mark(on, 1)
    facets = np.flatnonzero(on.array())
    vertices = np.array(mesh.topology()(D - 1, 0)(), dtype=np.int64).reshape(-1, D)[facets]
    cell_facets = np.array(mesh.topology()(D, D - 1)(), dtype=np.int64).reshape(-1, D + 1)
    regular = np.zeros(mesh.num_facets(), dtype=bool)
    regular[cell_facets[:mesh.topology().ghost_offset(D)].ravel()] = True
    return facets, vertices, regular[facets]


class BoundaryMarkers(df.cpp.mesh.MeshFunctionSizet):
    """A facet `MeshFunction('size_t')` of `mesh`, every facet `value`."""

    def __init__(self, mesh, value=0):
        super().__init__(mesh, mesh.topology().dim() - 1, value)

    def mark(self, subdomain, value):
        """The facets `subdomain` (a SubDomain or CompiledSubDomain) holds set to `value`."""
        subdomain.mark(self, value)
        return self

    def write(self, path):
        """`path` and its .h5: the exterior facets with a nonzero marker. Collective;
        returns `(facets, points)` counts."""
        mesh = self.mesh()
        comm = _comm(mesh)
        facets, vertices, owned = _exterior(mesh)
        values = self.array()[facets]
        keep = owned & (values != 0)
        return boundaries.write_exterior(
            path, vertices[keep], mesh.topology().global_indices(0), mesh.coordinates(),
            values[keep], mesh.num_entities_global(0), comm)

    @classmethod
    def read(cls, path, mesh):
        """The markers of `path`'s facets on `mesh`'s exterior facets, matched by vertex
        coordinates (`femtools.boundaries.match_exterior`); every other facet 0. Refuses a
        facet that matches no exterior facet. Collective."""
        self = cls(mesh)
        facets, vertices, owned = _exterior(mesh)
        at, marks = boundaries.match_exterior(path, mesh.coordinates(), vertices, owned,
                                              _comm(mesh))
        self.array()[facets[at]] = marks
        return self
