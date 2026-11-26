"""The faces of a box-shaped domain, and the walls of the obstacles inside it.

    Faces(1, 0, Ly).mark(boundaries, 2)      # the faces y = 0 and y = Ly
    mark_walls(boundaries, mesh)             # every boundary facet off the box's faces: 1

`tol` is how near a face a point must be to lie on it: the default is above the
rounding of mesh generators (gmsh leaves points ~1e-14 off a face) and far below any
cell. Imports dolfin.
"""
import dolfin as df

from feluga.periodic import extent

FACE_TOL = 1e-10


class Faces(df.SubDomain):
    """The boundary where coordinate `axis` equals one of `values`."""

    def __init__(self, axis, *values, tol=FACE_TOL):
        self.axis, self.values, self.tol = axis, values, tol
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and any(abs(x[self.axis] - v) < self.tol for v in self.values)


def mark_walls(boundaries, mesh, marker=1, tol=FACE_TOL):
    """Mark the walls of the obstacles inside the mesh's bounding box `marker` in the
    facet MeshFunction `boundaries`: every boundary facet except those on the faces
    of the box, which are set to 0 (for inlets and outlets to be marked there
    afterwards). Compiled, so fast on large meshes, and it needs no list of the
    obstacles. Collective (the box is the mesh's over all ranks)."""
    df.DomainBoundary().mark(boundaries, marker)
    lo, hi = extent(mesh)
    faces = " || ".join("near(x[{0}], lo{0}, tol) || near(x[{0}], hi{0}, tol)".format(i)
                        for i in range(mesh.geometry().dim()))
    bounds = {"lo{}".format(i): v for i, v in enumerate(lo)}
    bounds.update({"hi{}".format(i): v for i, v in enumerate(hi)})
    df.CompiledSubDomain("on_boundary && ({})".format(faces), tol=tol,
                         **bounds).mark(boundaries, 0)
