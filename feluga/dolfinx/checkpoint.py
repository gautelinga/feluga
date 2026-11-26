"""DOLFINx Functions in dolfin's HDF5 layout (feluga.dolfin_layout): written so that
dolfin's `HDF5File.read`, partrac and feluga read them, and read from the files dolfin
writes -- a checkpoint moves between the backends and between numbers of processes.

    write_dofs(path, V, U, "dofs")                     # the dof tables, once
    write_function(path, values, V, U, "u", dofs=("tables.h5", "dofs"))
    write_functions(path, [(values, V, U, "current"), (old, V, U, "previous")], dofs=...)
    values = read_function(path, "u", V, U)            # this rank's unique dofs

`values` are a space's unique dofs as numbered by its `UniqueDofs` U (this rank's
owned entries of a compact vector): what dolfin's `constrained_domain` space holds, so a
periodic field moves between the backends as it is.

What makes the layouts meet, cell by cell:

- Cells by global id: a DOLFINx mesh's original cell index, which is dolfin's cell index
  for a mesh from the same input (a file, or the generators, which number alike:
  dolfin's RectangleMesh and DOLFINx's create_rectangle give the same cells and vertices).
- Dofs in a cell in dolfin's (UFC's) order: the vertices by increasing global index (the
  original vertex index), then for degree 2 the edges, edge i opposite vertex i in a
  triangle (UFC's edge order in a tetrahedron); a mixed element's components one after
  the other, a vector's unrolled. DOLFINx's local dofs are found there through the dof
  layout's entity dofs (feluga.dolfinx.cells).

Continuous Lagrange elements of degree 1 and 2 on affine triangles and tetrahedra, vector
(blocked) elements of them, and mixed elements of them (Taylor-Hood too). The file's dofs
need not be numbered as here: a reader goes through the file's own `cells`, `cell_dofs`
and `vector_0`.
"""
import numpy as np

from feluga.dolfin_layout import SpaceLayout, write_dofs_of, write_functions_of
from feluga.dolfinx.cells import signature
from feluga.lru import WeakCache

_FINGERPRINTS = WeakCache()


def _global_tables(V, U):
    from feluga.tables import space_tables
    t = space_tables((V, U))
    return t.gc, t.dofs


def _layout(V, U):
    """V's SpaceLayout, its dofs numbered by U (feluga.dolfin_layout)."""
    mesh = V.mesh
    return SpaceLayout(mesh.comm, signature(V),
                       mesh.topology.index_map(mesh.topology.dim).size_global,
                       lambda: _global_tables(V, U),
                       lambda compute: _FINGERPRINTS.get((V, U), compute))


def write_dofs(path, V, U, name="dofs", mode="w"):
    """The dof tables of V numbered by U, as feluga.dolfin_layout.write_dofs writes a
    dolfin space's, with the element's `signature` and the numbering's `fingerprint`.
    Collective."""
    write_dofs_of(path, _layout(V, U), name, mode)


def write_function(path, values, V, U, name, mode="a", dofs=None):
    """A Function on V, given as this rank's unique values (`values`, numbered by U), as
    dolfin's `HDF5File.write` writes one. With `dofs = (file, group)` the tables
    `write_dofs` wrote are linked instead of written (feluga.dolfin_layout). Collective."""
    write_functions(path, [(values, V, U, name)], mode, dofs)


def write_functions(path, functions, mode="w", dofs=None):
    """Several Functions, `(values, V, U, name)` each as `write_function` takes one, in one
    file opened once: what `write_function` writes for each in turn, the first in `mode`
    and the rest appended (feluga.dolfin_layout.write_functions_of). Collective."""
    from feluga.hdf5_rows import agreed
    layouts, items = {}, []
    for values, V, U, name in functions:
        values = np.asarray(values, dtype=np.float64)
        lo, hi = U.owned_range
        if agreed(V.mesh.comm, len(values) != hi - lo):
            raise ValueError("write_function: {} values for {} unique dofs on some "
                             "rank".format(len(values), hi - lo))
        key = (id(V), id(U))
        if key not in layouts:
            layouts[key] = _layout(V, U)
        items.append((layouts[key], name, values, lo, U.n_global))
    write_functions_of(path, items, mode, dofs)


def read_function(path, name, V, U):
    """The Function `name` in path (dolfin's layout, written by dolfin or by
    write_function, on any number of processes) on V: this rank's unique values,
    numbered by U (feluga.tables.read_values, which dolfin's reader is too).
    Collective."""
    from feluga.tables import read_values, space_tables
    return read_values(path, name, space_tables((V, U)))
