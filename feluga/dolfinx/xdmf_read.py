"""DOLFINx Functions read from XDMF time series in feluga.xdmf.XDMFFile2's layout,
written by either backend (XDMFFile2 on dolfin, feluga.dolfinx.xdmf.XDMFSeries on
DOLFINx): the counterpart of `XDMFFile2.read_checkpoint` for its quadratic grids.

    read_nodal("u.xdmf", "u", -1, u)      # the last step of field "u" into Function u

When u's space is continuous Lagrange of the file's degree, its owned dofs are matched to
the file's nodes by position and take their values, then the ghosts are updated: no other
space, no interpolation. Onto any other degree, the field is read so into a plain Lagrange
space of the file's degree and interpolated from there. A periodic field, written on both
sides of the period, is the same there. Each rank reads one block of the file's nodes;
nodes and dofs meet by position (`feluga.xdmf.read_nodes_at`): nothing is read whole,
and rank zero alone parses the XML. A scalar or vector u (not a mixed one: read its
components), with as many components as the file's field. Collective.
"""
import numpy as np

from feluga.lru import WeakCache
from feluga.xdmf import read_nodes_at

_PLAIN = WeakCache()


def _plain_space(V, degree, bs):
    """The plain Lagrange space of `degree` (bs components) on V's mesh, made once while
    V lives (keyed on V, not the mesh: the space holds its mesh, and a cache entry must
    not keep its own key alive)."""
    from dolfinx import fem
    spaces = _PLAIN.get((V,), dict)
    if (degree, bs) not in spaces:
        spaces[degree, bs] = fem.functionspace(
            V.mesh, ("Lagrange", degree, (bs,)) if bs > 1 else ("Lagrange", degree))
    return spaces[degree, bs]


def _at_nodes(V, degree):
    """Whether V's dofs sit at the nodes of a grid of `degree`: continuous Lagrange of that
    degree, scalar or blocked."""
    e = V.ufl_element()
    return (getattr(e, "family_name", None) == "P" and not e.discontinuous
            and e.embedded_superdegree == degree)


def _read_owned(path, grid, name, f, comm):
    """The nodal values of grid's attribute `name` into the owned dofs of the Function f,
    whose space has its dofs at the grid's nodes (`_at_nodes`), ghosts updated after."""
    V = f.function_space
    bs = V.dofmap.index_map_bs
    n_owned = V.dofmap.index_map.size_local
    x = V.tabulate_dof_coordinates()[:n_owned, :V.mesh.geometry.dim]
    found = read_nodes_at(path, grid, name, x, comm)
    f.x.array[:n_owned * bs] = np.asarray(found[:, :bs], dtype=float).reshape(-1)
    f.x.scatter_forward()


def read_nodal(path, name, counter, u):
    """Step `counter` (an index into the steps holding `name`, negative from the end)
    of field `name` into the DOLFINx Function u, its array ghosts included. Returns u."""
    from dolfinx import fem
    from feluga.xdmf import _NODAL_DEGREE, _scan

    V = u.function_space
    mesh = V.mesh
    comm = mesh.comm if mesh.comm.size > 1 else None
    picked = [g for g in _scan(path, comm) if name in g["attributes"]]
    if not picked:
        raise ValueError("no field named {!r} in {}".format(name, path))
    if not -len(picked) <= counter < len(picked):
        raise ValueError("{} has {} step(s) of {!r}; counter {} is out of range".format(
            path, len(picked), name, counter))
    grid = picked[counter]
    degree = _NODAL_DEGREE[grid["topology"]]
    if V.element.num_sub_elements > 0 and V.dofmap.index_map_bs == 1:
        raise NotImplementedError("read_nodal: a mixed space; read its components")
    bs = V.dofmap.index_map_bs
    shape = grid["attributes"][name][2]
    columns = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    # a 2D vector is written padded to three components (feluga.xdmf._pad3)
    if not (columns == bs or (bs == 2 and columns == 3)):
        raise ValueError("{}: field {!r} has {} component(s); u has {}".format(
            path, name, columns, bs))
    if _at_nodes(V, degree):
        _read_owned(path, grid, name, u, comm)
        return u
    f = fem.Function(_plain_space(V, degree, bs))
    _read_owned(path, grid, name, f, comm)
    u.interpolate(f)
    return u
