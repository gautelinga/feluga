"""DOLFINx Functions read from XDMF time series in femtools.xdmf.XDMFFile2's layout,
written by either backend (XDMFFile2 on dolfin, femtools.dolfinx.xdmf.XDMFSeries on
DOLFINx): the counterpart of `XDMFFile2.read_checkpoint` for its quadratic grids.

    read_nodal("u.xdmf", "u", -1, u)      # the last step of field "u" into Function u

The file's nodes are matched to the dofs of a plain Lagrange space of the file's degree
by position, and that field is interpolated onto u's space, any degree and periodicity
(a periodic field, written on both sides of the period, is the same there). Each rank
reads one block of the file's nodes; nodes and dofs meet by position
(`femtools.xdmf.match_nodes`): nothing is read whole. A scalar or vector u (not a mixed
one: read its components), with as many components as the file's field. Collective.
"""
import numpy as np

from femtools.dolfinx.checkpoint import _WeakCache

_PLAIN = _WeakCache()


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


def read_nodal(path, name, counter, u):
    """Step `counter` (an index into the steps holding `name`, negative from the end)
    of field `name` into the DOLFINx Function u, its array ghosts included. Returns u."""
    from dolfinx import fem
    from femtools.hdf5_rows import agreed, as_rows, extent
    from femtools.xdmf import _NODAL_DEGREE, _block_of, _scan, match_nodes

    picked = [g for g in _scan(path) if name in g["attributes"]]
    if not picked:
        raise ValueError("no field named {!r} in {}".format(name, path))
    if not -len(picked) <= counter < len(picked):
        raise ValueError("{} has {} step(s) of {!r}; counter {} is out of range".format(
            path, len(picked), name, counter))
    grid = picked[counter]
    degree = _NODAL_DEGREE[grid["topology"]]
    V = u.function_space
    if V.element.num_sub_elements > 0 and V.dofmap.index_map_bs == 1:
        raise NotImplementedError("read_nodal: a mixed space; read its components")
    mesh = V.mesh
    comm = mesh.comm if mesh.comm.size > 1 else None
    gdim = mesh.geometry.dim
    bs = V.dofmap.index_map_bs
    shape = grid["attributes"][name][2]
    columns = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    # a 2D vector is written padded to three components (femtools.xdmf._pad3)
    if not (columns == bs or (bs == 2 and columns == 3)):
        raise ValueError("{}: field {!r} has {} component(s); u has {}".format(
            path, name, columns, bs))
    plain = _plain_space(V, degree, bs)
    x = plain.tabulate_dof_coordinates()[:, :gdim]

    coords = as_rows(np.asarray(_block_of(path, grid["geometry_item"], comm), dtype=float))
    values = as_rows(_block_of(path, grid["attributes"][name], comm))
    if coords.shape[1] < gdim:
        coords = np.hstack((coords, np.zeros((len(coords), gdim - coords.shape[1]))))
    coords = np.ascontiguousarray(coords[:, :gdim])
    lo, hi = extent(coords, comm, gdim)
    tol = 1e-9 * max(float(np.max(hi - lo)) if len(lo) else 0.0, 1.0)
    found, far = match_nodes(comm, coords, values, x, tol, lo)
    if agreed(comm, far > tol):
        raise ValueError("a dof sits farther than {:.1e} from every node in {}, so the mesh is "
                         "not the one the field was written on".format(tol, path))
    f = fem.Function(plain)
    f.x.array[:] = np.asarray(found[:, :bs], dtype=float).reshape(-1)
    u.interpolate(f)
    return u
