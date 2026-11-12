"""DOLFINx Functions read from XDMF time series in femtools.xdmf.XDMFFile2's layout,
written by either backend (XDMFFile2 on dolfin, femtools.dolfinx.xdmf.XDMFSeries on
DOLFINx): the counterpart of `XDMFFile2.read_checkpoint` for its quadratic grids.

    read_nodal("u.xdmf", "u", -1, u)      # the last step of field "u" into Function u

The file's nodes are matched to the dofs of a plain Lagrange space of the file's degree
by position, and that field is interpolated onto u's space, any degree and periodicity
(a periodic field, written on both sides of the period, is the same there). Each rank
reads one block of the file's nodes and sends each to the ranks whose dofs it may be
at: nothing is read whole. Collective.
"""
import numpy as np


def read_nodal(path, name, counter, u):
    """Step `counter` (an index into the steps holding `name`, negative from the end)
    of field `name` into the DOLFINx Function u, its array ghosts included. Returns u."""
    from scipy.spatial import cKDTree
    from dolfinx import fem
    from femtools.hdf5_rows import agreed, extent, route
    from femtools.xdmf import _NODAL_DEGREE, _block_of, _scan

    picked = [g for g in _scan(path) if name in g["attributes"]]
    if not picked:
        raise ValueError("no field named {!r} in {}".format(name, path))
    if not -len(picked) <= counter < len(picked):
        raise ValueError("{} has {} step(s) of {!r}; counter {} is out of range".format(
            path, len(picked), name, counter))
    grid = picked[counter]
    degree = _NODAL_DEGREE[grid["topology"]]
    V = u.function_space
    mesh = V.mesh
    comm = mesh.comm if mesh.comm.size > 1 else None
    gdim = mesh.geometry.dim
    bs = V.dofmap.index_map_bs
    plain = fem.functionspace(mesh, ("Lagrange", degree, (bs,)) if bs > 1
                              else ("Lagrange", degree))
    x = plain.tabulate_dof_coordinates()[:, :gdim]

    coords = np.asarray(_block_of(path, grid["geometry_item"], comm), dtype=float)
    coords = coords.reshape(len(coords), -1)
    values = np.asarray(_block_of(path, grid["attributes"][name], comm)).reshape(len(coords), -1)
    if coords.shape[1] < gdim:
        coords = np.hstack((coords, np.zeros((len(coords), gdim - coords.shape[1]))))
    coords = np.ascontiguousarray(coords[:, :gdim])
    lo, hi = extent(coords, comm, gdim)
    tol = 1e-9 * max(float(np.max(hi - lo)) if len(lo) else 0.0, 1.0)
    if comm is not None:
        # every node to each rank whose dofs' box holds it
        box_lo = x.min(axis=0) - tol if len(x) else np.full(gdim, np.inf)
        box_hi = x.max(axis=0) + tol if len(x) else np.full(gdim, -np.inf)
        rows, dest = [], []
        for r, (b_lo, b_hi) in enumerate(comm.allgather((box_lo, box_hi))):
            inside = np.nonzero(((coords >= b_lo) & (coords <= b_hi)).all(axis=1))[0]
            rows.append(inside)
            dest.append(np.full(len(inside), r, np.int64))
        rows = np.concatenate(rows)
        coords, values = route(comm, np.concatenate(dest), coords[rows], values[rows]).received
    if len(x) and len(coords):
        distance, index = cKDTree(coords).query(x)
        far = float(distance.max())
    else:
        index, far = np.zeros(len(x), np.int64), (np.inf if len(x) else 0.0)
    if agreed(comm, far > tol):
        raise ValueError("a dof sits farther than {:.1e} from every node in {}, so the mesh is "
                         "not the one the field was written on".format(tol, path))
    f = fem.Function(plain)
    f.x.array[:] = np.asarray(values[index, :bs], dtype=float).reshape(-1)
    u.interpolate(f)
    return u
