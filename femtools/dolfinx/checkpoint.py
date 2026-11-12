"""DOLFINx Functions in dolfin's HDF5 layout (femtools.dolfin_layout): written so that
dolfin's `HDF5File.read`, partrac and femtools read them, and read from the files dolfin
writes -- a checkpoint moves between the backends and between numbers of processes.

    write_dofs(path, V, U, "dofs")                     # the dof tables, once
    write_function(path, values, V, U, "u", dofs=("tables.h5", "dofs"))
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
  the other. DOLFINx's local dofs are matched to those points by their coordinates.

Continuous Lagrange elements of degree 1 and 2 on affine triangles and tetrahedra, vector
(blocked) elements of them, and mixed elements of them. The file's dofs need not be numbered as here: a reader goes
through the file's own `cells`, `cell_dofs` and `vector_0`.
"""
import numpy as np

from femtools.dolfin_layout import _fingerprint, _offsets, _write_tables
from femtools.hdf5_rows import write_ordered

# UFC's edges: edge i of a triangle is opposite vertex i; a tetrahedron's in UFC order.
EDGES = {2: [(1, 2), (0, 2), (0, 1)],
         3: [(2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1)]}
CELLS = {2: "triangle", 3: "tetrahedron"}


def _components(V):
    n = V.element.num_sub_elements
    return [V] if n == 0 else [V.sub(i) for i in range(n)]


def _lagrange_degree(S):
    e = S.ufl_element()
    if getattr(e, "discontinuous", False) or e.family_name not in ("P", "Lagrange", "Q"):
        raise NotImplementedError("dolfin layout: continuous Lagrange elements only, not "
                                  "{}".format(e))
    degree = e.embedded_superdegree
    if degree not in (1, 2):
        raise NotImplementedError("dolfin layout: degree 1 or 2, not {}".format(degree))
    return degree


def _blocked(V):
    return V.dofmap.index_map_bs > 1


def signature(V):
    """dolfin's signature of V's element (scalar Lagrange, vector, or mixed)."""
    tdim = V.mesh.topology.dim

    def one(S):
        return "FiniteElement('Lagrange', {}, {})".format(CELLS[tdim], _lagrange_degree(S))

    parts = [one(S) for S in _components(V)]
    if _blocked(V):
        return "VectorElement({}, dim={})".format(parts[0], len(parts))
    return parts[0] if V.element.num_sub_elements == 0 else \
        "MixedElement({})".format(", ".join(parts))


def _dof_coordinates(V):
    """The coordinates of every local dof of V (a mixed space through its components)."""
    from femtools.dolfinx.periodic import flat_collapse_map
    if V.element.num_sub_elements == 0:
        return V.tabulate_dof_coordinates()
    imap = V.dofmap.index_map
    X = np.zeros(((imap.size_local + imap.num_ghosts) * V.dofmap.index_map_bs, 3))
    for S in _components(V):
        Sc, parent = S.collapse()
        X[flat_collapse_map(parent)] = Sc.tabulate_dof_coordinates()
    return X


_TABLES = {}


def dolfin_cell_dofs(V):
    """`(gc, D)`: this rank's own cells' global ids (uint64), and for each cell V's local
    dofs in dolfin's order (unrolled for a vector space), a row a cell. Built once per
    space."""
    key = id(V)
    if key in _TABLES and _TABLES[key][0] is V:
        return _TABLES[key][1]
    mesh = V.mesh
    tdim = mesh.topology.dim
    if tdim not in EDGES or mesh.geometry.cmap.degree != 1:
        raise NotImplementedError("dolfin layout: affine triangles and tetrahedra only")
    n = mesh.topology.index_map(tdim).size_local
    G = mesh.geometry.dofmaps[0][:n]
    gv = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
    G = np.take_along_axis(G, np.argsort(gv[G], axis=1, kind="stable"), axis=1)
    vertices = mesh.geometry.x[G]                            # (n, tdim+1, 3), dolfin's order
    edges = np.stack([0.5 * (vertices[:, a] + vertices[:, b]) for a, b in EDGES[tdim]], axis=1)
    X = _dof_coordinates(V)
    lo, hi = mesh.geometry.x.min(axis=0), mesh.geometry.x.max(axis=0)
    tol = 1e-8 * max(float(np.max(hi - lo)), 1e-300)
    blocks = []
    for S in _components(V):
        points = vertices if _lagrange_degree(S) == 1 else np.concatenate([vertices, edges], 1)
        D = np.asarray(S.dofmap.list[:n])                    # V's local dofs, DOLFINx's order
        if D.shape[1] != points.shape[1]:
            raise NotImplementedError("dolfin layout: unexpected dofs per cell")
        dist = np.linalg.norm(points[:, :, None, :] - X[D][:, None, :, :], axis=-1)
        j = dist.argmin(axis=2)
        if n and (np.max(np.take_along_axis(dist, j[..., None], 2)) > tol
                  or np.any(np.sort(j, axis=1) != np.arange(D.shape[1]))):
            raise RuntimeError("dolfin layout: a cell's dofs do not sit at dolfin's points")
        blocks.append(np.take_along_axis(D, j, axis=1))
    D = np.concatenate(blocks, axis=1) if n else np.zeros((0, sum(b.shape[1] for b in blocks)),
                                                          dtype=np.int32)
    gc = np.asarray(mesh.topology.original_cell_index[:n], dtype=np.uint64)
    _TABLES[key] = (V, (gc, D))
    return gc, D


def _global_tables(V, U):
    gc, D = dolfin_cell_dofs(V)
    dofs = U.compact[D]
    dtype = np.int32 if U.n_global < 2**31 else np.int64   # dolfin's int32 when it fits
    return gc, dofs.astype(dtype)


def _n_cells(V):
    return V.mesh.topology.index_map(V.mesh.topology.dim).size_global


def write_dofs(path, V, U, name="dofs", mode="w"):
    """The dof tables of V numbered by U, as femtools.dolfin_layout.write_dofs writes a
    dolfin space's, with the element's `signature` and the numbering's `fingerprint`.
    Collective."""
    from femtools.hdf5_rows import opened
    comm = V.mesh.comm
    gc, dofs = _global_tables(V, U)
    _write_tables(path, name, comm, _n_cells(V), gc, dofs, mode)
    fingerprint = _fingerprint(comm, gc, dofs)
    with opened(path, "a", comm) as h5:
        if h5 is not None:
            h5[name].attrs["signature"] = np.bytes_(signature(V))
            h5[name].attrs["fingerprint"] = fingerprint


def write_function(path, values, V, U, name, mode="a", dofs=None):
    """A Function on V, given as this rank's unique values (`values`, numbered by U), as
    dolfin's `HDF5File.write` writes one. With `dofs = (file, group)` the tables
    `write_dofs` wrote are linked instead of written (femtools.dolfin_layout). Collective."""
    import h5py
    from femtools.hdf5_rows import agreed, opened
    comm = V.mesh.comm
    values = np.asarray(values, dtype=np.float64)
    lo, hi = U.owned_range
    if len(values) != hi - lo:
        raise ValueError("write_function: {} values for {} unique dofs".format(len(values),
                                                                              hi - lo))
    _, partition = _offsets(comm, hi - lo)
    if dofs is None:
        gc, tables = _global_tables(V, U)
        _write_tables(path, name, comm, _n_cells(V), gc, tables, mode)
        mode = "a"
    else:
        bad = _check_linked(path, dofs, V, U, comm)
        if agreed(comm, bad is not None):
            raise ValueError(bad or "another rank refused the dof tables")
    write_ordered(path, name + "/vector_0", np.arange(lo, hi, dtype=np.int64), values,
                  U.n_global, comm, mode=mode, attrs={"partition": partition})
    with opened(path, "a", comm) as h5:
        if h5 is not None:
            if dofs is not None:
                for k in ("x_cell_dofs", "cell_dofs", "cells"):
                    h5[name + "/" + k] = h5py.ExternalLink(dofs[0], dofs[1] + "/" + k)
            h5[name].attrs["signature"] = np.bytes_(signature(V))


def _check_linked(path, dofs, V, U, comm):
    """Why the linked tables are not V's numbered by U, or None."""
    import os
    import h5py
    target, group = dofs
    where = os.path.join(os.path.dirname(os.path.abspath(path)), target)
    found = None
    if comm.rank == 0:
        try:
            with h5py.File(where, "r") as h5:
                a = h5[group].attrs
                found = (bytes(a["signature"]), np.uint64(a["fingerprint"]))
        except (OSError, KeyError) as e:
            found = str(e)
    found = comm.bcast(found, root=0)
    fingerprint = _fingerprint(comm, *_global_tables(V, U))
    if os.path.realpath(where) == os.path.realpath(path):
        return "{}: the dof tables must be in another file than the Function's".format(where)
    if isinstance(found, str):
        return "{}:{}: no dof tables ({})".format(where, group, found)
    if found[0] != np.bytes_(signature(V)):
        return "{}:{}: the dof tables of another element".format(where, group)
    if found[1] != fingerprint:
        return "{}:{}: the dof tables of another numbering".format(where, group)
    return None


def read_function(path, name, V, U):
    """The Function `name` in path (dolfin's layout, written by dolfin or by
    write_function, on any number of processes) on V: this rank's unique values,
    numbered by U. Collective."""
    import h5py
    from femtools.hdf5_rows import agreed, fetch_rows, rows_of_labels
    comm = V.mesh.comm
    found = None
    if comm.rank == 0:
        with h5py.File(path, "r") as h5:
            a = h5[name].attrs
            found = bytes(a["signature"]).decode() if "signature" in a else None
    found = comm.bcast(found, root=0)
    if found is not None and found != signature(V):
        raise ValueError("{}:{} holds {}, not {}".format(path, name, found, signature(V)))
    gc, D = dolfin_cell_dofs(V)
    n, k = D.shape
    rows = rows_of_labels(path, name + "/cells", gc.astype(np.int64), comm)
    starts = fetch_rows(path, name + "/x_cell_dofs", rows, comm).astype(np.int64)
    ends = fetch_rows(path, name + "/x_cell_dofs", rows + 1, comm).astype(np.int64)
    if agreed(comm, bool(np.any(ends - starts != k))):
        raise ValueError("{}:{} has another number of dofs a cell than {}".format(
            path, name, signature(V)))
    flat = (starts[:, None] + np.arange(k)).reshape(-1)
    file_dofs = fetch_rows(path, name + "/cell_dofs", flat, comm).astype(np.int64)
    vals = fetch_rows(path, name + "/vector_0", file_dofs, comm)
    imap = V.dofmap.index_map
    full = np.full((imap.size_local + imap.num_ghosts) * V.dofmap.index_map_bs, np.nan)
    full[D.reshape(-1)] = vals
    out = full[U.owned_unique]
    if agreed(comm, bool(np.any(np.isnan(out)))):
        raise RuntimeError("{}:{}: some unique dofs are in none of this rank's cells".format(
            path, name))
    return out
