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
  the other, a vector's unrolled. DOLFINx's local dofs are found there through the dof
  layout's entity dofs (`dolfin_cell_dofs`).

Continuous Lagrange elements of degree 1 and 2 on affine triangles and tetrahedra, vector
(blocked) elements of them, and mixed elements of them (Taylor-Hood too). The file's dofs
need not be numbered as here: a reader goes through the file's own `cells`, `cell_dofs`
and `vector_0`.
"""
import numpy as np

from femtools.dolfin_layout import _fingerprint, _offsets, _write_tables
from femtools.hdf5_rows import write_ordered

# UFC's edges: edge i of a triangle is opposite vertex i; a tetrahedron's in UFC order.
EDGES = {2: [(1, 2), (0, 2), (0, 1)],
         3: [(2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1)]}
CELLS = {2: "triangle", 3: "tetrahedron"}


def _components(V):
    """V's scalar components in the order dolfin's cell dofs hold them: a mixed space's
    one after the other, a vector (blocked) space's or sub-space's unrolled."""
    n = V.element.num_sub_elements
    if n == 0:
        return [V]
    return [leaf for i in range(n) for leaf in _components(V.sub(i))]


def _lagrange_degree(S):
    e = S.ufl_element()
    if getattr(e, "discontinuous", False) or e.family_name not in ("P", "Lagrange"):
        raise NotImplementedError("dolfin layout: continuous Lagrange elements on "
                                  "simplices only, not {}".format(e))
    degree = e.embedded_superdegree
    if degree not in (1, 2):
        raise NotImplementedError("dolfin layout: degree 1 or 2, not {}".format(degree))
    return degree


def _blocked(V):
    return V.dofmap.dof_layout.block_size > 1


def signature(V):
    """dolfin's signature of V's element: scalar Lagrange, vector, or mixed of them
    (a Taylor-Hood space's `MixedElement(VectorElement(...), FiniteElement(...))`)."""
    tdim = V.mesh.topology.dim

    def sig(S):
        n = S.element.num_sub_elements
        if n == 0:
            return "FiniteElement('Lagrange', {}, {})".format(CELLS[tdim], _lagrange_degree(S))
        parts = [sig(S.sub(i)) for i in range(n)]
        if _blocked(S):
            return "VectorElement({}, dim={})".format(parts[0], n)
        return "MixedElement({})".format(", ".join(parts))

    return sig(V)


class _WeakCache:
    """Values computed once per live object (a space, a UniqueDofs), dropped with it: the
    object is not kept alive, and a later object reusing its id is not served."""

    def __init__(self):
        self._d = {}

    def get(self, objs, build):
        import weakref
        key = tuple(id(o) for o in objs)
        hit = self._d.get(key)
        if hit is not None and all(r() is o for r, o in zip(hit[0], objs)):
            return hit[1]
        value = build()
        refs = tuple(weakref.ref(o) for o in objs)
        self._d[key] = (refs, value)
        for o in objs:
            weakref.finalize(o, self._d.pop, key, None)
        return value

    def __len__(self):
        return len(self._d)


_TABLES = _WeakCache()


def _vertex_order(mesh):
    """This rank's own cells' local vertices (their reference positions) in dolfin's
    (UFC's) order: by increasing input global index. (n, tdim+1)."""
    tdim = mesh.topology.dim
    if tdim not in EDGES or mesh.geometry.cmap.degree != 1:
        raise NotImplementedError("dolfin layout: affine triangles and tetrahedra only")
    n = mesh.topology.index_map(tdim).size_local
    G = mesh.geometry.dofmaps[0][:n]
    gv = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
    return np.argsort(gv[G], axis=1, kind="stable")


def ufc_vertices(mesh):
    """This rank's own cells' geometry nodes (their vertices, the mesh being affine) in
    dolfin's (UFC's) order: by increasing input global index. (n, tdim+1)."""
    n = mesh.topology.index_map(mesh.topology.dim).size_local
    G = mesh.geometry.dofmaps[0][:n]
    return np.take_along_axis(G, _vertex_order(mesh), axis=1)


def _edge_index(tdim):
    """The reference cell's edge between local vertices i and j, (d+1, d+1): UFC's and
    basix's edges are the same."""
    E = np.full((tdim + 1, tdim + 1), -1, dtype=np.int64)
    for k, (a, b) in enumerate(EDGES[tdim]):
        E[a, b] = E[b, a] = k
    return E


def dolfin_cell_dofs(V):
    """`(gc, D)`: this rank's own cells' global ids (uint64), and for each cell V's local
    dofs in dolfin's order (vector components unrolled), a row a cell. From the dof
    layout's entity dofs: a cell's vertex j and edge e (reference numbering) carry dofs
    `entity_dofs(0, j)` and `entity_dofs(1, e)` of its row. Built once per space (kept
    while the space lives)."""
    return _TABLES.get((V,), lambda: _dolfin_cell_dofs(V))


def _dolfin_cell_dofs(V):
    mesh = V.mesh
    tdim = mesh.topology.dim
    n = mesh.topology.index_map(tdim).size_local
    perm = _vertex_order(mesh)                       # (n, d+1): reference vertex per slot
    pairs = np.asarray(EDGES[tdim])                  # dolfin's edges by dolfin's slots
    ref_edges = _edge_index(tdim)[perm[:, pairs[:, 0]], perm[:, pairs[:, 1]]]   # (n, n_e)
    blocks = []
    for S in _components(V):
        layout = S.dofmap.dof_layout
        vdof = np.array([layout.entity_dofs(0, j)[0] for j in range(tdim + 1)])
        cols = vdof[perm]
        if _lagrange_degree(S) == 2:
            edof = np.array([layout.entity_dofs(1, e)[0] for e in range(len(pairs))])
            cols = np.concatenate([cols, edof[ref_edges]], axis=1)
        D = np.asarray(S.dofmap.list[:n])            # V's local dofs, DOLFINx's order
        if D.shape[1] != cols.shape[1]:
            raise NotImplementedError("dolfin layout: unexpected dofs per cell")
        blocks.append(np.take_along_axis(D, cols, axis=1))
    D = np.concatenate(blocks, axis=1)
    gc = np.asarray(mesh.topology.original_cell_index[:n], dtype=np.uint64)
    return gc, D


def _global_tables(V, U):
    from femtools.tables import space_tables
    t = space_tables((V, U))
    return t.gc, t.dofs


_FINGERPRINTS = _WeakCache()


def _space_fingerprint(V, U):
    """The fingerprint of V's tables numbered by U (femtools.dolfin_layout), computed once
    per (V, U). Collective."""
    return _FINGERPRINTS.get((V, U), lambda: _fingerprint(V.mesh.comm, *_global_tables(V, U)))


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
    fingerprint = _FINGERPRINTS.get((V, U), lambda: _fingerprint(comm, gc, dofs))
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
    fingerprint = _space_fingerprint(V, U)
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
    numbered by U (femtools.tables.read_values, which dolfin's reader is too).
    Collective."""
    from femtools.tables import read_values, space_tables
    return read_values(path, name, space_tables((V, U)))
