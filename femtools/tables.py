"""A mesh's cells and a space's dofs as arrays, alike on dolfin and DOLFINx: what the
code that works on arrays (checkpoints, nested transfers, the dual mesh's markers) asks
of either backend, in one place.

    t = space_tables(V)              # dolfin, or femtools.backend.dolfinx; on plain
                                     # DOLFINx space_tables((V, U)), U its UniqueDofs
    t.gc, t.x, t.vgid, t.dofs        # this rank's own cells, UFC's order
    t.component(i)                   # (dofs of component i, its degree)
    values = read_values(path, "u", t)   # a Function in dolfin's layout, this rank's dofs

    f = facet_tables(mesh)           # local topology: cells' vertices and facets

A space's tables, for this rank's own cells, one row a cell:

- `gc`: the cells' global ids (uint64) -- dolfin's cell index; on DOLFINx the original
  cell index, which is dolfin's for a mesh from the same input;
- `x` (n, d+1, gdim), `vgid` (n, d+1): the cells' vertices and their global ids, in UFC's
  order -- by increasing global id;
- `dofs` (n, k): the cells' dofs as indices of the space's vectors (dolfin's global dofs,
  or on DOLFINx the unique dofs U numbers), in UFC's order: the vertices, then for degree
  2 the edges (edge i of a triangle opposite vertex i, a tetrahedron's in UFC's order), a
  mixed space's components one after the other, a vector space's unrolled; the dofmap's
  integer type;
- `components`: (first column, count, degree) per scalar component, in the order the
  cell's dofs hold them: a mixed space's components one after the other, a vector space's
  or vector sub-space's unrolled (a Taylor-Hood space in 2D has three: u_x, u_y, p) --
  the same on both backends;
- `owned` (lo, hi), `n_global`: the vector's range on this rank and size; `comm` (mpi4py);
  `n_cells`: the mesh's cells over all ranks; `signature`: dolfin's of the element.

numpy at import; dolfin or DOLFINx where a mesh or space of theirs is passed.
"""
import numpy as np


def is_dolfinx(obj):
    """Whether `obj` is a DOLFINx mesh or space (else dolfin's)."""
    try:
        import dolfinx
    except ImportError:
        return False
    return isinstance(obj, (dolfinx.mesh.Mesh, dolfinx.fem.FunctionSpace))


def _n_nodes(d, degree):
    return d + 1 if degree == 1 else (d + 1) * (d + 2) // 2


class SpaceTables:
    """A space's tables (see the module's docstring). `x` and `vgid` are made on first
    use: checkpoints need neither."""

    @property
    def x(self):
        if "x" not in self.__dict__:
            self.__dict__["x"] = self._make_x()
        return self.__dict__["x"]

    @property
    def vgid(self):
        if "vgid" not in self.__dict__:
            self.__dict__["vgid"] = self._make_vgid()
        return self.__dict__["vgid"]

    def component(self, i=None):
        """(dofs of component i, its degree); i None for a space of one component."""
        if i is None:
            if len(self.components) != 1:
                raise ValueError("tables: a mixed space needs its component")
            i = 0
        first, n, degree = self.components[i]
        return self.dofs[:, first:first + n], degree


def space_tables(space):
    """The tables of `space`: a dolfin or femtools.backend.dolfinx FunctionSpace, or a
    plain DOLFINx one with its UniqueDofs as `(V, U)`. Collective (dolfin's integer type is agreed over the ranks)."""
    V, U = space if isinstance(space, tuple) else (space, None)
    t = SpaceTables()
    if is_dolfinx(V):
        _dolfinx_space(t, V, U)
    else:
        _dolfin_space(t, V)
    if t.vertices_per_cell != t.gdim + 1:
        raise NotImplementedError("tables: triangles and tetrahedra in their own dimension")
    return t


def _dolfin_space(t, V):
    from femtools.periodic import cell_dofs
    mesh = V.mesh()
    comm = mesh.mpi_comm()
    d = mesh.topology().dim()
    own = mesh.topology().ghost_offset(d)
    dm = V.dofmap()
    to_global = np.asarray(dm.tabulate_local_to_global_dofs(), dtype=np.int64)
    # every cell has the element's dofs (Lagrange and mixed spaces of them): one compiled
    # call for all, not a pybind call a cell -- on every rank, a rank without cells too,
    # since the first call compiles (dolfin's JIT is collective)
    n = V.element().space_dimension()
    per_cell = cell_dofs(dm, np.arange(own), n, comm)
    # the dofmap's integer type, from a rank that has a cell
    dtype = next((k for k in comm.allgather(dm.cell_dofs(0).dtype.str if own else None) if k),
                 np.dtype(np.int32).str)
    t.gc = np.asarray(mesh.topology().global_indices(d), dtype=np.uint64)[:own]
    t.dofs = to_global[per_cell].astype(dtype)
    t.vertices_per_cell = mesh.cells().shape[1]
    t.gdim = mesh.geometry().dim()
    t._make_x = lambda: mesh.coordinates()[mesh.cells()[:own]]
    t._make_vgid = lambda: np.asarray(mesh.topology().global_indices(0),
                                      dtype=np.int64)[mesh.cells()[:own]]
    t.owned = tuple(int(v) for v in dm.ownership_range())
    t.n_global = int(V.dim())
    t.comm = comm
    t.n_cells = int(mesh.num_entities_global(d))
    t.signature = V.element().signature()
    t.components, first = [], 0
    for S in _dolfin_leaves(V):
        degree = int(S.ufl_element().degree())
        k = S.element().space_dimension()
        t.components.append((first, k, degree))
        first += k


def _dolfin_leaves(V):
    """V's scalar components, a vector sub-space's unrolled, in the cell dofs' order."""
    n = V.num_sub_spaces()
    return [V] if n == 0 else [leaf for i in range(n) for leaf in _dolfin_leaves(V.sub(i))]


def _dolfinx_space(t, V, U):
    from femtools.dolfinx.cells import (
        components, dolfin_cell_dofs, lagrange_degree, signature, ufc_vertices)
    if U is None and hasattr(V, "unique_dofs"):     # a femtools.backend.dolfinx space
        U = V.unique_dofs()
    if U is None:
        raise ValueError("tables: a DOLFINx space comes with its UniqueDofs, as (V, U)")
    mesh = V.mesh
    d = mesh.topology.dim
    t.gc, D = dolfin_cell_dofs(V)
    dtype = np.int32 if U.n_global < 2**31 else np.int64    # dolfin's int32 when it fits
    t.dofs = U.compact[D].astype(dtype)
    t.vertices_per_cell = mesh.geometry.dofmap.shape[1]
    t.gdim = mesh.geometry.dim
    t._make_x = lambda: mesh.geometry.x[ufc_vertices(mesh)][:, :, :mesh.geometry.dim]
    t._make_vgid = lambda: np.asarray(mesh.geometry.input_global_indices,
                                      dtype=np.int64)[ufc_vertices(mesh)]
    t.owned = tuple(int(v) for v in U.owned_range)
    t.n_global = int(U.n_global)
    t.comm = mesh.comm
    t.n_cells = int(mesh.topology.index_map(d).size_global)
    t.signature = signature(V)
    t.components, first = [], 0
    for S in components(V):           # a vector (sub-)space's components too, unrolled
        degree = lagrange_degree(S)
        k = _n_nodes(d, degree)
        t.components.append((first, k, degree))
        first += k


class FacetTables:
    """A mesh's local topology: `x` every local vertex's coordinates (gdim columns),
    `cell_vertices` and `cell_facets` this rank's own cells' (local indices),
    `facet_vertices` every local facet's."""


def facet_tables(mesh):
    """The local topology of a dolfin or DOLFINx mesh (FacetTables)."""
    f = FacetTables()
    if is_dolfinx(mesh):
        import dolfinx.mesh as dm
        top = mesh.topology
        d = top.dim
        for a, b in ((d, 0), (d, d - 1), (d - 1, 0), (0, d)):
            top.create_connectivity(a, b)
        own = top.index_map(d).size_local
        nv = top.index_map(0).size_local + top.index_map(0).num_ghosts
        nodes = dm.entities_to_geometry(mesh, 0, np.arange(nv, dtype=np.int32)).reshape(-1)
        f.x = mesh.geometry.x[nodes][:, :mesh.geometry.dim]
        f.cell_vertices = top.connectivity(d, 0).array.reshape(-1, d + 1)[:own].astype(np.int64)
        f.cell_facets = top.connectivity(d, d - 1).array.reshape(-1, d + 1)[:own].astype(
            np.int64)
        f.facet_vertices = top.connectivity(d - 1, 0).array.reshape(-1, d).astype(np.int64)
        return f
    d = mesh.topology().dim()
    mesh.init(d - 1, 0)
    mesh.init(d, d - 1)
    own = mesh.topology().ghost_offset(d)
    f.x = mesh.coordinates()
    f.cell_vertices = mesh.cells()[:own].astype(np.int64)
    f.cell_facets = np.array(mesh.topology()(d, d - 1)(), dtype=np.int64).reshape(
        -1, d + 1)[:own]
    f.facet_vertices = np.array(mesh.topology()(d - 1, 0)(), dtype=np.int64).reshape(-1, d)
    return f


def read_values(path, name, t):
    """The Function `name` in `path` (dolfin's layout: written by dolfin, by
    femtools.dolfin_layout or femtools.dolfinx.checkpoint, on any number of ranks and
    either backend) on the space whose tables `t` are: this rank's owned vector entries.
    Through the file's own cells, cell dofs and vector, cell by cell; under MPI each rank
    reads its blocks of the four datasets in one opening of the file. Collective."""
    import h5py
    from femtools.hdf5_rows import (
        _AHEAD, _ahead_key, agreed, fetch_rows, read_ahead, rows_of_labels)
    comm = t.comm
    found = None
    if comm.rank == 0:
        with h5py.File(path, "r") as h5:
            a = h5[name].attrs
            found = bytes(a["signature"]).decode() if "signature" in a else None
    found = comm.bcast(found, root=0)
    if found is not None and found != t.signature:
        raise ValueError("{}:{} holds {}, not {}".format(path, name, found, t.signature))
    n, k = t.dofs.shape
    datasets = [name + s for s in ("/cells", "/x_cell_dofs", "/cell_dofs", "/vector_0")]
    read_ahead(comm, [(path, d) for d in datasets], stagger=False)
    try:
        rows = rows_of_labels(path, name + "/cells", t.gc.astype(np.int64), comm)
        bounds = fetch_rows(path, name + "/x_cell_dofs", np.r_[rows, rows + 1],
                            comm).astype(np.int64)
        starts, ends = bounds[:n], bounds[n:]
        if agreed(comm, bool(np.any(ends - starts != k))):
            raise ValueError("{}:{} has another number of dofs a cell than {}".format(
                path, name, t.signature))
        flat = (starts[:, None] + np.arange(k)).reshape(-1)
        file_dofs = fetch_rows(path, name + "/cell_dofs", flat, comm).astype(np.int64)
        vals = fetch_rows(path, name + "/vector_0", file_dofs, comm)
    finally:
        for d in datasets:            # a block not served (an error) is not kept
            _AHEAD.pop(_ahead_key(path, d), None)
    lo, hi = t.owned
    mine = t.dofs.reshape(-1).astype(np.int64)
    own = (mine >= lo) & (mine < hi)
    out = np.empty(hi - lo)
    out[mine[own] - lo] = vals[own]
    covered = np.zeros(hi - lo, dtype=bool)
    covered[mine[own] - lo] = True
    if agreed(comm, not covered.all()):
        raise RuntimeError("{}:{}: some owned dofs are in none of this rank's cells".format(
            path, name))
    return out
