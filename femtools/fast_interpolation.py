"""dolfin.interpolate between Lagrange spaces on one mesh, as a sparse matrix product.

    from femtools.fast_interpolation import interpolate, values
    g = interpolate(f, V)           # dolfin.interpolate(f, V), into a new Function
    interpolate(f, V, out=g)        # into g
    x = values(f, V)                # this rank's owned values, as an array

For Lagrange elements interpolation is point evaluation, a fixed linear map from the
source's dofs to the target's: built once per pair of spaces as dolfin's transfer
matrix (PETScDMCollection.create_transfer_matrix), every call is then one sparse
matrix-vector product, where dolfin.interpolate evaluates the source element cell by
cell and builds a Function every call.

A scalar P1 -> P2 pair is built from the two dofmaps instead (`p1_to_p2`, on either
backend): each P2 vertex dof takes its vertex's P1 dof, each edge dof half of each end's,
with no point search and from local cells alone.

Any degrees, scalar, vector or tensor; on a periodic space; in parallel. The source may be a
component of a mixed Function (from Function.split()), whose vector is the mixed
one: its values are first gathered into the collapsed component space, ghosts
included, through the collapse's dof map.
"""
import numpy as np

from femtools.lru import LRU

_TRANSFERS = LRU(64)


def transfer(V, W):
    """The transfer from the source space V (a space, or a component of a mixed one)
    to the target space W, as a PETSc matrix built once per pair. For a component it
    acts on the mixed vector: dolfin's transfer from the collapsed space, times the
    matrix that picks the component's values out of the mixed vector. Collective."""
    import dolfin as df
    key = (V.id(), W.id())
    T = _TRANSFERS.get(key)
    if T is None:
        _check(V, W)
        if _scalar_p1_p2(V, W):
            T = p1_to_p2(V, W)
        elif len(V.component()):                 # a component of a mixed space
            source = V.collapse()
            D = df.PETScDMCollection.create_transfer_matrix(source, W).mat()
            P = _selection(V, source)
            product = D.matMult(P)
            # a product keeps its factors for a reuse, as long as it lives: the copy
            # alone is kept, and the factors destroyed with the product
            T = product.duplicate(copy=True)
            for M in (product, P, D):
                M.destroy()
        else:
            T = df.PETScDMCollection.create_transfer_matrix(V, W).mat()
        _TRANSFERS[key] = T
    return T


def p1_to_p2(S1, S2, chunk=1 << 20):
    """The P1 -> P2 interpolation on one mesh as a petsc4py Mat (rows S2's vector,
    columns S1's), from the two dofmaps alone: each P2 vertex dof takes its vertex's P1 dof
    with weight 1, each edge dof its two ends' with 1/2 (the P1 basis at the P2 nodes, in
    both elements' local order). The restriction PCMG wants is its transpose. Legacy
    dolfin, or femtools.backend.dolfinx (over the unique dofs; periodic spaces alike).
    Collective.

    No point search, no bounding-box tree and no copy of the mesh (dolfin's
    `create_transfer_matrix` has all three, a transient on top of a solve's peak at the
    hundreds of ranks stokes-beadpack's `--pmg` runs on): the cell dof columns a block of
    `chunk` cells at a time, and the matrix's own at most two entries a row. Every owned
    P2 dof lies in a local cell, so each rank sets its own rows; a column may be any
    rank's P1 dof. The periodic constraint is in both dofmaps and pairs vertices alike."""
    from femtools.backend import is_dolfinx
    if is_dolfinx(S2):
        return _p1_to_p2_dolfinx(S1, S2, chunk)
    return _p1_to_p2_dolfin(S1, S2, chunk)


def _p1_weights(nodes):
    """The P1 basis at the P2 reference `nodes` (barycentric: 0, 1/2 or 1), refusing
    anything else: each row is one 1 (a vertex) or two halves (an edge)."""
    W = np.array([[1.0 - sum(x)] + list(x) for x in np.asarray(nodes, dtype=float)])
    nnz = (np.abs(W) > 1e-12).sum(axis=1)
    if not (np.isin(nnz, (1, 2)).all() and np.allclose(W.sum(axis=1), 1.0)
            and np.allclose(W, (np.abs(W) > 1e-12) / nnz[:, None])):
        raise RuntimeError("p1_to_p2: unexpected P2 reference nodes")
    return np.abs(W) > 1e-12, nnz


def _p1_to_p2_csr(blocks, nz, nnz, m, n_global2, n1, n_global1, comm):
    """The interpolation's CSR from (P2 rows, P1 columns) of each block of cells: rows
    the owned P2 dofs' (-1 for another rank's), columns the P1 dofs' global indices."""
    from petsc4py import PETSc
    n2 = len(nnz)
    cols = np.full((m, 2), -1, dtype=np.int64)
    count = np.zeros(m, dtype=np.int64)
    for rows, c1 in blocks:
        for j in range(n2):
            own = rows[:, j] >= 0
            r = rows[own, j]
            for t, k in enumerate(np.flatnonzero(nz[j])):
                cols[r, t] = c1[own, k]
            count[r] = nnz[j]
    from femtools.hdf5_rows import agreed               # every rank raises, or none
    if agreed(comm, (count == 0).any()):
        raise RuntimeError("p1_to_p2: {} of {} owned P2 dofs are in no local cell"
                           .format(int((count == 0).sum()), m))
    cols.sort(axis=1)                                     # -1 first, then ascending
    keep = cols >= 0
    if agreed(comm, (keep.all(axis=1) & (cols[:, 0] == cols[:, 1])).any()):
        # an edge whose two ends are one periodic vertex: a mesh one cell across a period
        raise RuntimeError("p1_to_p2: an edge joins a vertex to its own periodic image")
    indptr = np.r_[0, np.cumsum(count)].astype(PETSc.IntType)
    indices = cols[keep].astype(PETSc.IntType)
    values = np.repeat(1.0 / count, count).astype(PETSc.ScalarType)
    P = PETSc.Mat().createAIJ(size=((m, n_global2), (n1, n_global1)),
                              csr=(indptr, indices, values), comm=comm)
    P.assemble()
    return P


def _p1_to_p2_dolfin(S1, S2, chunk):
    import FIAT
    from femtools.periodic import cell_dofs
    mesh = S2.mesh()
    tdim = mesh.topology().dim()
    nz, nnz = _p1_weights([list(lf.get_point_dict())[0] for lf in
                           FIAT.Lagrange(FIAT.ufc_simplex(tdim), 2).dual_basis()])
    dm2, dm1 = S2.dofmap(), S1.dofmap()
    lo2, hi2 = dm2.ownership_range()
    lo1, hi1 = dm1.ownership_range()
    l2g1 = np.asarray(dm1.tabulate_local_to_global_dofs(), dtype=np.int64)
    m = hi2 - lo2                # owned local dofs are 0..m-1 (block size 1)
    ncell = mesh.num_cells()
    comm = mesh.mpi_comm()
    cell_dofs(dm2, np.zeros(0, np.int64), nz.shape[0], comm)   # compiled on the mesh's
    #                                    ranks, every one (a rank may hold no cell)

    def blocks():
        for start in range(0, ncell, chunk):
            block = np.arange(start, min(start + chunk, ncell), dtype=np.int64)
            d2 = cell_dofs(dm2, block, nz.shape[0], comm)
            yield np.where(d2 < m, d2, -1), l2g1[cell_dofs(dm1, block, nz.shape[1], comm)]
    return _p1_to_p2_csr(blocks(), nz, nnz, m, dm2.global_dimension(), hi1 - lo1,
                         dm1.global_dimension(), mesh.mpi_comm())


def _p1_to_p2_dolfinx(S1, S2, chunk):
    U2, U1 = S2.unique_dofs(), S1.unique_dofs()
    if U2.bs != 1 or U1.bs != 1:
        raise NotImplementedError("p1_to_p2: scalar spaces")
    nz, nnz = _p1_weights(S2.element.interpolation_points)
    D2, D1 = np.asarray(S2.dofmap.list), np.asarray(S1.dofmap.list)
    if D2.shape[1] != nz.shape[0] or D1.shape[1] != nz.shape[1]:
        raise ValueError("p1_to_p2: a P1 and a P2 space on one mesh")
    lo2, hi2 = U2.owned_range

    def blocks():
        for start in range(0, len(D2), chunk):
            c2 = U2.compact[D2[start:start + chunk]]
            own = (c2 >= lo2) & (c2 < hi2)
            yield np.where(own, c2 - lo2, -1), U1.compact[D1[start:start + chunk]]
    return _p1_to_p2_csr(blocks(), nz, nnz, U2.n_owned, U2.n_global, U1.n_owned,
                         U1.n_global, S2.mesh.comm)


def _scalar_p1_p2(V, W):
    """Whole scalar Lagrange spaces of degrees 1 and 2 (legacy dolfin)."""
    if len(V.component()):
        return False
    a, b = V.ufl_element(), W.ufl_element()
    return (type(a).__name__ == type(b).__name__ == "FiniteElement"
            and a.family() == b.family() == "Lagrange" and a.degree() == 1
            and b.degree() == 2)


def forget(V):
    """Drop every transfer onto or from the space V, and the values buffer on it: for the
    owner of a space it built for a while (`femtools.xdmf`), whose transfers would
    otherwise stay until 64 newer ones pushed them out."""
    i = V.id()
    for key in [k for k in _TRANSFERS if i in k]:
        T = _TRANSFERS.pop(key)
        if hasattr(T, 'destroy'):
            T.destroy()


#: cells taken at a time when matching a component's dofs with its collapsed space's
CELL_BLOCK = 100_000


def _selection(V, source):
    """The matrix from the mixed vector to the component's collapsed space: a one in
    each row, at the mixed dof that the row's collapsed dof is. The two are matched
    cell by cell through their dofmaps (as `collapse` matches them), a block of cells at
    a time, into one array of the collapsed space's local dofs: `collapse`'s own map is a
    Python dict of every dof. Collective."""
    import dolfin as df
    from petsc4py import PETSc
    from femtools.periodic import cell_dofs
    mesh = source.mesh()
    comm = mesh.mpi_comm()
    owned = df.IndexMap.MapSize.OWNED
    n = source.dofmap().index_map().size(owned)
    k = source.element().space_dimension()
    parent = np.full(n, -1, dtype=np.int64)
    n_cells = mesh.num_cells()
    for start in range(0, max(n_cells, 1), CELL_BLOCK):
        cells = np.arange(start, min(start + CELL_BLOCK, n_cells), dtype=np.int64)
        collapsed = cell_dofs(source.dofmap(), cells, k, comm).ravel()
        mixed = cell_dofs(V.dofmap(), cells, k, comm).ravel()
        mine = collapsed < n
        parent[collapsed[mine]] = mixed[mine]
    to_global = V.dofmap().tabulate_local_to_global_dofs()
    lo, hi = V.dofmap().ownership_range()
    cols = np.asarray(to_global[parent], dtype=PETSc.IntType)
    rows = np.arange(n + 1, dtype=PETSc.IntType)
    P = PETSc.Mat().createAIJ([(n, PETSc.DECIDE), (hi - lo, PETSc.DECIDE)],
                              csr=(rows, cols, np.ones(n)), comm=comm)
    P.assemble()
    return P


def interpolate(f, V, out=None):
    """dolfin.interpolate(f, V) for Lagrange spaces on one mesh: into out (a Function
    on V, made if None), which is returned."""
    from dolfin import as_backend_type, Function
    T = transfer(f.function_space(), V)
    out = Function(V) if out is None else out
    T.mult(as_backend_type(f.vector()).vec(), as_backend_type(out.vector()).vec())
    out.vector().apply("insert")
    return out


def values(f, V):
    """The values of f at V's owned dofs, as an array: dolfin.interpolate(f, V)'s
    get_local(), without a Function to hold them."""
    from dolfin import as_backend_type
    key = ("out", V.id())
    y = _TRANSFERS.get(key)
    if y is None:
        from dolfin import Function
        y = _TRANSFERS[key] = Function(V).vector()
    transfer(f.function_space(), V).mult(as_backend_type(f.vector()).vec(),
                                         as_backend_type(y).vec())
    return y.get_local()


def _check(V, W):
    if len(W.component()):
        raise ValueError("fast_interpolation: the target must be a whole space, not a "
                         "component of a mixed one")
    for space, what in ((W, "target"), (V, "source")):
        el = space.ufl_element()
        kind = type(el).__name__
        base = el.sub_elements()[0] if kind in ("VectorElement", "TensorElement") else el
        if kind not in ("FiniteElement", "VectorElement", "TensorElement") \
                or base.family() not in ("Lagrange", "Q"):
            raise ValueError("fast_interpolation: the {} must be a scalar, vector or tensor "
                             "continuous Lagrange space, not {}".format(what, el))
    if V.ufl_element().value_shape() != W.ufl_element().value_shape():
        raise ValueError("fast_interpolation: the source has value shape {} and the target "
                         "{}".format(V.ufl_element().value_shape(), W.ufl_element().value_shape()))
    if V.mesh().id() != W.mesh().id():
        raise ValueError("fast_interpolation: the source and the target must share a mesh")
