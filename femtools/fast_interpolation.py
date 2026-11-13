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
        if len(V.component()):                   # a component of a mixed space
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
