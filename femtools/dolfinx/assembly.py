"""Assembly over unique dofs: matrices folded by the dof map, right-hand sides folded by
one scatter, Dirichlet values set on compact vectors.

    U = UniqueDofs(V, *periodic_box(V, periodic=(True, False)))
    A = create_matrix(a, U)               # exact preallocation, no slave rows
    assemble_matrix(A, a, bcs)            # as often as needed
    b = U.create_vector()
    assemble_vector(b, L, U, a=a, bcs=bcs)
    ksp.solve(b, x); U.prolong(x, u)

`a` and `L` are compiled forms (`dolfinx.fem.form`). For a rectangular form give the
column space's UniqueDofs as `cols`. Dirichlet conditions on periodic faces must agree on
slave and master (as periodic data does). Matrices over blocked spaces are not supported
yet (scalar and mixed spaces are).
"""
import numpy as np


def create_matrix(a, rows, cols=None):
    """The matrix of bilinear form `a` over unique dofs (`rows`, `cols`: UniqueDofs),
    preallocated exactly: one plain assembly gives the pattern (DOLFINx's sparsity
    pattern does not expose the global columns of the ghost entries it adds), which is
    folded onto unique dofs, each entry sent to the rank owning its row. Collective."""
    from dolfinx.fem import petsc
    from petsc4py import PETSc
    from femtools.hdf5_rows import route, unique_rows

    cols = rows if cols is None else cols
    if rows.bs != 1 or cols.bs != 1:
        raise NotImplementedError("matrices over blocked spaces")
    comm = rows.comm
    A = petsc.create_matrix(a)
    petsc.assemble_matrix(A, a)  # values do not matter, the pattern does
    A.assemble()
    indptr, gcols, values = A.getValuesCSR()
    A.destroy()
    del values
    local_rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    # An owned row may hold columns that are not local dofs here (entries another rank
    # assembled for its ghost row), so the columns are looked up globally.
    pairs = unique_rows(np.stack([np.asarray(rows.compact, np.int64)[local_rows],
                                  cols.compact_of_global(gcols).astype(np.int64)], axis=1))
    del indptr, gcols, local_rows
    if comm.size > 1:   # entries of folded rows owned elsewhere to their owner
        starts = np.array(comm.allgather(rows.start), dtype=np.int64)
        owner = np.searchsorted(starts, pairs[:, 0], side="right") - 1
        pairs = unique_rows(route(comm, owner, pairs).received[0])
    r = pairs[:, 0] - rows.start
    c = pairs[:, 1]
    diag = (c >= cols.start) & (c < cols.start + cols.n_owned)
    d_nnz = np.bincount(r[diag], minlength=rows.n_owned)
    o_nnz = np.bincount(r[~diag], minlength=rows.n_owned)

    K = PETSc.Mat().createAIJ(((rows.n_owned, rows.n_global), (cols.n_owned, cols.n_global)),
                              nnz=(d_nnz.astype(PETSc.IntType), o_nnz.astype(PETSc.IntType)),
                              comm=comm)
    K.setLGMap(rows.lgmap, cols.lgmap)
    K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    return K


def assemble_matrix(K, a, bcs=(), diag=1.0):
    """Assemble form `a` into `K` (from `create_matrix`), Dirichlet rows and columns
    zeroed with `diag` on the diagonal."""
    from dolfinx.fem import petsc

    K.zeroEntries()
    petsc.assemble_matrix(K, a, bcs=list(bcs), diag=diag)
    K.assemble()
    return K


def assemble_vector(b, L, space, a=None, bcs=(), x0=None, alpha=1.0):
    """Assemble linear form `L` into compact vector `b` over `space` (UniqueDofs): with
    `bcs` and the system's bilinear form `a`, lifted and with Dirichlet values set
    (`x0`, `alpha` as in DOLFINx: b - alpha*A(g - x0), values alpha*(g - x0))."""
    from dolfinx.fem import petsc

    s = space.plain_vector(0)
    with s.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(s, L)
    x0p = None if x0 is None else space.prolong_vec(x0, space.plain_vector(1))
    if bcs and a is not None:
        petsc.apply_lifting(s, [a], bcs=[list(bcs)], x0=None if x0p is None else [x0p],
                            alpha=alpha)
    space.fold(s, b)
    if bcs:
        _set_bc(b, bcs, space, x0p, alpha)
    return b


def set_bc(b, bcs, space, x0=None, alpha=1.0):
    """Dirichlet values alpha*(g - x0) into compact vector `b` over `space`."""
    x0p = None if x0 is None else space.prolong_vec(x0, space.plain_vector(1))
    return _set_bc(b, bcs, space, x0p, alpha)


def _set_bc(b, bcs, space, x0p, alpha):
    """DOLFINx computes the values in the space's own layout; the owned, unique
    Dirichlet dofs are copied over (a slave's value is its master's, set by its owner)."""
    from dolfinx.fem import petsc

    s = space.plain_vector(2)
    with s.localForm() as loc:
        loc.set(0.0)
    petsc.set_bc(s, list(bcs), x0=x0p, alpha=alpha)
    vals = s.getArray(readonly=True)
    out = b.getArray()
    lo, hi = space.owned_range
    for bc in bcs:
        dofs, n_owned = bc.dof_indices()
        own = dofs[:n_owned]
        own = own[~space.is_slave[own]]
        target = space.compact[own]
        keep = (target >= lo) & (target < hi)
        out[target[keep] - lo] = vals[own[keep]]
    return b


def constant_nullspace(space):
    """The constants over unique dofs as a PETSc null space (a scalar space)."""
    from petsc4py import PETSc

    v = space.create_vector()
    v.set(1.0)
    v.normalize()
    return PETSc.NullSpace().create(vectors=[v], comm=space.comm)
