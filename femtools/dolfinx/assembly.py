"""Assembly over unique dofs: matrices folded by the dof map, right-hand sides folded by
one scatter.

    U = UniqueDofs(V, *periodic_box(V, periodic=(True, False)))
    A = create_matrix(a, U)               # exact preallocation, no slave rows
    assemble_matrix(A, a)                 # as often as needed
    b = U.create_vector()
    assemble_vector(b, L, U)

`a` and `L` are compiled forms (`dolfinx.fem.form`). For a rectangular form give the
column space's UniqueDofs as `cols`. Dirichlet conditions act on the assembled compact
tensors (`femtools.backend.dolfinx.DirichletBC`). Matrices over blocked spaces are not
supported (scalar and mixed spaces are).
"""
import hashlib

import numpy as np

from femtools.lru import WeakCache

# Preallocations per (rows, cols): {key of the integration domains: (d_nnz, o_nnz)}
_PATTERNS = WeakCache()


def _domains_key(a, comm):
    """What decides the pattern of form `a` besides its spaces: the entities each of its
    integrals runs over, on every rank (DOLFINx couples all dofs of those entities,
    whatever the integrand). The same on every rank. Collective."""
    h = hashlib.blake2b(digest_size=16)
    cpp = a._cpp_object
    for itype in sorted(cpp.integral_types, key=lambda t: t.name):
        for i in range(cpp.num_integrals(itype, 0)):
            d = np.ascontiguousarray(cpp.domains(itype, i))
            h.update("{}:{}:{}:{}".format(itype.name, i, d.dtype, d.shape).encode())
            h.update(d.tobytes())
    return tuple(comm.allgather(h.digest()))


def create_matrix(a, rows, cols=None):
    """The matrix of bilinear form `a` over unique dofs (`rows`, `cols`: UniqueDofs),
    preallocated exactly. The preallocation is computed once per pair of UniqueDofs and
    integration domains, and shared by the forms that have the same. Collective."""
    from mpi4py import MPI
    from petsc4py import PETSc

    cols = rows if cols is None else cols
    if rows.bs != 1 or cols.bs != 1:
        raise NotImplementedError("matrices over blocked spaces")
    comm = rows.comm
    patterns = _PATTERNS.get((rows, cols), dict)
    key = _domains_key(a, comm)
    nnz = patterns.get(key)
    if not comm.allreduce(nnz is not None, op=MPI.LAND):
        nnz = patterns[key] = _preallocation(a, rows, cols)
    K = PETSc.Mat().createAIJ(((rows.n_owned, rows.n_global), (cols.n_owned, cols.n_global)),
                              nnz=nnz, comm=comm)
    K.setLGMap(rows.lgmap, cols.lgmap)
    K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    return K


def _preallocation(a, rows, cols):
    """(d_nnz, o_nnz) of the folded matrix of `a`. One plain assembly gives the pattern
    (DOLFINx's sparsity pattern does not expose the global columns of the ghost entries
    it adds), which is folded onto unique dofs. The fold is one to one except at
    periodic slaves, so only the entries in a slave's row or column, or in a column not
    known on this rank, can meet another: those alone are sent to the owner of their row
    and made unique there, with the other entries of the rows they reach. Collective."""
    from dolfinx.fem import petsc
    from petsc4py import PETSc
    from femtools.hdf5_rows import route, unique_rows

    comm = rows.comm
    A = petsc.create_matrix(a)
    petsc.assemble_matrix(A, a)  # values do not matter, the pattern does
    A.assemble()
    indptr, gcols = A.getValuesCSR()[:2]
    A.destroy()
    n = len(indptr) - 1                         # the space's owned dofs: the rows
    lrow = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    del indptr
    # An owned row may hold columns that are not local dofs here (entries another rank
    # assembled for its ghost row): their owners know them.
    lcol = cols.local_of_global(gcols)
    known = lcol >= 0
    ccol = np.empty(len(gcols), np.int64)
    ccol[known] = cols.compact[lcol[known]]
    ccol[~known] = cols.compact_of_global(gcols[~known])
    del gcols
    loose = ~known
    loose[known] = cols.is_slave[lcol[known]]
    del lcol, known
    loose |= rows.is_slave[lrow]
    clean = ~loose
    clo, chi = cols.owned_range
    start, n_owned = rows.start, rows.n_owned
    # the clean entries: unique, each in the owned row of a local row that is no slave
    own = ~rows.is_slave[:n]
    row_of = rows.compact[:n][own] - start
    diag = (ccol >= clo) & (ccol < chi)
    d_nnz = np.zeros(n_owned, np.int64)
    o_nnz = np.zeros(n_owned, np.int64)
    d_nnz[row_of] = np.bincount(lrow[clean & diag], minlength=n)[own]
    o_nnz[row_of] = np.bincount(lrow[clean & ~diag], minlength=n)[own]
    del diag
    pairs = np.stack([rows.compact[lrow[loose]], ccol[loose]], axis=1)
    if comm.size > 1:
        starts = np.array(comm.allgather(start), dtype=np.int64)
        owner = np.searchsorted(starts, pairs[:, 0], side="right") - 1
        pairs = route(comm, owner, pairs).received[0]
    if len(pairs):
        # the rows the loose entries reach, recounted with their clean entries
        touched = np.zeros(n_owned, bool)
        touched[pairs[:, 0] - start] = True
        reached = np.zeros(n, bool)
        reached[rows.owned_unique[touched]] = True
        also = clean & reached[lrow]
        pairs = unique_rows(np.concatenate(
            [pairs, np.stack([rows.compact[lrow[also]], ccol[also]], axis=1)]))
        r = pairs[:, 0] - start
        diag = (pairs[:, 1] >= clo) & (pairs[:, 1] < chi)
        d_nnz[touched] = 0
        o_nnz[touched] = 0
        d_nnz += np.bincount(r[diag], minlength=n_owned)
        o_nnz += np.bincount(r[~diag], minlength=n_owned)
    return d_nnz.astype(PETSc.IntType), o_nnz.astype(PETSc.IntType)


def assemble_matrix(K, a):
    """Assemble form `a` into `K` (from `create_matrix`)."""
    from dolfinx.fem import petsc

    K.zeroEntries()
    petsc.assemble_matrix(K, a)
    K.assemble()
    return K


def assemble_vector(b, L, space):
    """Assemble linear form `L` into compact vector `b` over `space` (UniqueDofs)."""
    from dolfinx.fem import petsc

    s = space.plain_vector()
    with s.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(s, L)
    return space.fold(s, b)


def constant_nullspace(space):
    """The constants over unique dofs as a PETSc null space (a scalar space)."""
    from petsc4py import PETSc

    v = space.create_vector()
    v.set(1.0)
    v.normalize()
    return PETSc.NullSpace().create(vectors=[v], comm=space.comm)
