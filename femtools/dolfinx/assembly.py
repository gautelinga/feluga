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

A matrix is assembled by DOLFINx's assembler with an inserter of femtools' own
(cpp/folded_assembly.cpp), which adds each element matrix straight into PETSc's arrays
instead of through MatSetValuesLocal's index mapping and lookups.
"""
import hashlib

import numpy as np

from femtools.lru import WeakCache

# Preallocations per (rows, cols): {key of the integration domains: (d_nnz, o_nnz)}
_PATTERNS = WeakCache()
# Pairs of dofs enumerated at a time (int64 keys: 8 bytes each)
_CHUNK = 1 << 21
# [the C++ inserter, or None where it cannot be built], once tried
_INSERTER = []


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
    integration domains, and shared by the forms that have the same. The PETSc objects
    Python has dropped on the rows' communicator are destroyed first
    (femtools.garbage). Collective."""
    from mpi4py import MPI
    from petsc4py import PETSc

    from femtools.garbage import cleanup

    cols = rows if cols is None else cols
    if rows.bs != 1 or cols.bs != 1:
        raise NotImplementedError("matrices over blocked spaces")
    comm = rows.comm
    cleanup(comm)  # matrices dropped since the last one, destroyed before another is made
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


def _coupled_cells(a):
    """Per integral of compiled form `a`, the cells whose dofs it couples, one row per
    integration entity: (n, 1) for cells, exterior facets, ridges and vertices (the
    entity's cell), (n, 2) for interior facets (the facet's two cells)."""
    cpp = a._cpp_object
    for itype in cpp.integral_types:
        for i in range(cpp.num_integrals(itype, 0)):
            d = np.asarray(cpp.domains(itype, i))
            yield d.reshape(-1, 1) if d.ndim == 1 else d[..., 0].reshape(len(d), -1)


def _preallocation(a, rows, cols):
    """(d_nnz, o_nnz) of the folded matrix of `a`, from the dofmaps alone: every (row,
    column) pair of dofs that an integration entity couples (all dofs of its cells, as
    in DOLFINx's sparsity pattern, which adds no diagonal of its own) is taken to
    compact indices, sent to the owner of its row if another rank owns that, and
    counted once. Pairs are made at most about `_CHUNK` at a time and kept as one int64
    key each, split by row range so that each range is made unique on its own.
    Collective."""
    from petsc4py import PETSc
    from femtools.hdf5_rows import route, unique_rows

    comm = rows.comm
    spaces = a._cpp_object.function_spaces
    if any(V.mesh is not a._cpp_object.mesh for V in spaces):
        raise NotImplementedError("forms whose spaces are on another mesh")
    start, n = rows.start, rows.n_owned
    clo, chi = cols.owned_range
    width = max(cols.n_global, 1)
    if n * width >= 2 ** 63:
        raise NotImplementedError("more (row, column) pairs than an int64 numbers")
    rmap, cmap = (V.dofmap.map() for V in spaces)
    per_cell = rmap.shape[1] * cmap.shape[1]
    domains = list(_coupled_cells(a))
    total = sum(len(c) * c.shape[1] ** 2 * per_cell for c in domains)
    # keys of the owned rows, split in row ranges of about _CHUNK keys
    n_ranges = max(1, min(n, -(-total // _CHUNK)))
    edges = np.linspace(0, n, n_ranges + 1).astype(np.int64)
    bounds = edges[1:-1] * width
    held = [[] for _ in range(n_ranges)]

    def hold(keys):
        keys = np.unique(keys)
        for r, part in enumerate(np.split(keys, np.searchsorted(keys, bounds))):
            if len(part):
                held[r].append(part)

    far = []  # pairs in rows owned elsewhere: (compact row, compact column)
    for cells in domains:
        # The cells in the order of their first row: a row's pairs then come mostly in
        # one chunk, where they are made unique before they are held.
        step = max(1, _CHUNK // (cells.shape[1] ** 2 * per_cell))
        first = np.empty(len(cells), np.int64)
        for s in range(0, len(cells), step):
            c = cells[s:s + step]
            first[s:s + step] = rows.compact[rmap[c].reshape(len(c), -1)].min(axis=1)
        cells = cells[np.argsort(first, kind="stable")]
        del first
        for s in range(0, len(cells), step):
            c = cells[s:s + step]
            R = rows.compact[rmap[c].reshape(len(c), -1)] - start
            C = cols.compact[cmap[c].reshape(len(c), -1)]
            mine = (R >= 0) & (R < n)
            hold(((R * width)[:, :, None] + C[:, None, :])[mine].ravel())
            if not mine.all():
                ent, k = np.nonzero(~mine)
                far.append(unique_rows(np.stack(
                    [np.repeat(R[ent, k] + start, C.shape[1]), C[ent].ravel()], axis=1)))
    if comm.size > 1:
        pairs = unique_rows(np.concatenate(far)) if far else np.zeros((0, 2), np.int64)
        del far
        starts = np.array(comm.allgather(start), dtype=np.int64)
        owner = np.searchsorted(starts, pairs[:, 0], side="right") - 1
        pairs = route(comm, owner, pairs).received[0]
        hold((pairs[:, 0] - start) * width + pairs[:, 1])
        del pairs
    nnz = np.zeros(n, np.int64)
    d_nnz = np.zeros(n, np.int64)
    for r in range(n_ranges):
        keys = np.unique(np.concatenate(held[r])) if held[r] else np.zeros(0, np.int64)
        held[r] = None
        lo, hi = edges[r], edges[r + 1]
        row, col = np.divmod(keys, width)
        row -= lo
        nnz[lo:hi] = np.bincount(row, minlength=hi - lo)
        d_nnz[lo:hi] = np.bincount(row[(col >= clo) & (col < chi)], minlength=hi - lo)
    return d_nnz.astype(PETSc.IntType), (nnz - d_nnz).astype(PETSc.IntType)


def _inserter(comm):
    """The C++ extension that adds element matrices straight into a matrix's AIJ arrays
    (cpp/folded_assembly.cpp), or None if it cannot be built here: then rank 0 warns
    once and matrices are assembled through MatSetValuesLocal, unless the environment
    sets FEMTOOLS_REQUIRE_EXTENSIONS, when every rank raises. Collective."""
    import os
    if _INSERTER:
        return _INSERTER[0]
    from femtools.dolfinx.compiled import module
    try:
        _INSERTER.append(module(comm, "folded_assembly", "femtools.dolfinx.assembly"))
    except RuntimeError as e:
        if os.environ.get("FEMTOOLS_REQUIRE_EXTENSIONS"):
            raise
        if comm.rank == 0:
            import sys
            print("femtools: assembling matrices through PETSc's MatSetValuesLocal, more "
                  "slowly, since the C++ inserter could not be built:\n{}".format(e),
                  file=sys.stderr)
        _INSERTER.append(None)
    return _INSERTER[0]


def assemble_matrix(K, a):
    """Assemble form `a` into `K` (from `create_matrix`). From the second assembly on,
    the element matrices are added straight into K's arrays (its pattern then final);
    the first goes through MatSetValuesLocal. Either way the extension packs the form's
    coefficients a block of cells at a time. A value outside K's pattern in a row owned
    here raises on every rank; in a row owned elsewhere, PETSc refuses it there.
    Collective."""
    from mpi4py import MPI
    from dolfinx.fem import petsc

    comm = K.comm.tompi4py()
    inserter = _inserter(comm) if a.dtype == np.float64 else None
    K.zeroEntries()
    missed = inserter.assemble(a._cpp_object, K.handle) if inserter is not None else -2
    if missed == -1 and inserter.assemble_local(a._cpp_object, K.handle):
        pass                            # the first assembly, through the local maps
    elif missed < 0:
        petsc.assemble_matrix(K, a)
    elif comm.allreduce(missed, op=MPI.SUM):
        raise RuntimeError("assemble_matrix: {} values of the form fall outside the "
                           "matrix's pattern".format(comm.allreduce(missed, op=MPI.SUM)))
    K.assemble()
    return K


def assemble_vector(b, L, space):
    """Assemble linear form `L` into compact vector `b` over `space` (UniqueDofs): by
    the C++ extension, which packs the coefficients a block of cells at a time, else by
    DOLFINx. Collective."""
    from dolfinx.fem import petsc

    s = space.plain_vector()
    inserter = _inserter(space.comm) if L.dtype == np.float64 else None
    with s.localForm() as loc:
        loc.set(0.0)
        done = inserter is not None and inserter.assemble_vector(L._cpp_object,
                                                                 loc.getArray())
    if not done:
        petsc.assemble_vector(s, L)
    return space.fold(s, b)


def constant_nullspace(space):
    """The constants over unique dofs as a PETSc null space (a scalar space)."""
    from petsc4py import PETSc

    v = space.create_vector()
    v.set(1.0)
    v.normalize()
    return PETSc.NullSpace().create(vectors=[v], comm=space.comm)
