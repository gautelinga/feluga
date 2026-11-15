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

`create_matrix(a, U, couplings=mask, bcs=bcs)` stores only the blocks of the components
`mask` couples (femtools.couplings), and the diagonal of a square form; `assemble_matrix`
then skips the other blocks' entries by component, and raises if one of them is nonzero.
With Dirichlet conditions `bcs` the rows and columns of the dofs they constrain hold the
diagonal alone, the number of cells holding the dof: the conditions applied symmetrically,
as dolfin's assemble_system applies them (a right-hand side of inhomogeneous ones is the
caller's: g times that diagonal, the columns lifted).

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


def create_matrix(a, rows, cols=None, couplings=None, bcs=()):
    """The matrix of bilinear form `a` over unique dofs (`rows`, `cols`: UniqueDofs),
    preallocated exactly. The preallocation is computed once per pair of UniqueDofs,
    integration domains and couplings, and shared by the forms that have the same.
    `couplings`, a boolean mask (test components x trial components, femtools.couplings),
    keeps only the coupled blocks and, when `rows` is `cols`, the diagonal; Dirichlet
    conditions `bcs` (femtools.backend.dolfinx's, on a square form) keep only the
    diagonal in their dofs' rows and columns. The matrix remembers both for
    `assemble_matrix`. The PETSc objects Python has dropped on the rows'
    communicator are destroyed first (femtools.garbage). Collective."""
    from mpi4py import MPI
    from petsc4py import PETSc

    from femtools.garbage import cleanup

    cols = rows if cols is None else cols
    if rows.bs != 1 or cols.bs != 1:
        raise NotImplementedError("matrices over blocked spaces")
    comm = rows.comm
    cleanup(comm)  # matrices dropped since the last one, destroyed before another is made
    if bcs and rows is not cols:
        raise ValueError("Dirichlet conditions on a matrix whose rows and columns differ")
    filt = None
    if couplings is not None or bcs:
        filt = _Couplings(a, couplings, rows is cols, rows, bcs)
    patterns = _PATTERNS.get((rows, cols), dict)
    key = _domains_key(a, comm) + (None if filt is None else filt.key,)
    nnz = patterns.get(key)
    if not comm.allreduce(nnz is not None, op=MPI.LAND):
        nnz = patterns[key] = _preallocation(a, rows, cols, filt)
    K = PETSc.Mat().createAIJ(((rows.n_owned, rows.n_global), (cols.n_owned, cols.n_global)),
                              nnz=nnz, comm=comm)
    K.setLGMap(rows.lgmap, cols.lgmap)
    K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    if filt is not None:
        _FILTERS[K.handle] = filt
    return K


# The couplings of the matrices create_matrix made with them, by PETSc handle (petsc4py
# objects take no attributes); a handle reused by a later matrix is overwritten there
_FILTERS = {}


class _Couplings:
    """A couplings mask with the component of each local dof of the form's two spaces."""

    def __init__(self, a, mask, square, U, bcs=()):
        from femtools.couplings import dolfinx_labels
        V0, V1 = a._cpp_object.function_spaces
        self.rows = dolfinx_labels(V0)
        self.cols = self.rows if V1 is V0 else dolfinx_labels(V1)
        n0, n1 = int(self.rows.max(initial=-1)) + 1, int(self.cols.max(initial=-1)) + 1
        if mask is None:
            mask = np.ones((n0, n1), dtype=bool)
        self.mask = np.ascontiguousarray(np.asarray(mask, dtype=bool), dtype=np.uint8)
        if self.mask.shape[0] < n0 or self.mask.shape[1] < n1:
            raise ValueError("a couplings mask of shape {} for spaces of {} x {} components"
                             .format(self.mask.shape, n0, n1))
        self.square = bool(square)
        self.has_bcs = len(bcs) > 0
        # the dofs the conditions constrain: owned rows (compact, global) and every local
        # dof (owned and ghost) whose unique dof is one of them
        self.fixed_rows = np.unique(np.concatenate(
            [np.asarray(bc.global_rows(), dtype=np.int64) for bc in bcs]
            or [np.zeros(0, np.int64)]))
        self.fixed = _fixed_local(U, self.fixed_rows) if len(bcs) else \
            np.zeros(0, dtype=np.int8)
        h = hashlib.blake2b(self.fixed_rows.tobytes(), digest_size=16).digest()
        self.key = (self.mask.shape, self.mask.tobytes(), self.square,
                    tuple(U.comm.allgather(h)))

    def pairs(self, R_cells, C_cells, diagonal):
        """Which (row, column) entries of each cell's element matrix are kept: (m, r, c);
        `diagonal` (m, r, c) marks the entries on the matrix's diagonal, which a fixed dof
        keeps."""
        keep = self.mask[self.rows[R_cells][:, :, None], self.cols[C_cells][:, None, :]] != 0
        if len(self.fixed):
            fixed = (self.fixed[R_cells] != 0)[:, :, None] | (self.fixed[C_cells] != 0)[:, None, :]
            keep &= ~fixed | diagonal
        return keep


def _fixed_local(U, rows):
    """1 at each local dof of U's space (owned and ghost) whose unique dof is one of the
    owned compact `rows` of some rank, else 0 (int8): one forward scatter."""
    from petsc4py import PETSc
    c = U.create_vector()
    c.getArray()[np.asarray(rows, dtype=np.int64) - U.owned_range[0]] = 1.0
    local = PETSc.Vec().createSeq(len(U.compact), comm=PETSc.COMM_SELF)
    U._scatter.scatter(c, local, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    out = (local.getArray(readonly=True) > 0.5).astype(np.int8)
    for o in (c, local):
        o.destroy()
    return out


def _coupled_cells(a):
    """Per integral of compiled form `a`, the cells whose dofs it couples, one row per
    integration entity: (n, 1) for cells, exterior facets, ridges and vertices (the
    entity's cell), (n, 2) for interior facets (the facet's two cells)."""
    cpp = a._cpp_object
    for itype in cpp.integral_types:
        for i in range(cpp.num_integrals(itype, 0)):
            d = np.asarray(cpp.domains(itype, i))
            yield d.reshape(-1, 1) if d.ndim == 1 else d[..., 0].reshape(len(d), -1)


def _preallocation(a, rows, cols, filt=None):
    """(d_nnz, o_nnz) of the folded matrix of `a`, from the dofmaps alone: every (row,
    column) pair of dofs that an integration entity couples (all dofs of its cells, as
    in DOLFINx's sparsity pattern, which adds no diagonal of its own; with couplings
    `filt`, those of coupled components, and the diagonal of a square form) is taken to
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
            Rl, Cl = rmap[c].reshape(len(c), -1), cmap[c].reshape(len(c), -1)
            R = rows.compact[Rl] - start
            C = cols.compact[Cl]
            mine = (R >= 0) & (R < n)
            keys = (R * width)[:, :, None] + C[:, None, :]
            if filt is None:
                hold(keys[mine].ravel())
                if not mine.all():
                    ent, k = np.nonzero(~mine)
                    far.append(unique_rows(np.stack(
                        [np.repeat(R[ent, k] + start, C.shape[1]), C[ent].ravel()], axis=1)))
                continue
            keep = filt.pairs(Rl, Cl, (R + start)[:, :, None] == C[:, None, :])
            hold(keys[keep & mine[:, :, None]])
            fe, fk, fj = np.nonzero(keep & ~mine[:, :, None])
            if len(fe):
                far.append(unique_rows(np.stack([R[fe, fk] + start, C[fe, fj]], axis=1)))
    if filt is not None and filt.square:
        hold(np.arange(n, dtype=np.int64) * width + start + np.arange(n, dtype=np.int64))
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


_NO_LABELS = np.zeros(0, dtype=np.int8)
_NO_MASK = np.zeros((0, 0), dtype=np.uint8)


def assemble_matrix(K, a):
    """Assemble form `a` into `K` (from `create_matrix`), the Dirichlet conditions it was
    made with applied (their rows' and columns' entries dropped, the diagonal the number of
    cells holding the dof, as dolfin's assemble_system). From the
    second assembly on,
    the element matrices are added straight into K's arrays (its pattern then final);
    the first goes through MatSetValuesLocal. Either way the extension packs the form's
    coefficients a block of cells at a time. A value outside K's pattern in a row owned
    here raises on every rank; in a row owned elsewhere, PETSc refuses it there. A matrix
    made with couplings skips the entries between components they do not couple, raising
    on every rank if any of them is nonzero, and on its first assembly stores its
    diagonal (zeros where the form leaves it empty). Collective."""
    from mpi4py import MPI
    from dolfinx.fem import petsc

    comm = K.comm.tompi4py()
    filt = _FILTERS.get(K.handle)
    inserter = _inserter(comm) if a.dtype == np.float64 else None
    if filt is not None and inserter is None:
        raise NotImplementedError("couplings need femtools' C++ inserter, which could not "
                                  "be built here")
    labels = (_NO_LABELS, _NO_LABELS, _NO_MASK, _NO_LABELS) if filt is None else \
        (filt.rows, filt.cols, filt.mask, filt.fixed)
    K.zeroEntries()
    missed, skipped = inserter.assemble(a._cpp_object, K.handle, *labels) \
        if inserter is not None else (-2, 0)
    if missed == -1:
        first = inserter.assemble_local(a._cpp_object, K.handle, *labels)
        if first >= 0:
            missed, skipped = 0, first
            if filt is not None and filt.square:
                from petsc4py import PETSc
                lo, hi = K.getOwnershipRange()
                i = np.arange(lo, hi, dtype=PETSc.IntType).reshape(-1, 1)
                K.setValuesRCV(i, i, np.zeros((len(i), 1)), addv=PETSc.InsertMode.ADD_VALUES)
        else:
            missed = -2
    if missed == -2:
        petsc.assemble_matrix(K, a)
        missed = 0
    total = comm.allreduce(skipped, op=MPI.SUM)
    if total:
        raise RuntimeError("assemble_matrix: {} nonzero values of the form couple components "
                           "the matrix's couplings leave out".format(total))
    if comm.allreduce(missed, op=MPI.SUM):
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
