"""Transfers between a mesh and its uniform refinement, as sparse matrices built once, on
dolfin and DOLFINx alike: the dual mesh's (a field solved on the fine mesh, another on
the coarse one).

    fine, ancestors = refine_uniformly(mesh, passes, upper, axes)
    nest = Nested(mesh, fine, ancestors, passes)
    P = nest.prolongation(Vc, Vf)               # fine = P coarse: exact (nested spaces)
    R = nest.injection(Vf, Vc, fine_component=0)   # coarse = R fine: the fine field's
                                                   # values at the coarse nodes, exact
    lattice = nest.lattice(Vf)                  # a fine field by (coarse cell, point)
    lattice.write(path, "phig", x); lattice.read(path, "phig", x)
    fine_values = nest.facet_values(coarse_values)   # facet markers on the fine mesh

The matrices are petsc4py Mats over the spaces' vectors: dolfin's global dofs, or on
DOLFINx a space's unique dofs as its `UniqueDofs` numbers them (a
feluga.backend.dolfinx space knows them; pass `(V, U)` for a plain DOLFINx one).
Periodic spaces need nothing more: every cell's dofs are its own, a slave's being its
master's.

A uniform refinement puts every node of a fine Lagrange space of degree q, after k
passes, on the lattice of barycentric multi-indices of denominator N = q 2^k of its
coarse ancestor cell. A fine node is then identified exactly, by integers, not by a
point search: the prolongation evaluates the coarse basis at it, the injection picks the
fine node at each coarse node (a coarse node of degree p is one when p divides N), and
the lattice orders a cell's points canonically -- by the cell's vertices' global ids, as
feluga' checkpoints order them -- so a fine field is written by coarse cell and read
back on any number of ranks and on either backend, the fine mesh being made anew from
the coarse one.

Built on `feluga.ufc` (barycentric coordinates, the Lagrange nodes and basis) and on
`feluga.tables` (a cell's vertices and dofs in UFC's order, as both backends'
checkpoints write them). The children must stay on their parent's rank (`redistribute=False`, as
`refine_uniformly` refines), so every matrix is built from local cells alone.

Memory: a fine node lies in every fine cell around it, so
nothing is evaluated per (fine cell, node). The prolongation evaluates the coarse basis
once per owned fine dof, in the first local fine cell holding it; the injection and the
lattice look at the fine cells a block at a time (`hierarchical_transfer.EVAL_CHUNK`)
and keep only what they found, one entry per coarse dof or lattice point.
"""
import numpy as np

from feluga import hierarchical_transfer
from feluga.hierarchical_transfer import compose
from feluga.backend import is_dolfinx
from feluga.ufc import barycentric_simplex, ufc_basis, ufc_nodes


def _split(space):
    return space if isinstance(space, tuple) else (space, None)


def _tables(space):
    """A space's tables (feluga.tables.space_tables), its dofs as int64."""
    from feluga.tables import space_tables
    t = space_tables(space)
    t.dofs = np.asarray(t.dofs, dtype=np.int64)
    return t


def _owned_rows(rows, cols, vals, owned):
    """The entries of this rank's own rows, each (row, column) once (its first), ordered
    by row and column."""
    lo, hi = owned
    keep = (rows >= lo) & (rows < hi)
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    if len(rows):
        width = int(cols.max()) + 1
        if (hi - lo) * width < 2 ** 63:     # (row, column) packed into one int64
            _, first = np.unique((rows - lo) * width + cols, return_index=True)
        else:
            _, first = np.unique(np.stack([rows, cols], axis=1), axis=0, return_index=True)
        rows, cols, vals = rows[first], cols[first], vals[first]
    return rows, cols, vals


def _check_rows(comm, rows, owned, what):
    lo, hi = owned
    covered = np.zeros(hi - lo, bool)
    covered[rows - lo] = True
    missing = comm.allreduce(int((~covered).sum()))
    if missing:
        raise RuntimeError("nested_transfer: {} of the {} have no entry from this "
                           "rank's cells".format(missing, what))


def _petsc_ints(comm, a, n_global, what):
    """`a` (global indices below `n_global`) as PETSc's integers; raises on every rank
    when `n_global` does not fit them, where a cast would wrap without a word.
    Collective."""
    from petsc4py import PETSc
    from feluga.hdf5_rows import agreed
    top = np.iinfo(PETSc.IntType).max
    if agreed(comm, n_global - 1 > top or (len(a) and int(np.max(a)) > top)):
        raise OverflowError(
            "nested_transfer: {} {} beyond PETSc's {}-bit integers; this needs a PETSc "
            "built with 64-bit indices (--with-64-bit-indices)".format(
                n_global, what, 8 * np.dtype(PETSc.IntType).itemsize))
    return np.asarray(a).astype(PETSc.IntType)


def _mat(comm, rows, cols, vals, row_layout, col_layout):
    """A petsc4py AIJ matrix from this rank's own rows' entries, ordered by row and
    column (as `_owned_rows` gives them). Collective."""
    from petsc4py import PETSc
    (rlo, rhi), n_rows = row_layout
    (clo, chi), n_cols = col_layout
    _petsc_ints(comm, (), n_rows, "rows")
    cols = _petsc_ints(comm, cols, n_cols, "columns")
    indptr = np.zeros(rhi - rlo + 1, dtype=PETSc.IntType)
    np.cumsum(np.bincount(rows - rlo, minlength=rhi - rlo), out=indptr[1:])
    A = PETSc.Mat().create(comm=comm)
    A.setSizes(((rhi - rlo, n_rows), (chi - clo, n_cols)))
    A.setType(PETSc.Mat.Type.AIJ)
    # local CSR of the own rows (PETSc 3.19's petsc4py has no COO assembly)
    A.setPreallocationCSR((indptr, cols, vals.astype(PETSc.ScalarType)))
    A.assemble()
    return A


def refine_uniformly(mesh, passes, upper, axes, lower=None):
    """`mesh` refined uniformly `passes` times by feluga' periodic Plaza (the faces
    identified along `axes` of the box from `lower` to `upper` bisected alike, and
    checked), the children kept on their parent's rank. Returns `(fine, ancestors)`,
    `ancestors` each fine cell's ancestor among `mesh`'s local cells. The same cells on
    either backend and any number of ranks. Collective."""
    if is_dolfinx(mesh):
        from feluga.dolfinx.periodic import box_extent
        from feluga.dolfinx.refine import refine_congruent, refine_periodic

        def all_cells(m):
            return np.ones(m.topology.index_map(m.topology.dim).size_local, bool)
        n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    else:
        import dolfin as df
        from feluga.periodic import extent as box_extent
        from feluga.plaza_periodic import refine_periodic
        from feluga.refine_periodic import refine_congruent

        def all_cells(m):
            return df.MeshFunction("bool", m, m.topology().dim(), True)
        n_cells = mesh.num_cells()
    fine = mesh
    ancestors = np.arange(n_cells, dtype=np.int64)
    for _ in range(passes):
        if axes:
            fine, _, parent = refine_congruent(fine, all_cells(fine), upper, axes,
                                               redistribute=False, lower=lower)
        else:   # periodic Plaza with no axes is plain Plaza
            lo, hi = box_extent(fine)
            fine, _, parent = refine_periodic(fine, all_cells(fine), hi, [],
                                              redistribute=False, lower=lo)
        ancestors = compose(ancestors, np.asarray(parent, dtype=np.int64))
    return fine, ancestors


def _first_holder(dofs, lo, hi):
    """For each dof in [lo, hi), the first (fine cell, node) of `dofs` (n_cells, n_nodes)
    holding it, in the order of the cells and then of their nodes, as the flat index
    cell * n_nodes + node; -1 where no cell holds it. A block of cells at a time, the
    last block first, so that the first holder is assigned last."""
    n_cells, n_nodes = dofs.shape
    first = np.full(hi - lo, -1, dtype=np.int64)
    chunk = max(1, hierarchical_transfer.EVAL_CHUNK)
    for start in range(((n_cells - 1) // chunk) * chunk if n_cells else 0, -1, -chunk):
        block = np.ravel(dofs[start:start + chunk])
        at = np.flatnonzero((block >= lo) & (block < hi))[::-1]
        # numpy assigns repeated indices in order: the last, the first holder, stays
        first[block[at] - lo] = start * n_nodes + at
    return first


class Nested:
    """A coarse mesh and its uniform refinement by `passes` passes (`refine_uniformly`),
    `ancestors` mapping each of the fine mesh's local cells to its ancestor among the
    coarse mesh's local cells."""

    def __init__(self, coarse, fine, ancestors, passes):
        self.coarse, self.fine, self.passes = coarse, fine, int(passes)
        self.ancestors = np.asarray(ancestors, dtype=np.int64)

    def _check_ancestors(self, tf, tc):
        """Raise on every rank unless every fine cell has an ancestor among the coarse
        mesh's own cells. Collective."""
        from feluga.hdf5_rows import agreed
        if agreed(tf.comm, len(self.ancestors) != len(tf.x)):
            raise ValueError("nested_transfer: as many ancestors as fine cells, on "
                             "every rank")
        if agreed(tf.comm, len(self.ancestors) and self.ancestors.max() >= len(tc.x)):
            raise ValueError("nested_transfer: an ancestor beyond the coarse mesh's own "
                             "cells (were the children redistributed?)")

    def _fine_nodes(self, tf, tc, cells, degree):
        """The barycentric coordinates in their ancestor of the nodes of degree `degree`
        of the fine cells `cells`, (len(cells), n_nodes, d+1)."""
        K = tc.x[self.ancestors[cells]]
        x = tf.x[cells]
        d = K.shape[2]
        lam_v = np.stack([barycentric_simplex(K, x[:, j]) for j in range(d + 1)], axis=1)
        return ufc_nodes(lam_v, degree)

    @staticmethod
    def _multi_index(lam, N):
        """`(m, off)`: the lattice multi-indices of barycentric coordinates `lam`,
        denominator N, and how far `lam * N` is from them at most (0 for none)."""
        m = np.rint(lam * N)
        off = float(np.abs(lam * N - m).max()) if m.size else 0.0
        return m.astype(np.int64), off

    @staticmethod
    def _check_lattice(comm, off):
        """Raise on every rank when some rank's fine nodes were `off` their ancestor's
        lattice. Collective."""
        from feluga.hdf5_rows import agreed
        if agreed(comm, off > 1e-6):
            raise RuntimeError("nested_transfer: a fine node off its ancestor's lattice "
                               "(is the refinement uniform, and passes right?)")

    def _blocks(self, n_cells):
        """The fine cells in blocks of `hierarchical_transfer.EVAL_CHUNK`."""
        chunk = max(1, hierarchical_transfer.EVAL_CHUNK)
        return (np.arange(a, min(a + chunk, n_cells)) for a in range(0, n_cells, chunk))

    def prolongation(self, coarse_space, fine_space, coarse_component=None,
                     fine_component=None):
        """The matrix P, fine vector = P coarse vector: each fine node's value from the
        coarse field in its ancestor cell, exact for a coarse field the fine space holds
        (fine degree at least the coarse one). Each owned fine dof's row is the coarse
        basis at its node in the first local fine cell holding it. Collective."""
        tc, tf = _tables(coarse_space), _tables(fine_space)
        cdofs, cdeg = tc.component(coarse_component)
        fdofs, fdeg = tf.component(fine_component)
        self._check_ancestors(tf, tc)
        lo, hi = tf.owned
        first = _first_holder(fdofs, lo, hi)
        held = np.flatnonzero(first >= 0)
        cell, node = np.divmod(first[held], fdofs.shape[1])
        first = None

        def block(sl):
            lam = self._fine_nodes(tf, tc, cell[sl], fdeg)
            return ufc_basis(lam[np.arange(len(lam)), node[sl]], cdeg)    # (b, nnc)
        B = hierarchical_transfer._in_blocks(len(held), block)
        rows = np.broadcast_to((lo + held)[:, None], B.shape)
        cols = cdofs[self.ancestors[cell]]
        keep = np.abs(B) > 1e-12
        r, c, v = _owned_rows(rows[keep], cols[keep], B[keep], tf.owned)
        _check_rows(tf.comm, r, tf.owned, "fine dofs")
        return _mat(tf.comm, r, c, v, (tf.owned, tf.n_global), (tc.owned, tc.n_global))

    def injection(self, fine_space, coarse_space, fine_component=None,
                  coarse_component=None):
        """The matrix R, coarse vector = R fine vector: the fine field at each coarse
        node, a node of the fine space too (the coarse degree must divide the fine
        degree times 2^passes). Collective."""
        tc, tf = _tables(coarse_space), _tables(fine_space)
        cdofs, cdeg = tc.component(coarse_component)
        fdofs, fdeg = tf.component(fine_component)
        N = fdeg * 2 ** self.passes
        if N % cdeg:
            raise ValueError("nested_transfer: degree {} nodes are not among the fine "
                             "degree {} nodes after {} passes".format(cdeg, fdeg, self.passes))
        self._check_ancestors(tf, tc)
        d = tc.x.shape[2]
        m_c, _ = self._multi_index(ufc_nodes(np.eye(d + 1)[None], cdeg)[0], N)  # (nnc, d+1)
        weights = (N + 1) ** np.arange(d + 1)
        key_c = m_c @ weights
        lo, hi = tc.owned
        found, off = [], 0.0
        for cells in self._blocks(len(fdofs)):
            m_f, o = self._multi_index(self._fine_nodes(tf, tc, cells, fdeg), N)
            off = max(off, o)
            key_f = m_f @ weights                                   # (b, nnf)
            rows, cols = [], []
            for j, k in enumerate(key_c):
                c, node = np.nonzero(key_f == k)
                rows.append(cdofs[self.ancestors[cells[c]], j])
                cols.append(fdofs[cells[c], node])
            rows, cols = np.concatenate(rows), np.concatenate(cols)
            own = (rows >= lo) & (rows < hi)
            r, c, _ = _owned_rows(rows[own], cols[own], np.ones(int(own.sum())), tc.owned)
            found.append(np.stack([r, c], axis=1))
        self._check_lattice(tc.comm, off)
        found = np.concatenate(found) if found else np.zeros((0, 2), np.int64)
        r, c, v = _owned_rows(found[:, 0], found[:, 1], np.ones(len(found)), tc.owned)
        _check_rows(tc.comm, r, tc.owned, "coarse dofs")
        return _mat(tc.comm, r, c, v, (tc.owned, tc.n_global), (tf.owned, tf.n_global))

    def lattice(self, fine_space):
        """A fine space's dofs by (coarse cell, lattice point), every component of it:
        `Lattice`."""
        return Lattice(self, fine_space)

    def facet_values(self, coarse_values, default=0):
        """Facet markers carried to the fine mesh: `coarse_values` per local facet of the
        coarse mesh (owned and ghost, its local numbering) -> the same per local facet of
        the fine mesh: a fine facet of an owned fine cell that lies in a facet of the
        ancestor cell takes that facet's value, the others `default`. Exact: a fine
        facet's vertices have barycentric coordinate 0 for the ancestor's vertex opposite
        the facet it lies in."""
        from feluga.tables import facet_tables
        c, f = facet_tables(self.coarse), facet_tables(self.fine)
        coarse_values = np.asarray(coarse_values)
        d = c.x.shape[1]
        anc = self.ancestors
        K = c.x[c.cell_vertices[anc]]                          # (nf, d+1, d)
        # the coarse facet opposite each vertex of each coarse cell
        opposite = np.empty(c.cell_facets.shape, dtype=np.int64)
        for j in range(d + 1):
            for k in range(d + 1):
                facet = c.cell_facets[:, k]
                holds = (c.facet_vertices[facet] == c.cell_vertices[:, j][:, None]).any(axis=1)
                opposite[~holds, j] = facet[~holds]
        values = np.full(len(f.facet_vertices), default, dtype=coarse_values.dtype)
        for k in range(d + 1):
            facet = f.cell_facets[:, k]                         # (nf,) fine facets
            lam = np.stack([barycentric_simplex(K, f.x[f.facet_vertices[facet][:, i]])
                            for i in range(d)], axis=1)         # (nf, d, d+1)
            on = np.abs(lam).max(axis=1) < 1e-8                 # (nf, d+1): in facet j
            cell, j = np.nonzero(on)
            values[facet[cell]] = coarse_values[opposite[anc[cell], j]]
        return values


def lattice_points(d, N):
    """The multi-indices of the lattice of denominator N in a d-simplex, (n, d+1),
    lexicographically: the canonical order of a cell's points."""
    from itertools import product
    pts = [m + (N - sum(m),) for m in product(range(N + 1), repeat=d) if sum(m) <= N]
    return np.array(sorted(pts), dtype=np.int64)


class Lattice:
    """A fine field's values by coarse cell: row `gc` (the coarse cell's global id) of a
    dataset holds the field at its lattice points, component after component, each
    component's points lexicographic in the barycentric coordinates of the cell's
    vertices ordered by their global ids. The same rows on any number of ranks and on
    either backend, for the same coarse mesh (feluga' checkpoints number it alike)."""

    def __init__(self, nest, fine_space):
        tf = _tables(fine_space)
        tc = _tables(_coarse_space_like(nest.coarse, fine_space))
        nest._check_ancestors(tf, tc)
        self.comm, self.owned, self.gc = tf.comm, tf.owned, tc.gc
        self._scatter = None
        self.passes = nest.passes
        self.n_cells = tc.n_cells
        self.n_dofs = tf.n_global
        d = tc.x.shape[2]
        n_own = len(tc.gc)
        # each coarse cell's vertices ordered by global id: the canonical order
        perm = np.argsort(tc.vgid, axis=1, kind="stable")              # (nc, d+1)
        parts, off = [], 0.0
        for i in range(len(tf.components)):
            fdofs, fdeg = tf.component(i)
            N = fdeg * 2 ** nest.passes
            pts = lattice_points(d, N)
            weights = (N + 1) ** np.arange(d + 1)[::-1]
            order = np.argsort(pts @ weights)
            sorted_keys = (pts @ weights)[order]
            # the fine dof at each (own coarse cell, lattice point), -1 until found
            dof_at = np.full((n_own, len(pts)), -1, dtype=np.int64)
            for cells in nest._blocks(len(fdofs)):
                m, o = nest._multi_index(nest._fine_nodes(tf, tc, cells, fdeg), N)
                off = max(off, o)
                anc = nest.ancestors[cells]
                m = np.take_along_axis(m, perm[anc][:, None, :], axis=2)   # (b, nnf, d+1)
                at = np.minimum(np.searchsorted(sorted_keys, m @ weights), len(pts) - 1)
                dof_at[anc[:, None], order[at]] = fdofs[cells]
            parts.append(dof_at)
        nest._check_lattice(self.comm, off)
        dof_at = np.concatenate(parts, axis=1)
        parts = None
        self.width = dof_at.shape[1]
        # every lattice point of every own coarse cell is some fine node
        if self.comm.allreduce(int((dof_at < 0).sum())):
            raise RuntimeError("nested_transfer: lattice points without a fine node")
        self.cell_of, self.slot = np.nonzero(dof_at >= 0)
        self.dof = dof_at[self.cell_of, self.slot]

    def gather(self, x):
        """This rank's own coarse cells' rows from the fine vector `x` (a petsc4py Vec),
        (n_cells, width)."""
        from petsc4py import PETSc
        if self._scatter is None:           # built once: any vector of the fine layout
            idx = PETSc.IS().createGeneral(
                _petsc_ints(self.comm, self.dof, self.n_dofs, "fine dofs"),
                comm=PETSc.COMM_SELF)
            y = PETSc.Vec().createSeq(len(self.dof), comm=PETSc.COMM_SELF)
            self._scatter = (PETSc.Scatter().create(x, idx, y, None), y)
        scatter, y = self._scatter
        scatter.scatter(x, y, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)
        rows = np.empty((len(self.gc), self.width))
        rows[self.cell_of, self.slot] = y.getArray()
        return rows

    def scatter(self, rows, x):
        """The fine vector `x`'s own entries from this rank's own coarse cells' rows."""
        lo, hi = self.owned
        own = (self.dof >= lo) & (self.dof < hi)
        a = x.getArray()
        a[self.dof[own] - lo] = rows[self.cell_of[own], self.slot[own]]
        x.assemble()

    def write(self, path, name, x, mode="a", attrs=None):
        """Dataset `name` of `path`: the fine vector `x`, a row a coarse cell (`attrs`
        on it; `passes` and `width` always). Collective."""
        from feluga.hdf5_rows import write_ordered
        attrs = dict(attrs or {}, passes=self.passes, width=self.width)
        write_ordered(path, name, self.gc.astype(np.int64), self.gather(x), self.n_cells,
                      self.comm, mode=mode, attrs=attrs)

    def read(self, path, name, x):
        """The fine vector `x`'s own entries from dataset `name` of `path`, which must
        have been written for as many passes and as wide rows. Collective."""
        from feluga.hdf5_rows import fetch_rows
        found = None
        if self.comm.rank == 0:
            import h5py
            with h5py.File(path, "r") as h5:
                a = h5[name].attrs
                found = (int(a.get("passes", -1)), int(a.get("width", -1)))
        found = self.comm.bcast(found, root=0)
        if found != (self.passes, self.width):
            raise ValueError("{}:{}: a fine field of {} passes and rows of {}, not {} and {}"
                             .format(path, name, found[0], found[1], self.passes, self.width))
        self.scatter(fetch_rows(path, name, self.gc.astype(np.int64), self.comm), x)


def _coarse_space_like(coarse, fine_space):
    """A P1 space on the coarse mesh, for its cells' tables (no dofs are used)."""
    V, U = _split(fine_space)
    if is_dolfinx(V):
        from dolfinx import fem
        from feluga.dolfinx.unique import UniqueDofs
        P1 = fem.functionspace(coarse, ("Lagrange", 1))
        return P1, UniqueDofs(P1)
    import dolfin as df
    return df.FunctionSpace(coarse, "CG", 1)
