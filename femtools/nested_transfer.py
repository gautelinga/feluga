"""Transfers between a mesh and its uniform refinement, as sparse matrices built once, on
dolfin and DOLFINx alike: the dual mesh's (a field solved on the fine mesh, another on
the coarse one).

    fine, ancestors = refine_uniformly(mesh, passes, box, axes)
    nest = Nested(mesh, fine, ancestors, passes)
    P = nest.prolongation(Vc, Vf)               # fine = P coarse: exact (nested spaces)
    R = nest.injection(Vf, Vc, component=0)     # coarse = R fine: the fine field's values
                                                # at the coarse nodes, exact
    lattice = nest.lattice(Vf)                  # a fine field by (coarse cell, point)
    lattice.write(path, "phig", x); lattice.read(path, "phig", x)
    fine_values = nest.facet_values(coarse_values)   # facet markers on the fine mesh

The matrices are petsc4py Mats over the spaces' vectors: dolfin's global dofs, or on
DOLFINx a space's unique dofs as its `UniqueDofs` numbers them (pass `(V, U)` for a
space). Periodic spaces need nothing more: every cell's dofs are its own, a slave's being
its master's.

A uniform refinement puts every node of a fine Lagrange space of degree q, after k
passes, on the lattice of barycentric multi-indices of denominator N = q 2^k of its
coarse ancestor cell. A fine node is then identified exactly, by integers, not by a
point search: the prolongation evaluates the coarse basis at it, the injection picks the
fine node at each coarse node (a coarse node of degree p is one when p divides N), and
the lattice orders a cell's points canonically -- by the cell's vertices' global ids, as
femtools' checkpoints order them -- so a fine field is written by coarse cell and read
back on any number of ranks and on either backend, the fine mesh being made anew from
the coarse one.

Built on `femtools.hierarchical_transfer` (ancestors composed over passes, barycentric
coordinates, the Lagrange basis) and on `femtools.tables` (a cell's vertices and dofs in
UFC's order, as both backends' checkpoints write them).
The children must stay on their parent's rank (`redistribute=False`, as
`refine_uniformly` refines), so every matrix is built from local cells alone.
"""
import numpy as np

from femtools.hierarchical_transfer import (
    barycentric_simplex, compose, ufc_basis, ufc_nodes)
from femtools.tables import _is_dolfinx


def _split(space):
    return space if isinstance(space, tuple) else (space, None)


def _Tables(space):
    """A space's tables (femtools.tables.space_tables), its dofs as int64."""
    from femtools.tables import space_tables
    t = space_tables(space)
    t.dofs = np.asarray(t.dofs, dtype=np.int64)
    return t


def _owned_rows(rows, cols, vals, owned):
    """The entries of this rank's own rows, each (row, column) once."""
    keep = (rows >= owned[0]) & (rows < owned[1])
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    if len(rows):
        key = np.stack([rows, cols], axis=1)
        _, first = np.unique(key, axis=0, return_index=True)
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


def _mat(comm, rows, cols, vals, row_layout, col_layout):
    """A petsc4py AIJ matrix from this rank's own rows' entries."""
    from petsc4py import PETSc
    (rlo, rhi), n_rows = row_layout
    (clo, chi), n_cols = col_layout
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    indptr = np.zeros(rhi - rlo + 1, dtype=PETSc.IntType)
    np.add.at(indptr, rows - rlo + 1, 1)
    np.cumsum(indptr, out=indptr)
    A = PETSc.Mat().create(comm=comm)
    A.setSizes(((rhi - rlo, n_rows), (chi - clo, n_cols)))
    A.setType(PETSc.Mat.Type.AIJ)
    # local CSR of the own rows (PETSc 3.19's petsc4py has no COO assembly)
    A.setPreallocationCSR((indptr, cols.astype(PETSc.IntType),
                           vals.astype(PETSc.ScalarType)))
    A.assemble()
    return A


def refine_uniformly(mesh, passes, box, axes, lower=None):
    """`mesh` refined uniformly `passes` times by femtools' periodic Plaza (the faces
    identified along `axes` of the box with upper corner `box` bisected alike, and
    checked), the children kept on their parent's rank. Returns `(fine, ancestors)`,
    `ancestors` each fine cell's ancestor among `mesh`'s local cells. The same cells on
    either backend and any number of ranks. Collective."""
    if _is_dolfinx(mesh):
        from femtools.dolfinx.refine import refine_congruent
        tdim = mesh.topology.dim
        n_own = mesh.topology.index_map(tdim).size_local
        ancestors = np.arange(n_own, dtype=np.int64)
    else:
        import dolfin as df
        from femtools.refine_periodic import refine_congruent
        ancestors = np.arange(mesh.num_cells(), dtype=np.int64)
    fine = mesh
    for _ in range(passes):
        if _is_dolfinx(fine):
            n_own = fine.topology.index_map(fine.topology.dim).size_local
            markers = np.ones(n_own, bool)
        else:
            markers = df.MeshFunction("bool", fine, fine.topology().dim(), True)
        if axes:
            fine, _, parent = refine_congruent(fine, markers, box, axes, redistribute=False,
                                               lower=lower)
        else:
            fine, parent = _refine_plain(fine, markers)
        ancestors = compose(ancestors, np.asarray(parent, dtype=np.int64))
    return fine, ancestors


def _refine_plain(mesh, markers):
    """A refinement with no periodic faces, children on their parent's rank, and the
    parent map: periodic Plaza with no axes is plain Plaza."""
    if _is_dolfinx(mesh):
        from femtools.dolfinx.refine import refine_periodic
        from femtools.dolfinx.periodic import box_extent
        lo, hi = box_extent(mesh)
        fine, _, parent = refine_periodic(mesh, markers, hi, [], redistribute=False, lower=lo)
    else:
        from femtools.periodic import extent
        from femtools.plaza_periodic import refine_periodic
        lo, hi = extent(mesh)
        fine, _, parent = refine_periodic(mesh, markers, hi, [], redistribute=False, lower=lo)
    return fine, parent


class Nested:
    """A coarse mesh and its uniform refinement by `passes` passes (`refine_uniformly`),
    `ancestors` mapping each of the fine mesh's local cells to its ancestor among the
    coarse mesh's local cells."""

    def __init__(self, coarse, fine, ancestors, passes):
        self.coarse, self.fine, self.passes = coarse, fine, int(passes)
        self.ancestors = np.asarray(ancestors, dtype=np.int64)
        self._lam = {}

    def _fine_nodes(self, tf, tc, degree):
        """The barycentric coordinates in their ancestor of the nodes of degree `degree`
        of each fine cell, (n_fine, n_nodes, d+1). Collective (its checks are agreed)."""
        from femtools.hdf5_rows import agreed
        if degree not in self._lam:
            if agreed(tf.comm, len(self.ancestors) != len(tf.x)):
                raise ValueError("nested_transfer: as many ancestors as fine cells, on "
                                 "every rank")
            if agreed(tf.comm, len(self.ancestors) and self.ancestors.max() >= len(tc.x)):
                raise ValueError("nested_transfer: an ancestor beyond the coarse mesh's own "
                                 "cells (were the children redistributed?)")
            K = tc.x[self.ancestors]
            d = K.shape[2]
            lam_v = np.stack([barycentric_simplex(K, tf.x[:, j]) for j in range(d + 1)], axis=1)
            self._lam[degree] = ufc_nodes(lam_v, degree)
        return self._lam[degree]

    def _multi_index(self, lam, N, comm):
        """The lattice multi-indices of barycentric coordinates `lam`, denominator N.
        Collective on comm (its check is agreed)."""
        from femtools.hdf5_rows import agreed
        m = np.rint(lam * N)
        if agreed(comm, len(m) and np.abs(lam * N - m).max() > 1e-6):
            raise RuntimeError("nested_transfer: a fine node off its ancestor's lattice "
                               "(is the refinement uniform, and passes right?)")
        return m.astype(np.int64)

    def prolongation(self, coarse_space, fine_space, coarse_component=None,
                     fine_component=None):
        """The matrix P, fine vector = P coarse vector: each fine node's value from the
        coarse field in its ancestor cell, exact for a coarse field the fine space holds
        (fine degree at least the coarse one)."""
        tc, tf = _Tables(coarse_space), _Tables(fine_space)
        cdofs, cdeg = tc.component(coarse_component)
        fdofs, fdeg = tf.component(fine_component)
        lam = self._fine_nodes(tf, tc, fdeg)                     # (nf, nnf, d+1)
        B = ufc_basis(lam, cdeg)                                 # (nf, nnf, nnc)
        rows = np.broadcast_to(fdofs[:, :, None], B.shape)
        cols = np.broadcast_to(cdofs[self.ancestors][:, None, :], B.shape)
        keep = np.abs(B) > 1e-12
        r, c, v = _owned_rows(rows[keep], cols[keep], B[keep], tf.owned)
        _check_rows(tf.comm, r, tf.owned, "fine dofs")
        return _mat(tf.comm, r, c, v, (tf.owned, tf.n_global), (tc.owned, tc.n_global))

    def injection(self, fine_space, coarse_space, fine_component=None,
                  coarse_component=None):
        """The matrix R, coarse vector = R fine vector: the fine field at each coarse
        node, a node of the fine space too (the coarse degree must divide the fine
        degree times 2^passes)."""
        tc, tf = _Tables(coarse_space), _Tables(fine_space)
        cdofs, cdeg = tc.component(coarse_component)
        fdofs, fdeg = tf.component(fine_component)
        N = fdeg * 2 ** self.passes
        if N % cdeg:
            raise ValueError("nested_transfer: degree {} nodes are not among the fine "
                             "degree {} nodes after {} passes".format(cdeg, fdeg, self.passes))
        d = tc.x.shape[2]
        m_f = self._multi_index(self._fine_nodes(tf, tc, fdeg), N, tc.comm)   # (nf, nnf, d+1)
        m_c = self._multi_index(ufc_nodes(np.eye(d + 1)[None], cdeg)[0], N, None)  # (nnc, d+1)
        weights = (N + 1) ** np.arange(d + 1)
        key_f, key_c = m_f @ weights, m_c @ weights
        rows, cols = [], []
        for j, k in enumerate(key_c):
            cell, node = np.nonzero(key_f == k)
            rows.append(cdofs[self.ancestors[cell], j])
            cols.append(fdofs[cell, node])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        r, c, v = _owned_rows(rows, cols, np.ones(len(rows)), tc.owned)
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
        from femtools.tables import facet_tables
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
    either backend, for the same coarse mesh (femtools' checkpoints number it alike)."""

    def __init__(self, nest, fine_space):
        tf = _Tables(fine_space)
        tc = _Tables(_coarse_space_like(nest.coarse, fine_space))
        self.comm, self.owned, self.gc = tf.comm, tf.owned, tc.gc
        self._scatter = None
        self.passes = nest.passes
        self.n_cells = tc.n_cells
        d = tc.x.shape[2]
        # each coarse cell's vertices ordered by global id: the canonical order
        perm = np.argsort(tc.vgid, axis=1, kind="stable")[nest.ancestors]   # (nf, d+1)
        self.cell_of, self.slot, self.dof = [], [], []
        width = 0
        for i in range(len(tf.components)):
            fdofs, fdeg = tf.component(i)
            N = fdeg * 2 ** nest.passes
            m = nest._multi_index(nest._fine_nodes(tf, tc, fdeg), N, tf.comm)       # (nf, nnf, d+1)
            m = np.take_along_axis(m, perm[:, None, :], axis=2)
            pts = lattice_points(d, N)
            weights = (N + 1) ** np.arange(d + 1)[::-1]
            order = np.argsort(pts @ weights)
            j = order[np.searchsorted((pts @ weights)[order], m @ weights)]
            cell = np.broadcast_to(nest.ancestors[:, None], j.shape)
            self.cell_of.append(cell.ravel())
            self.slot.append((width + j).ravel())
            self.dof.append(fdofs.ravel())
            width += len(pts)
        self.cell_of = np.concatenate(self.cell_of)
        self.slot = np.concatenate(self.slot)
        self.dof = np.concatenate(self.dof)
        self.width = width
        # every lattice point of every own coarse cell is some fine node
        filled = np.zeros((len(tc.gc), width), bool)
        filled[self.cell_of, self.slot] = True
        if self.comm.allreduce(int((~filled).sum())):
            raise RuntimeError("nested_transfer: lattice points without a fine node")

    def gather(self, x):
        """This rank's own coarse cells' rows from the fine vector `x` (a petsc4py Vec),
        (n_cells, width)."""
        from petsc4py import PETSc
        if self._scatter is None:           # built once: any vector of the fine layout
            idx = PETSc.IS().createGeneral(self.dof.astype(PETSc.IntType),
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
        from femtools.hdf5_rows import write_ordered
        attrs = dict(attrs or {}, passes=self.passes, width=self.width)
        write_ordered(path, name, self.gc.astype(np.int64), self.gather(x), self.n_cells,
                      self.comm, mode=mode, attrs=attrs)

    def read(self, path, name, x):
        """The fine vector `x`'s own entries from dataset `name` of `path`, which must
        have been written for as many passes and as wide rows. Collective."""
        from femtools.hdf5_rows import fetch_rows
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
    if _is_dolfinx(V):
        from dolfinx import fem
        from femtools.dolfinx.unique import UniqueDofs
        P1 = fem.functionspace(coarse, ("Lagrange", 1))
        return P1, UniqueDofs(P1)
    import dolfin as df
    return df.FunctionSpace(coarse, "CG", 1)
