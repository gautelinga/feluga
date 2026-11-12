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
coordinates, the Lagrange basis) and on the dof tables both backends' checkpoints write
(`femtools.dolfin_layout`, `femtools.dolfinx.checkpoint`): a cell's dofs in UFC's order.
The children must stay on their parent's rank (`redistribute=False`, as
`refine_uniformly` refines), so every matrix is built from local cells alone.
"""
import numpy as np

from femtools.hierarchical_transfer import (
    barycentric_simplex, compose, ufc_basis, ufc_nodes)


def _is_dolfinx(obj):
    try:
        import dolfinx
    except ImportError:
        return False
    return isinstance(obj, (dolfinx.mesh.Mesh, dolfinx.fem.FunctionSpace))


def _split(space):
    return space if isinstance(space, tuple) else (space, None)


def _n_nodes(d, degree):
    return d + 1 if degree == 1 else (d + 1) * (d + 2) // 2


class _Tables:
    """A space's cells as both backends' checkpoints see them: this rank's own cells,
    each with its global id (`gc`), its vertices (`x`, UFC's order) and their global ids
    (`vgid`), its dofs in UFC's order as global vector indices (`dofs`), a block of
    columns a component (`components`: (first column, count, degree)); the vector's
    owned range and size; the communicator (mpi4py)."""

    def __init__(self, space):
        V, U = _split(space)
        if _is_dolfinx(V):
            self._dolfinx(V, U)
        else:
            self._dolfin(V)
        d = self.x.shape[2]
        if self.x.shape[1] != d + 1:
            raise NotImplementedError("nested_transfer: triangles and tetrahedra only")

    def _dolfin(self, V):
        from femtools.dolfin_layout import _dof_tables
        mesh = V.mesh()
        d = mesh.topology().dim()
        if mesh.geometry().dim() != d:
            raise NotImplementedError("nested_transfer: a mesh of its own dimension")
        self.gc, dofs = _dof_tables(V)
        own = len(self.gc)
        cells = mesh.cells()[:own]
        self.x = mesh.coordinates()[cells]
        self.vgid = np.asarray(mesh.topology().global_indices(0), dtype=np.int64)[cells]
        self.dofs = np.asarray(dofs, dtype=np.int64)
        self.owned = tuple(int(v) for v in V.dofmap().ownership_range())
        self.n_global = int(V.dim())
        subs = [V.sub(i) for i in range(V.num_sub_spaces())] or [V]
        self.components, first = [], 0
        for S in subs:
            degree = int(S.ufl_element().degree())
            n = _n_nodes(d, degree)
            self.components.append((first, n, degree))
            first += n
        self.comm = mesh.mpi_comm()

    def _dolfinx(self, V, U):
        from femtools.dolfinx.checkpoint import (
            _blocked, _components, _global_tables, _lagrange_degree)
        if U is None:
            raise ValueError("nested_transfer: a DOLFINx space comes with its UniqueDofs, "
                             "as (V, U)")
        if _blocked(V):
            raise NotImplementedError("nested_transfer: scalar or mixed spaces, not blocked")
        mesh = V.mesh
        d = mesh.topology.dim
        if mesh.geometry.dim != d:
            raise NotImplementedError("nested_transfer: a mesh of its own dimension")
        self.gc, dofs = _global_tables(V, U)
        n = len(self.gc)
        # the vertices in the order the tables put the dofs in (dolfin_cell_dofs): by
        # their input global index
        G = mesh.geometry.dofmaps[0][:n]
        gv = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
        G = np.take_along_axis(G, np.argsort(gv[G], axis=1, kind="stable"), axis=1)
        self.x = mesh.geometry.x[G][:, :, :d]
        self.vgid = gv[G]
        self.dofs = np.asarray(dofs, dtype=np.int64)
        self.owned = tuple(int(v) for v in U.owned_range)
        self.n_global = int(U.n_global)
        self.components, first = [], 0
        for S in _components(V):
            degree = _lagrange_degree(S)
            k = _n_nodes(d, degree)
            self.components.append((first, k, degree))
            first += k
        self.comm = mesh.comm

    def component(self, i):
        """(dofs of component i, its degree); i None for a space of one component."""
        if i is None:
            if len(self.components) != 1:
                raise ValueError("nested_transfer: a mixed space needs its component")
            i = 0
        first, n, degree = self.components[i]
        return self.dofs[:, first:first + n], degree


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
        of each fine cell, (n_fine, n_nodes, d+1)."""
        if degree not in self._lam:
            if len(self.ancestors) != len(tf.x):
                raise ValueError("nested_transfer: {} ancestors for {} fine cells".format(
                    len(self.ancestors), len(tf.x)))
            if len(self.ancestors) and self.ancestors.max() >= len(tc.x):
                raise ValueError("nested_transfer: an ancestor beyond the coarse mesh's own "
                                 "cells (were the children redistributed?)")
            K = tc.x[self.ancestors]
            d = K.shape[2]
            lam_v = np.stack([barycentric_simplex(K, tf.x[:, j]) for j in range(d + 1)], axis=1)
            self._lam[degree] = ufc_nodes(lam_v, degree)
        return self._lam[degree]

    def _multi_index(self, lam, N):
        m = np.rint(lam * N)
        if len(m) and np.abs(lam * N - m).max() > 1e-6:
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
        m_f = self._multi_index(self._fine_nodes(tf, tc, fdeg), N)   # (nf, nnf, d+1)
        m_c = self._multi_index(ufc_nodes(np.eye(d + 1)[None], cdeg)[0], N)  # (nnc, d+1)
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
        self.n_cells = self.comm.allreduce(int(tc.gc.max()) + 1 if len(tc.gc) else 0,
                                           op=_mpi_max())
        d = tc.x.shape[2]
        # each coarse cell's vertices ordered by global id: the canonical order
        perm = np.argsort(tc.vgid, axis=1, kind="stable")[nest.ancestors]   # (nf, d+1)
        self.cell_of, self.slot, self.dof = [], [], []
        width = 0
        for i in range(len(tf.components)):
            fdofs, fdeg = tf.component(i)
            N = fdeg * 2 ** nest.passes
            m = nest._multi_index(nest._fine_nodes(tf, tc, fdeg), N)       # (nf, nnf, d+1)
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
        idx = PETSc.IS().createGeneral(self.dof.astype(PETSc.IntType), comm=PETSc.COMM_SELF)
        y = PETSc.Vec().createSeq(len(self.dof), comm=PETSc.COMM_SELF)
        scatter = PETSc.Scatter().create(x, idx, y, None)
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

    def write(self, path, name, x, mode="a"):
        from femtools.hdf5_rows import write_ordered
        write_ordered(path, name, self.gc.astype(np.int64), self.gather(x), self.n_cells,
                      self.comm, mode=mode)

    def read(self, path, name, x):
        from femtools.hdf5_rows import fetch_rows
        self.scatter(fetch_rows(path, name, self.gc.astype(np.int64), self.comm), x)


def _mpi_max():
    from mpi4py import MPI
    return MPI.MAX


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
