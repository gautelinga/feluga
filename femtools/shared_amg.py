"""One BoomerAMG for the velocity components of a block-diagonal Stokes preconditioner.

    sets = component_index_sets(W, comm)            # either backend
    check_blocks(A, sets, comm)                       # the blocks are one Laplacian
    pc.setType("python")
    pc.setPythonContext(SharedVelocityPC(A, sets, p_dofs, diag_M, comm, amg_options))

The velocity block of a Stokes operator stored by component (femtools.couplings) is
block-diagonal by component, and with the same conditions on every component the blocks
are the same scalar Laplacian -- once each component's dofs are listed in the same node
order. A fieldsplit copies each block out of `A` and hands each to its own BoomerAMG, which
copies it again into hypre's format: the scalar Laplacian six times over, besides `A`, and
three hierarchies. Here the first block is extracted once, converted to hypre's format (the
AIJ copy then let go) and set up once, and the preconditioner applies that one AMG to each
component and divides the pressure by diag(M), what the fieldsplit applies.

The node order comes from the cell dofmaps -- per cell, the same local node in each
component -- and not from the sorted component dofs, which are not in node order, nor from
`collapse()`, which of a periodic mixed space is not periodic. The AMG's vectors are laid
out as the first component's owned dofs, and the others are scattered to them: across ranks
where a node's components are owned apart (legacy dolfin owns a mixed space's dofs dof by
dof; DOLFINx 0.11 gives every dof the owner of its mesh entity, so there they never are). `check_blocks` confirms the blocks are the one the AMG
was built on, by one product with `A`. From stokes-beadpack (2026-10-25), where it is the
Stokes solve's velocity preconditioner.
"""
import numpy as np
from petsc4py import PETSc

PREFIX = "velocity_"
VELOCITY = ((0, 0), (0, 1), (0, 2))      # [P2^3, P1]'s velocity, as paths


def velocity_paths(W):
    """The paths of W's scalar components but the last: a Stokes space's velocity, whatever
    its layout -- ((0, 0), (0, 1), (0, 2)) for [P2^3, P1], ((0,), (1,), (2,)) for
    [P2, P2, P2, P1] (`femtools.couplings.leaves`)."""
    from femtools.couplings import leaves
    paths = leaves(W)
    if len(paths) < 2:
        raise ValueError("velocity_paths: W has {} scalar component(s); a Stokes space has "
                         "the velocity's and the pressure's".format(len(paths)))
    return tuple(paths[:-1])


def component_index_sets(W, comm, components=None):
    """The dofs of each of `components` (sub-space paths of `W`; by default every scalar
    component but the last, the velocity of a Stokes space of either layout:
    `velocity_paths`) as global index arrays of A's numbering in one node order: entry i of
    each is the same node, the nodes being those whose first component this rank owns.
    Legacy dolfin (A's numbering is W's) or femtools.backend.dolfinx (the unique dofs').
    Raises if two cells pair a dof differently."""
    if components is None:
        components = velocity_paths(W)
    from femtools.backend import is_dolfinx
    if is_dolfinx(W):
        return _dolfinx_sets(W, comm, components)
    return _dolfin_sets(W, comm, components)


def _sub(W, path, legacy):
    for i in path:
        W = W.sub(i) if legacy else W.sub([i])
    return W


def _paired(first, other, c, comm):
    """`other`'s entry at each distinct value of `first` (aligned), refusing two (on every
    rank together)."""
    from femtools.hdf5_rows import agreed
    keys, inv = np.unique(first, return_inverse=True)
    pair = np.full(len(keys), -1, dtype=np.int64)
    pair[inv] = other
    if agreed(comm, not np.array_equal(pair[inv], other)):
        raise RuntimeError("component {} is paired with the first differently by two "
                           "cells".format(c))
    return keys, pair


def _dolfin_sets(W, comm, components):
    from femtools.periodic import cell_dofs
    lo, _ = W.dofmap().ownership_range()
    to_global = np.asarray(W.dofmap().tabulate_local_to_global_dofs(), dtype=np.int64)
    from femtools.hdf5_rows import agreed
    cells = np.arange(W.mesh().num_cells(), dtype=np.int64)
    maps = [_sub(W, p, True).dofmap() for p in components]
    nodes = _sub(W, components[0], True).element().space_dimension()   # a cell's, every rank
    rows = [cell_dofs(m, cells, nodes, comm).ravel() for m in maps]
    first = np.asarray(maps[0].dofs(), dtype=np.int64) - lo
    sets = [first + lo]
    for c in range(1, len(components)):
        pair = np.full(len(to_global), -1, dtype=np.int64)
        pair[rows[0]] = rows[c]
        if agreed(comm, not np.array_equal(pair[rows[0]], rows[c])):   # every rank, or none
            raise RuntimeError('component {} is paired with the first differently by two '
                               'cells'.format(c))
        mine = pair[first]
        if agreed(comm, np.any(mine < 0)):
            raise RuntimeError('an owned first component on rank {} is in no cell of its '
                               'own'.format(comm.rank))
        sets.append(to_global[mine])
    return sets


def _dolfinx_sets(W, comm, components):
    U = W.unique_dofs()
    cpp = W._cpp_object
    maps = [np.asarray(_sub(cpp, p, False).dofmap.map()) for p in components]
    first = U.compact[maps[0]].ravel()
    lo, hi = U.owned_range
    keys = None
    sets = []
    for c in range(len(components)):
        k, pair = _paired(first, U.compact[maps[c]].ravel(), c, comm)
        keys = k if keys is None else keys
        sets.append(pair[(k >= lo) & (k < hi)])
    # every unique dof of the first component this rank owns is in a cell here
    first_local = np.zeros(len(U.compact), dtype=bool)
    first_local[maps[0].ravel()] = True
    own = np.sort(U.compact[U.owned_unique[first_local[U.owned_unique]]])
    from femtools.hdf5_rows import agreed
    if agreed(comm, not np.array_equal(own, sets[0])):   # every rank raises, or none
        raise RuntimeError('an owned first component on rank {} is in no cell of its '
                           'own'.format(comm.rank))
    return sets


def check_blocks(A, sets, comm, tol=1e-12):
    """Raises unless A's block on each component, in the shared node order, acts as the
    first one does: one random vector through `A` per component."""
    first = PETSc.IS().createGeneral(np.asarray(sets[0], dtype=PETSc.IntType), comm=comm)
    K0 = A.createSubMatrix(first, first)
    x0, y0 = K0.createVecs()
    yc = y0.duplicate()
    w, Aw = A.createVecs()
    try:                                  # a copy of a whole block: destroyed if it raises
        x0.setRandom()
        K0.mult(x0, y0)
        for c in range(1, len(sets)):
            is_c = PETSc.IS().createGeneral(np.asarray(sets[c], dtype=PETSc.IntType),
                                            comm=comm)
            scatter = PETSc.Scatter().create(w, is_c, x0, None)
            try:
                w.set(0.0)
                scatter.scatter(x0, w, PETSc.InsertMode.INSERT, PETSc.ScatterMode.REVERSE)
                A.mult(w, Aw)
                scatter.scatter(Aw, yc, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                yc.axpy(-1.0, y0)
                if yc.norm() > tol * y0.norm():       # a global norm: every rank raises
                    raise RuntimeError('velocity component {} has a block unlike the first: '
                                       '|dK x| / |K x| = {:.3e}'.format(
                                           c, yc.norm() / y0.norm()))
            finally:
                scatter.destroy()
                is_c.destroy()
    finally:
        for o in (x0, y0, yc, w, Aw, K0, first):
            o.destroy()


class SharedVelocityPC:
    """The python context of a PCSHELL: one AMG on each velocity component in turn, and the
    pressure divided by `diag_M` (its rows `p_dofs`, global indices of A). `amg_options`
    (without the prefix) are set for the AMG's setup alone, PETSC_OPTIONS winning
    (`femtools.ksp.defaults`)."""

    def __init__(self, A, sets, p_dofs, diag_M, comm, amg_options, prefix=PREFIX):
        self.comm = comm
        self.sets = [PETSc.IS().createGeneral(np.asarray(s, dtype=PETSc.IntType), comm=comm)
                     for s in sets]
        self.p_is = PETSc.IS().createGeneral(np.asarray(p_dofs, dtype=PETSc.IntType), comm=comm)
        # hypre's own format, so the AMG holds the block once: PCHYPRE uses a MATHYPRE as it is
        # and would otherwise keep both the AIJ and its converted copy. petsc4py's convert
        # with no `out` converts in place, letting the AIJ storage go.
        self.K = A.createSubMatrix(self.sets[0], self.sets[0])
        self.K.convert('hypre')
        from femtools.ksp import defaults, scoped
        self.amg = PETSc.PC().create(comm)
        self.amg.setOptionsPrefix(prefix)
        self.amg.setOperators(self.K, self.K)
        self.amg.setType('hypre')
        # the options for the setup alone
        with scoped('', defaults(prefix, amg_options)):
            self.amg.setFromOptions()
            self.amg.setUp()
        w, _ = A.createVecs()
        self.xc, self.yc = self.K.createVecs()
        self.scatters = [PETSc.Scatter().create(w, s, self.xc, None) for s in self.sets]
        self.xp = PETSc.Vec().createMPI((len(p_dofs), PETSc.DECIDE), comm=comm)
        self.yp = self.xp.duplicate()
        self.p_scatter = PETSc.Scatter().create(w, self.p_is, self.xp, None)
        self.inv_M = self.xp.duplicate()
        self.inv_M.setArray(1.0 / np.asarray(diag_M, dtype=float))
        w.destroy()

    def apply(self, pc, x, y):
        for sc in self.scatters:
            sc.scatter(x, self.xc, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            self.amg.apply(self.xc, self.yc)
            sc.scatter(self.yc, y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.REVERSE)
        self.p_scatter.scatter(x, self.xp, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        self.yp.pointwiseMult(self.xp, self.inv_M)
        self.p_scatter.scatter(self.yp, y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.REVERSE)

    def destroy(self, pc=None):
        """Called by PETSc with the PC when the KSP goes, and by `free_solver`: once is
        enough."""
        if self.K is None:
            return
        for o in (self.amg, self.K, self.xc, self.yc, self.xp, self.yp, self.inv_M,
                  self.p_scatter, self.p_is, *self.scatters, *self.sets):
            o.destroy()
        self.K, self.scatters, self.sets = None, [], []
