"""Unique dofs: a space's dofs with periodic slaves folded onto their masters.

dolfin's `constrained_domain` gives a periodic space a dofmap without slaves; this is the
DOLFINx counterpart, kept outside the dofmap. `UniqueDofs(V, slaves, masters)` numbers
the space's unique dofs (every dof except the slaves) compactly over the ranks, and
sends each slave to its master's compact index. Everything a solver algorithm sees is
then a PETSc vector or matrix over unique dofs, as on dolfin: no slave entries, so
norms, dot products, means and pointwise reciprocals need no masks.

- Matrices take `lgmap`, in which every local dof (slaves included) points at its
  compact index: DOLFINx inserts element matrices through it, so assembly folds the
  periodic operator directly (see `femtools.dolfinx.assembly`).
- `fold(b_local, b)`: a right-hand side assembled in the space's own (ghosted) layout,
  added into a compact vector; ghost and slave contributions in one reverse scatter,
  which replaces the usual ghost update.
- `prolong(c, u)`: a compact vector into a Function, ghosts and slaves included (one
  forward scatter, which replaces the ghost update and any back-substitution).
- `restrict(u, c)`: a Function's unique values into a compact vector (local copy).

Without slaves the compact numbering is the space's own, and the same code applies.
Constraints with one master and coefficient 1 (periodicity) only; mixed spaces are
numbered dof by dof (their dofmap block size is 1), blocked spaces block by block.
"""
import numpy as np


class UniqueDofs:
    """The unique dofs of `V`, given periodic `slaves` (local dof indices, owned and
    ghost) and `masters` (their masters' global dof indices), e.g. from
    `femtools.dolfinx.periodic.periodic_box`. With neither, the space's own dofs."""

    def __init__(self, V, slaves=None, masters=None):
        from petsc4py import PETSc

        self.V = V
        self.comm = V.mesh.comm
        imap = V.dofmap.index_map
        bs = self.bs = V.dofmap.index_map_bs
        n_own, n_loc = imap.size_local, imap.size_local + imap.num_ghosts
        slaves = np.zeros(0, np.int32) if slaves is None else np.asarray(slaves)
        masters = np.zeros(0, np.int64) if masters is None else np.asarray(masters)
        if bs > 1:  # whole blocks are slaves; work with blocks
            if len(slaves) % bs or np.any(slaves[::bs] % bs) or np.any(masters[::bs] % bs):
                raise ValueError("a blocked space's slaves must be whole blocks")
            slaves, masters = slaves[::bs] // bs, masters[::bs] // bs

        is_slave = np.zeros(n_loc, dtype=bool)
        is_slave[slaves] = True
        owned_unique = np.flatnonzero(~is_slave[:n_own])
        n_unique = len(owned_unique)
        start = self.comm.exscan(n_unique) or 0
        n_global = self.comm.allreduce(n_unique)

        # Compact index of every local block, and of every master, from their owners:
        # a vector over the space's own global numbering holds the owners' compact
        # indices, and one scatter fetches the entries this rank needs.
        own_ids = np.full(n_own, -1.0)
        own_ids[owned_unique] = start + np.arange(n_unique)
        ids = PETSc.Vec().createMPI((n_own, imap.size_global), comm=self.comm)
        ids.setArray(own_ids)
        l2g = imap.local_to_global(np.arange(n_loc, dtype=np.int32)).astype(np.int64)
        need = np.concatenate([l2g, masters]).astype(PETSc.IntType)
        got = PETSc.Vec().createSeq(len(need), comm=PETSc.COMM_SELF)
        sc = PETSc.Scatter().create(ids, PETSc.IS().createGeneral(need, comm=PETSc.COMM_SELF),
                                    got, None)
        sc.scatter(ids, got, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        vals = np.rint(got.getArray(readonly=True)).astype(np.int64)
        for o in (sc, got):
            o.destroy()
        compact = vals[:n_loc]
        compact[slaves] = vals[n_loc:]
        if np.any(compact < 0):
            raise ValueError("a slave's master is itself a slave")
        # Owned slaves too, so compact_of_global resolves any global block.
        ids.setArray(compact[:n_own].astype(float))
        self._ids = ids

        self.n_owned = n_unique * bs           # compact entries owned here
        self.n_global = n_global * bs
        self.start = start * bs
        self.owned_range = (self.start, self.start + self.n_owned)
        comp = np.arange(bs)
        self.compact = (compact[:, None] * bs + comp).ravel()  # per local dof, unrolled
        self.owned_unique = (owned_unique[:, None] * bs + comp).ravel()
        self.is_slave = np.repeat(is_slave, bs)
        self.lgmap = PETSc.LGMap().create(compact.astype(PETSc.IntType), bsize=bs,
                                          comm=self.comm)
        self._template = self.create_vector()
        local = PETSc.Vec().createSeq(n_loc * bs, comm=PETSc.COMM_SELF)
        self._scatter = PETSc.Scatter().create(
            self._template, PETSc.IS().createGeneral(self.compact.astype(PETSc.IntType),
                                                     comm=PETSc.COMM_SELF), local, None)
        local.destroy()
        self._plain = {}

    def compact_of_global(self, gdofs):
        """Compact indices of arbitrary global dofs of the space (unrolled), slaves
        resolved to their masters. Collective: every rank must call it."""
        from petsc4py import PETSc

        gdofs = np.asarray(gdofs, dtype=np.int64)
        blk, comp = np.divmod(gdofs, self.bs)
        need, inverse = np.unique(blk, return_inverse=True)
        got = PETSc.Vec().createSeq(len(need), comm=PETSc.COMM_SELF)
        sc = PETSc.Scatter().create(
            self._ids, PETSc.IS().createGeneral(need.astype(PETSc.IntType), comm=PETSc.COMM_SELF),
            got, None)
        sc.scatter(self._ids, got, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        vals = np.rint(got.getArray(readonly=True)).astype(np.int64)
        sc.destroy()
        got.destroy()
        return vals[inverse] * self.bs + comp

    def create_vector(self):
        """A compact vector (unique dofs, no ghosts)."""
        from petsc4py import PETSc

        v = PETSc.Vec().createMPI((self.n_owned, self.n_global), bsize=self.bs,
                                  comm=self.comm)
        v.set(0.0)
        return v

    def plain_vector(self, key=0):
        """A cached ghosted vector in the space's own layout (scratch for assembly)."""
        from dolfinx.fem import petsc

        if key not in self._plain:
            self._plain[key] = petsc.create_vector(self.V)
        return self._plain[key]

    def restrict(self, u, c):
        """The unique values of Function `u` into compact vector `c` (local copy)."""
        c.getArray()[:] = u.x.array[self.owned_unique]
        return c

    def prolong(self, c, u):
        """Compact vector `c` into Function `u`, ghosts and slaves included."""
        from petsc4py import PETSc

        with u.x.petsc_vec.localForm() as loc:
            self._scatter.scatter(c, loc, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        return u

    def prolong_vec(self, c, plain):
        """Compact vector `c` into a ghosted vector in the space's own layout."""
        from petsc4py import PETSc

        with plain.localForm() as loc:
            self._scatter.scatter(c, loc, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        return plain

    def fold(self, plain, c):
        """A vector assembled in the space's own layout (owned and ghost entries, not
        yet ghost-updated) added into compact vector `c`, which is zeroed first."""
        from petsc4py import PETSc

        c.zeroEntries()
        with plain.localForm() as loc:
            self._scatter.scatter(loc, c, PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
        return c
