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


def _node_keys(V, n_local):
    """Sort keys that put the components of a mixed space at a node next to each other:
    component i of the node of component 0's dof d gets n_sub*d + i. The components'
    cell dofmaps pair them (the same local dof of the same cell is at the same node).
    Spaces that are not mixed, or whose components differ, keep their order. Returns
    the keys and the number of dofs per node, n_sub or 1 (key // it is the node)."""
    n_sub = V.element.num_sub_elements
    keys = np.arange(n_local, dtype=np.int64)
    if n_sub == 0 or V.dofmap.index_map_bs != 1:
        return keys, 1
    elements = [V.sub(i).ufl_element() for i in range(n_sub)]
    if any(e != elements[0] for e in elements[1:]):
        return keys, 1
    first = V.sub(0).dofmap.list.ravel().astype(np.int64)
    keys = np.full(n_local, -1, dtype=np.int64)
    for i in range(n_sub):
        keys[V.sub(i).dofmap.list.ravel()] = n_sub * first + i
    missing = keys < 0  # dofs on no cell (none in a Lagrange space): keep them last
    keys[missing] = n_sub * n_local + np.flatnonzero(missing)
    return keys, n_sub


def _gps_order(V, owned_unique, rep, nodes):
    """`owned_unique` reordered by the Gibbs-Poole-Stockmeyer ordering (DOLFINx's
    bandwidth reduction) of the graph of their nodes, in which two nodes are joined when
    a cell has dofs at both. `rep[i]` is the owned unique dof that local dof i stands
    for (a slave its master), -1 if none; `nodes[i]` is the node of local dof i. The
    dofs of a node keep their order in `owned_unique`."""
    from dolfinx.cpp.graph import AdjacencyList_int32, reorder_gps

    node_ids, node_of = np.unique(nodes[owned_unique], return_inverse=True)
    n_nodes = len(node_ids)
    index = np.full(len(rep), -1, dtype=np.int32)  # node (into node_ids) of an owned unique dof
    index[owned_unique] = node_of
    r = rep[V.dofmap.list.reshape(V.dofmap.list.shape[0], -1)]
    cells = np.where(r >= 0, index[np.maximum(r, 0)], -1)
    del r, index
    edges = [np.zeros(0, np.int64)]  # each edge once, as lower * n_nodes + higher
    for i in range(cells.shape[1]):
        for j in range(i + 1, cells.shape[1]):
            a, b = cells[:, i], cells[:, j]
            keep = (a >= 0) & (b >= 0) & (a != b)
            a, b = a[keep], b[keep]
            edges.append(np.minimum(a, b).astype(np.int64) * n_nodes + np.maximum(a, b))
    del cells
    edges = np.concatenate(edges)
    edges.sort()  # by lower node, then higher; in place
    first = np.ones(len(edges), dtype=bool)
    first[1:] = edges[1:] != edges[:-1]
    edges = edges[first]
    del first
    lower = (edges // n_nodes).astype(np.int32)
    higher = (edges % n_nodes).astype(np.int32)
    del edges
    # Each node's neighbours in ascending order: those below it (where it is the higher)
    # first, then those above, each part already ascending; a stable sort keeps that.
    src = np.concatenate([higher, lower])
    dst = np.concatenate([lower, higher])
    del lower, higher
    dst = dst[np.argsort(src, kind="stable")]
    offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    np.cumsum(np.bincount(src, minlength=n_nodes), out=offsets[1:])
    del src
    place = np.asarray(reorder_gps(AdjacencyList_int32(dst, offsets)),
                       dtype=np.int64)  # place[node]: its position in the new order
    return owned_unique[np.argsort(place[node_of], kind="stable")]


class UniqueDofs:
    """The unique dofs of `V`, given periodic `slaves` (local dof indices, owned and
    ghost) and `masters` (their masters' global dof indices), e.g. from
    `femtools.dolfinx.periodic.periodic_box`. With neither, the space's own dofs.

    The compact numbering keeps the space's order of the unique dofs, except with
    `interleave` on a mixed space whose components share one element: then the
    components at a node are numbered one after the other, as dolfin numbers a mixed
    space (DOLFINx numbers each component's dofs of a cell in a run). Incomplete
    factorizations such as ILU(0), which depend on the order, then work as on dolfin.

    With `reorder`, each rank numbers its unique dofs in a bandwidth-reducing order of
    the periodic (folded) cell graph, node by node (with `interleave`, a mixed space's
    components at a node stay together). The space's own order does not see the
    periodic couplings; in the reduced order incomplete factorizations converge in
    fewer iterations, and algebraic multigrid cycles (BoomerAMG) take less time.
    """

    def __init__(self, V, slaves=None, masters=None, interleave=False, reorder=False):
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
        keys, per_node = _node_keys(V, n_loc) if interleave else (None, 1)
        if interleave:
            owned_unique = owned_unique[np.argsort(keys[owned_unique], kind="stable")]
        if reorder:
            lo = imap.local_range[0]
            rep = np.full(n_loc, -1, dtype=np.int32)
            rep[owned_unique] = owned_unique
            here = (masters >= lo) & (masters < lo + n_own)
            rep[slaves[here]] = masters[here] - lo
            nodes = np.arange(n_loc) if keys is None else keys // per_node
            owned_unique = _gps_order(V, owned_unique, rep, nodes)
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
        lo = imap.local_range[0]
        self._block_range = (lo, lo + n_own)
        ghosts = np.asarray(imap.ghosts, dtype=np.int64)
        self._ghost_order = np.argsort(ghosts, kind="stable")
        self._ghosts = ghosts[self._ghost_order]

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
        self._plain = None

    def local_of_global(self, gdofs):
        """Local indices of global dofs of the space (unrolled), -1 where a dof is
        neither owned nor a ghost here."""
        g = np.asarray(gdofs, dtype=np.int64)
        blk, comp = np.divmod(g, self.bs) if self.bs > 1 else (g, 0)
        lo, hi = self._block_range
        out = np.full(len(blk), -1, dtype=np.int64)
        own = (blk >= lo) & (blk < hi)
        out[own] = blk[own] - lo
        rest = np.flatnonzero(~own)
        if len(rest) and len(self._ghosts):
            pos = np.searchsorted(self._ghosts, blk[rest])
            pos[pos == len(self._ghosts)] = 0
            hit = self._ghosts[pos] == blk[rest]
            out[rest[hit]] = hi - lo + self._ghost_order[pos[hit]]
        return np.where(out >= 0, out * self.bs + comp, -1)

    def compact_of_global(self, gdofs):
        """Compact indices of arbitrary global dofs of the space (unrolled), slaves
        resolved to their masters: the local dofs' from `compact`, the others' from their
        owners. Collective: every rank must call it."""
        from petsc4py import PETSc

        local = self.local_of_global(gdofs)
        out = np.empty(len(local), dtype=np.int64)
        known = local >= 0
        out[known] = self.compact[local[known]]
        far = np.asarray(gdofs, dtype=np.int64)[~known]
        blk, comp = np.divmod(far, self.bs)
        need, inverse = np.unique(blk, return_inverse=True)
        got = PETSc.Vec().createSeq(len(need), comm=PETSc.COMM_SELF)
        sc = PETSc.Scatter().create(
            self._ids, PETSc.IS().createGeneral(need.astype(PETSc.IntType), comm=PETSc.COMM_SELF),
            got, None)
        sc.scatter(self._ids, got, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        vals = np.rint(got.getArray(readonly=True)).astype(np.int64)
        sc.destroy()
        got.destroy()
        out[~known] = vals[inverse] * self.bs + comp
        return out

    def create_vector(self):
        """A compact vector (unique dofs, no ghosts)."""
        from petsc4py import PETSc

        v = PETSc.Vec().createMPI((self.n_owned, self.n_global), bsize=self.bs,
                                  comm=self.comm)
        v.set(0.0)
        return v

    def plain_vector(self):
        """A ghosted vector in the space's own layout, made once: scratch for assembly."""
        from dolfinx.fem import petsc

        if self._plain is None:
            self._plain = petsc.create_vector(self.V)
        return self._plain

    def restrict(self, u, c):
        """The unique values of Function `u` into compact vector `c` (local copy)."""
        np.take(u.x.array, self.owned_unique, out=c.getArray())
        return c

    def prolong(self, c, u):
        """Compact vector `c` into Function `u`, ghosts and slaves included."""
        from petsc4py import PETSc

        with u.x.petsc_vec.localForm() as loc:
            self._scatter.scatter(c, loc, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        return u

    def fold(self, plain, c):
        """A vector assembled in the space's own layout (owned and ghost entries, not
        yet ghost-updated) added into compact vector `c`, which is zeroed first."""
        from petsc4py import PETSc

        c.zeroEntries()
        with plain.localForm() as loc:
            self._scatter.scatter(loc, c, PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
        return c
