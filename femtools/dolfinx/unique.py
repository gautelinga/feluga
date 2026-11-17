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


def _scalar_leaves(V):
    """V's scalar sub-spaces, depth first (a vector component's own components too); their
    dofmaps are in V's numbering."""
    n = V.element.num_sub_elements
    if n == 0:
        return [V]
    return [leaf for i in range(n) for leaf in _scalar_leaves(V.sub(i))]


def _node_keys(V, n_local):
    """Sort keys that put the scalar components of a mixed space at a node next to each
    other: component i at the node of the reference component's dof d gets n_sub*d + i.
    The components are the space's scalar leaves (a vector sub-space's by component, so
    [P2^3, P1] and [P2, P2, P2, P1] alike); the reference is the one with the most dofs a
    cell; a component's dof is at the node of the reference's dof on the same cell entity
    (vertex, edge, ...), so Taylor-Hood's pressure joins the velocity at the vertices.
    Spaces that are not mixed, or whose components place dofs on an entity otherwise than
    none or as many as the reference (P2 against P3 edges), keep their order.
    Returns the keys, the number of dofs per node (n_sub or 1: key // it is the node) and
    the reference component's cell dofs (None when the order is kept)."""
    keys = np.arange(n_local, dtype=np.int64)
    if V.element.num_sub_elements == 0 or V.dofmap.index_map_bs != 1:
        return keys, 1, None
    subs = _scalar_leaves(V)
    if any(s.dofmap.index_map_bs != 1 or s.dofmap.bs != 1 for s in subs):
        return keys, 1, None
    n_sub = len(subs)
    layouts = [s.dofmap.dof_layout for s in subs]
    ref = int(np.argmax([lay.num_dofs for lay in layouts]))
    from dolfinx.cpp.mesh import cell_num_entities
    cell = V.mesh.topology.cell_type
    positions = []                 # each component's local dofs -> the reference's
    for lay in layouts:
        pos = np.full(lay.num_dofs, -1, dtype=np.int64)
        for dim in range(V.mesh.topology.dim + 1):
            for e in range(cell_num_entities(cell, dim)):
                mine = np.asarray(lay.entity_dofs(dim, e), dtype=np.int64)
                theirs = np.asarray(layouts[ref].entity_dofs(dim, e), dtype=np.int64)
                if len(mine) not in (0, len(theirs)):
                    return keys, 1, None
                pos[mine] = theirs[:len(mine)]
        if np.any(pos < 0):
            return keys, 1, None
        positions.append(pos)
    ref_cells = subs[ref].dofmap.list
    first = ref_cells.astype(np.int64)
    keys = np.full(n_local, -1, dtype=np.int64)
    for i, (sub, pos) in enumerate(zip(subs, positions)):
        keys[sub.dofmap.list] = n_sub * first[:, pos] + i
    missing = np.flatnonzero(keys < 0)  # dofs on no cell (none in a Lagrange space): last,
    keys[missing] = n_sub * (n_local + np.arange(len(missing)))  # each a node of its own
    return keys, n_sub, ref_cells


def _rcm_order(owned_unique, rep, nodes, cell_dofs):
    """`owned_unique` reordered by the reverse Cuthill-McKee ordering (PETSc's) of the
    graph of their nodes, in which two nodes are joined when a cell has dofs at both.
    `rep[i]` is the owned unique dof that local dof i stands for (a slave its master),
    -1 if none; `nodes[i]` is the node of local dof i; `cell_dofs` each cell's local
    dofs, one at each of its nodes enough (a mixed space's reference component: the
    graph is then built from its pairs alone, 45 a P2 tetrahedron against 561 for
    Taylor-Hood's every dof). The dofs of a node keep their order in `owned_unique`."""
    from petsc4py import PETSc

    node_ids, node_of = np.unique(nodes[owned_unique], return_inverse=True)
    n_nodes = len(node_ids)
    index = np.full(len(rep), -1, dtype=np.int32)  # node (into node_ids) of an owned unique dof
    index[owned_unique] = node_of
    r = rep[cell_dofs.reshape(cell_dofs.shape[0], -1)]
    cells = np.where(r >= 0, index[np.maximum(r, 0)], -1)
    del r, index
    n_cols = cells.shape[1]
    edges = np.empty(len(cells) * n_cols * (n_cols - 1) // 2, dtype=np.int64)
    at = 0                         # each edge as lower * n_nodes + higher, filled in place
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            a, b = cells[:, i], cells[:, j]
            keep = (a >= 0) & (b >= 0) & (a != b)
            a, b = a[keep], b[keep]
            n = len(a)
            edges[at:at + n] = np.minimum(a, b)
            edges[at:at + n] *= n_nodes
            edges[at:at + n] += np.maximum(a, b)
            at += n
    del cells
    edges = edges[:at]
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
    if n_nodes == 0:
        return owned_unique
    graph = PETSc.Mat().createAIJ((n_nodes, n_nodes), comm=PETSc.COMM_SELF, csr=(
        offsets.astype(PETSc.IntType), dst.astype(PETSc.IntType), np.ones(len(dst))))
    del dst, offsets
    rows, cols = graph.getOrdering(PETSc.Mat.OrderingType.RCM)
    order = rows.getIndices()                   # the nodes in their new order
    for o in (rows, cols, graph):
        o.destroy()
    place = np.empty(n_nodes, dtype=np.int64)   # place[node]: its position in it
    place[order] = np.arange(n_nodes)
    return owned_unique[np.argsort(place[node_of], kind="stable")]


class UniqueDofs:
    """The unique dofs of `V`, given periodic `slaves` (local dof indices, owned and
    ghost) and `masters` (their masters' global dof indices), e.g. from
    `femtools.dolfinx.periodic.periodic_box`. With neither, the space's own dofs.

    The compact numbering keeps the space's order of the unique dofs, except with
    `interleave` on a mixed space: then its scalar components at a node are numbered one
    after the other (DOLFINx numbers each component's dofs of a cell in a run; dolfin
    interleaves a mixed space of equal elements so). Components with unlike elements meet
    at the cell entities they share (`_node_keys`: Taylor-Hood's pressure at the
    vertices, with the velocity), a vector component by its own components. Incomplete
    factorizations such as ILU(0), which depend on the order, work better so.

    With `reorder`, each rank numbers its unique dofs in a bandwidth-reducing order of
    the periodic (folded) cell graph, node by node (with `interleave`, a mixed space's
    components at a node stay together). The space's own order does not see the
    periodic couplings; in the reduced order incomplete factorizations converge in
    fewer iterations, and algebraic multigrid cycles (BoomerAMG) take less time.
    """

    def __init__(self, V, slaves=None, masters=None, interleave=False, reorder=False):
        from petsc4py import PETSc

        from femtools.garbage import register
        from femtools.memory import release

        self.V = V
        self.comm = V.mesh.comm
        register(self.comm)
        imap = V.dofmap.index_map
        bs = self.bs = V.dofmap.index_map_bs
        n_own, n_loc = imap.size_local, imap.size_local + imap.num_ghosts
        slaves = np.zeros(0, np.int32) if slaves is None else np.asarray(slaves)
        masters = np.zeros(0, np.int64) if masters is None else np.asarray(masters)
        from femtools.hdf5_rows import agreed
        if bs > 1:  # whole blocks are slaves; work with blocks
            if agreed(self.comm, len(slaves) % bs or np.any(slaves[::bs] % bs)
                      or np.any(masters[::bs] % bs)):      # every rank raises, or none
                raise ValueError("a blocked space's slaves must be whole blocks")
            slaves, masters = slaves[::bs] // bs, masters[::bs] // bs

        # whether any rank has a periodic slave: a Function's array written directly is
        # then not the compact vector's prolongation until it is prolonged again
        self.any_slave = bool(agreed(self.comm, len(slaves) > 0))
        is_slave = np.zeros(n_loc, dtype=bool)
        is_slave[slaves] = True
        owned_unique = np.flatnonzero(~is_slave[:n_own])
        keys, per_node, ref_cells = (_node_keys(V, n_loc) if interleave
                                     else (None, 1, None))
        if interleave:
            owned_unique = owned_unique[np.argsort(keys[owned_unique], kind="stable")]
        if reorder:
            lo = imap.local_range[0]
            rep = np.full(n_loc, -1, dtype=np.int32)
            rep[owned_unique] = owned_unique
            here = (masters >= lo) & (masters < lo + n_own)
            rep[slaves[here]] = masters[here] - lo
            nodes = np.arange(n_loc) if keys is None else keys // per_node
            cell_dofs = V.dofmap.list if ref_cells is None else ref_cells
            owned_unique = _rcm_order(owned_unique, rep, nodes, cell_dofs)
            release()       # the graph's sorts, freed below glibc's mmap threshold
        n_unique = len(owned_unique)
        start = self.comm.exscan(n_unique) or 0
        n_global = self.comm.allreduce(n_unique)
        if n_global * bs > np.iinfo(PETSc.IntType).max:
            raise OverflowError(
                "{} unique dofs exceed this PETSc's {}-bit indices: a PETSc (and hypre) "
                "built with 64-bit indices is needed".format(
                    n_global * bs, 8 * np.dtype(PETSc.IntType).itemsize))

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
        if agreed(self.comm, np.any(compact < 0)):
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
        # DOLFINx's own array, as it stands (a backend Function's `x.array` would first
        # bring its owned entries up to the compact vector: the write being read lost)
        np.take(u._cpp_object.x.array, self.owned_unique, out=c.getArray())
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
