"""A DOLFINx space's cells and dofs in dolfin's (UFC's) order: what feluga.tables
(and through it the checkpoints, feluga.dolfinx.checkpoint, and the nested transfers)
reads of a DOLFINx space.

- Cells by global id: a DOLFINx mesh's original cell index, which is dolfin's cell index
  for a mesh from the same input (a file, or the generators, which number alike).
- Dofs in a cell in dolfin's order: the vertices by increasing global index (the original
  vertex index), then for degree 2 the edges (feluga.ufc), a mixed element's components
  one after the other, a vector's unrolled -- found through the dof layout's entity dofs.

Continuous Lagrange elements of degree 1 and 2 on affine triangles and tetrahedra, vector
(blocked) elements of them, and mixed elements of them (Taylor-Hood too); and piecewise
constants (DG0), a cell's one dof.
"""
import numpy as np

from feluga.lru import WeakCache
from feluga.ufc import CELL_NAMES, UFC_EDGES, edge_index


def components(V):
    """V's scalar components in the order dolfin's cell dofs hold them: a mixed space's
    one after the other, a vector (blocked) space's or sub-space's unrolled."""
    n = V.element.num_sub_elements
    if n == 0:
        return [V]
    return [leaf for i in range(n) for leaf in components(V.sub(i))]


def is_dg0(V):
    """Whether V is piecewise constant (DG0, scalar or vector): one value per cell."""
    e = V.ufl_element()
    return (bool(getattr(e, "discontinuous", False))
            and e.family_name in ("P", "Lagrange", "DG", "Discontinuous Lagrange")
            and e.embedded_superdegree == 0)


def dg0_values(f, n_cells):
    """The DG0 Function f's values on this rank's first n_cells cells (its owned ones,
    for n_cells = the cells' local size), in local cell order: (n_cells, block size)."""
    V = f.function_space
    bs = V.dofmap.index_map_bs
    dofs = np.asarray(V.dofmap.list)[:n_cells, 0]
    return np.asarray(f.x.array).reshape(-1, bs)[dofs].copy()


def layout_degree(S):
    """The degree of the scalar space S in dolfin's layout, refusing what that layout
    here does not hold: 1 or 2 for continuous Lagrange on simplices, 0 for DG0."""
    if is_dg0(S):
        return 0
    e = S.ufl_element()
    if getattr(e, "discontinuous", False) or e.family_name not in ("P", "Lagrange"):
        raise NotImplementedError("dolfin layout: continuous Lagrange elements on "
                                  "simplices only, not {}".format(e))
    degree = e.embedded_superdegree
    if degree not in (1, 2):
        raise NotImplementedError("dolfin layout: degree 1 or 2, not {}".format(degree))
    return degree


def _blocked(V):
    return V.dofmap.dof_layout.block_size > 1


def signature(V):
    """dolfin's signature of V's element: scalar Lagrange, vector, or mixed of them
    (a Taylor-Hood space's `MixedElement(VectorElement(...), FiniteElement(...))`)."""
    tdim = V.mesh.topology.dim

    def sig(S):
        n = S.element.num_sub_elements
        if n == 0:
            degree = layout_degree(S)
            return "FiniteElement('{}', {}, {})".format(
                "Discontinuous Lagrange" if degree == 0 else "Lagrange", CELL_NAMES[tdim],
                degree)
        parts = [sig(S.sub(i)) for i in range(n)]
        if _blocked(S):
            return "VectorElement({}, dim={})".format(parts[0], n)
        return "MixedElement({})".format(", ".join(parts))

    return sig(V)


def _vertex_order(mesh):
    """This rank's own cells' local vertices (their reference positions) in dolfin's
    (UFC's) order: by increasing input global index. (n, tdim+1)."""
    tdim = mesh.topology.dim
    if tdim not in UFC_EDGES or mesh.geometry.cmaps[0].degree != 1:
        raise NotImplementedError("dolfin layout: affine triangles and tetrahedra only")
    n = mesh.topology.index_map(tdim).size_local
    G = mesh.geometry.dofmaps[0][:n]
    gv = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
    return np.argsort(gv[G], axis=1, kind="stable")


def ufc_vertices(mesh):
    """This rank's own cells' geometry nodes (their vertices, the mesh being affine) in
    dolfin's (UFC's) order: by increasing input global index. (n, tdim+1)."""
    n = mesh.topology.index_map(mesh.topology.dim).size_local
    G = mesh.geometry.dofmaps[0][:n]
    return np.take_along_axis(G, _vertex_order(mesh), axis=1)


_TABLES = WeakCache()


def dolfin_cell_dofs(V):
    """`(gc, D)`: this rank's own cells' global ids (uint64), and for each cell V's local
    dofs in dolfin's order (vector components unrolled), a row a cell. From the dof
    layout's entity dofs: a cell's vertex j and edge e (reference numbering) carry dofs
    `entity_dofs(0, j)` and `entity_dofs(1, e)` of its row. Built once per space (kept
    while the space lives)."""
    return _TABLES.get((V,), lambda: _dolfin_cell_dofs(V))


def _dolfin_cell_dofs(V):
    mesh = V.mesh
    tdim = mesh.topology.dim
    n = mesh.topology.index_map(tdim).size_local
    perm = _vertex_order(mesh)                       # (n, d+1): reference vertex per slot
    pairs = np.asarray(UFC_EDGES[tdim])              # dolfin's edges by dolfin's slots
    ref_edges = edge_index(tdim)[perm[:, pairs[:, 0]], perm[:, pairs[:, 1]]]   # (n, n_e)
    blocks = []
    for S in components(V):
        layout = S.dofmap.dof_layout
        if layout_degree(S) == 0:                    # the cell's one dof
            cols = np.full((n, 1), layout.entity_dofs(tdim, 0)[0])
        else:
            cols = np.array([layout.entity_dofs(0, j)[0] for j in range(tdim + 1)])[perm]
        if layout_degree(S) == 2:
            edof = np.array([layout.entity_dofs(1, e)[0] for e in range(len(pairs))])
            cols = np.concatenate([cols, edof[ref_edges]], axis=1)
        D = np.asarray(S.dofmap.list[:n])            # V's local dofs, DOLFINx's order
        if D.shape[1] != cols.shape[1]:
            raise NotImplementedError("dolfin layout: unexpected dofs per cell")
        blocks.append(np.take_along_axis(D, cols, axis=1))
    D = np.concatenate(blocks, axis=1)
    gc = np.asarray(mesh.topology.original_cell_index[:n], dtype=np.uint64)
    return gc, D
