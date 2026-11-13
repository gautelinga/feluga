"""DOLFINx Functions written as femtools.xdmf.XDMFFile2 writes dolfin's: the same XDMF and
HDF5 layout, so ParaView, `femtools.xdmf.grids`, `merge_series`, `continued` and
`XDMFFile2.read_checkpoint` treat either backend's files alike.

    with XDMFSeries(comm, "u.xdmf") as f:
        f.write(u, t)                        # a time step: one grid in the series

A degree-2 field keeps its midside nodes (`Triangle_6`, `Tetrahedron_10`), a degree-1
one is written on the vertices; nodes are the mesh's vertices, then (degree 2) one per
edge at its midpoint, each with the field's value there -- a periodic field on the whole
mesh, both sides of the period. The mesh is stored once, every step pointing at it (as
twoasis has XDMFFile2 write its series: `rewrite_function_mesh` off), and the XML and
HDF5 are flushed after each step. Every rank writes its own cells and the nodes it owns
(`XdmfDocument.rows`): nothing is gathered. What a write takes of a mesh and a space is
made once and shared by the series writing on them; the mesh's own node numbers and
positions are made when it is stored, and no rank keeps the XML of written steps.

Node numbers are DOLFINx's (global vertex and edge indices), not dolfin's; a reader
matches nodes by position (`XDMFFile2.read_checkpoint`), so it does not care. A node's
value comes from its dof through the dof layout (no point search). Scalar and
vector Lagrange Functions of degree 1 or 2 (not mixed: write their components).
"""
import weakref

import numpy as np

from femtools.lru import WeakCache
from femtools.ufc import CELL_NAMES, edge_index
from femtools.xdmf import (QUADRATIC, XDMF_TOPOLOGY, XdmfDocument, _pad3, node_dofs,
                           node_layout, node_positions, topology_array, is_wide)


def _edges_of_cells(mesh, n_cells, pairs):
    """For each of the first n_cells cells, its edges (local indices) in the order of
    the reference vertex pairs `pairs`: a cell's local edge e joins the reference
    vertices of edge e of the reference cell (basix's, UFC's)."""
    tdim = mesh.topology.dim
    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(tdim, 1)
    E = mesh.topology.connectivity(tdim, 1).array.reshape(-1, len(pairs))[:n_cells]
    index = edge_index(tdim)
    return E[:, [index[a, b] for a, b in pairs]]


EDGES_PER_CELL = {2: QUADRATIC["triangle"][1], 3: QUADRATIC["tetrahedron"][1]}

#: a mesh's node layouts, by degree (`_MeshNodes`), shared by the series writing fields
#: on it and dropped with the mesh
_MESHES = WeakCache()
#: a space's node dofs (`_space_dofs`), shared likewise and dropped with the space
_SPACES = WeakCache()


def _topology(mesh, degree):
    """`node_layout`'s arguments for a DOLFINx mesh at `degree`, and the local vertices'
    positions: a dict of `c2v` (every local cell's vertices), `n_cells` (the owned
    cells), `gvert`, `owned` (the owned vertices), `vx`, and at degree 2 `edges` (with
    `c2e`, every local cell's edges, in the reference pairs' order)."""
    import dolfinx.mesh
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    mesh.topology.create_connectivity(0, tdim)
    cmap, vmap = mesh.topology.index_map(tdim), mesh.topology.index_map(0)
    n_vloc = vmap.size_local + vmap.num_ghosts
    out = dict(c2v=mesh.topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1),
               n_cells=cmap.size_local, n_cells_global=cmap.size_global,
               gvert=vmap.local_to_global(np.arange(n_vloc, dtype=np.int32)).astype(np.int64),
               owned=np.arange(n_vloc) < vmap.size_local, n_vertices=vmap.size_global,
               edges=None)
    # vertex positions, through the geometry (affine cells: a node a vertex)
    out["vx"] = mesh.geometry.x[dolfinx.mesh.entities_to_geometry(
        mesh, 0, np.arange(n_vloc, dtype=np.int32)).reshape(-1)]
    if degree == 2:
        pairs = EDGES_PER_CELL[tdim]
        cells = _edges_of_cells(mesh, out["n_cells"], pairs)
        emap = mesh.topology.index_map(1)
        n_eloc = emap.size_local + emap.num_ghosts
        mesh.topology.create_connectivity(1, 0)
        out["edges"] = dict(
            gedge=emap.local_to_global(np.arange(n_eloc, dtype=np.int32)).astype(np.int64),
            cells=cells, ends=mesh.topology.connectivity(1, 0).array.reshape(-1, 2),
            owned=np.arange(n_eloc) < emap.size_local, n=emap.size_global,
            c2e=mesh.topology.connectivity(tdim, 1).array.reshape(-1, len(pairs)))
    return out


class _MeshNodes:
    """What writing a field of a degree on a mesh takes of the mesh, the same every step
    (femtools.xdmf.node_layout): the owned nodes' global numbers (`rows`, read-only, with
    `plans`, their routing, `XdmfDocument.rows`) and masks, the counts, the cells' first
    row, and the topology's type and columns. The cells' node numbers and the nodes'
    positions, which a write of the mesh alone takes, are made then (`mesh_nodes`)."""

    def __init__(self, mesh, degree):
        if mesh.topology.dim not in CELL_NAMES:
            raise NotImplementedError("XDMFSeries: triangles and tetrahedra only")
        self._mesh, self.degree = weakref.ref(mesh), degree
        t = _topology(mesh, degree)
        L = node_layout(mesh.comm, t["gvert"], t["c2v"][:t["n_cells"]], t["owned"],
                        t["n_vertices"], t["n_cells_global"], t["edges"])
        self.rows, self.owned = L["rows"], L["owned"]
        self.rows.flags.writeable = False
        self.n_nodes, self.n_cells, self.cell_offset = (L["n_nodes"], L["n_cells"],
                                                        L["cell_offset"])
        self.per_cell = L["conn"].shape[1]
        self.topology = XDMF_TOPOLOGY[(CELL_NAMES[mesh.topology.dim], self.per_cell)]
        self.gdim = mesh.geometry.dim
        self.plans = {}

    def mesh_nodes(self):
        """`(conn, coords)`: this rank's cells' global node numbers and the positions of
        the nodes it writes. Collective."""
        mesh = self._mesh()
        t = _topology(mesh, self.degree)
        L = node_layout(mesh.comm, t["gvert"], t["c2v"][:t["n_cells"]], t["owned"],
                        t["n_vertices"], t["n_cells_global"], t["edges"])
        return L["conn"], node_positions(t["vx"], L)[:, :self.gdim]


def _space_dofs(V, nodes):
    """The local dof (block) at each node this rank writes (`_MeshNodes`), through the
    dof layout's entity dofs over every local cell (an owned node may lie in ghost cells
    alone). Refused on every rank when an owned node is in no local cell."""
    from femtools.hdf5_rows import agreed
    mesh = V.mesh
    tdim = mesh.topology.dim
    t = _topology(mesh, nodes.degree)
    dm = V.dofmap
    cell_dofs = np.asarray(dm.list)
    vdof = [dm.dof_layout.entity_dofs(0, j)[0] for j in range(tdim + 1)]
    vertex_dofs = np.full(len(t["gvert"]), -1, dtype=np.int64)
    vertex_dofs[t["c2v"]] = cell_dofs[:, vdof]
    edge_dofs = None
    if nodes.degree == 2:
        c2e = t["edges"]["c2e"]
        edof = [dm.dof_layout.entity_dofs(1, k)[0] for k in range(c2e.shape[1])]
        edge_dofs = np.full(len(t["edges"]["gedge"]), -1, dtype=np.int64)
        edge_dofs[c2e] = cell_dofs[:, edof]
    dofs = node_dofs(dict(owned=nodes.owned, ends=None), vertex_dofs, edge_dofs)[0]
    if agreed(mesh.comm, bool(np.any(dofs < 0))):
        raise RuntimeError("XDMFSeries: an owned node in no local cell")
    return dofs


def _check_space(V):
    """The degree of a space XDMFSeries writes: continuous Lagrange of degree 1 or 2,
    scalar or vector, not mixed."""
    e = V.ufl_element()
    if V.element.num_sub_elements > 0 and V.dofmap.index_map_bs == 1:
        raise NotImplementedError("XDMFSeries: a mixed space; write its components")
    degree = e.embedded_superdegree
    if degree not in (1, 2) or getattr(e, "discontinuous", False):
        raise NotImplementedError("XDMFSeries: continuous Lagrange of degree 1 or 2")
    return degree


class XDMFSeries:
    """A time series of one or more DOLFINx Functions in XDMFFile2's layout (see the
    module's docstring); `write(u, t)` adds a step. What a write takes of the mesh and
    the space is shared with the other series writing on them."""

    def __init__(self, comm, path):
        self.comm, self.path = comm, path
        self._doc = None
        self._stored = {}          # (mesh id, degree) -> (topology ref, geometry ref)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def write(self, u, t):
        """A step at time t: u (a Function, its values in DOLFINx's array, ghosts
        included) on a grid of its own in the temporal collection. Collective."""
        if self._doc is None:
            self._doc = XdmfDocument(self.path, comm=self.comm if self.comm.size > 1 else None)
        doc, V = self._doc, u.function_space
        degree = _check_space(V)
        nodes = _MESHES.get((V.mesh,), dict)
        if degree not in nodes:
            nodes[degree] = _MeshNodes(V.mesh, degree)
        L = nodes[degree]
        dofs = _SPACES.get((V,), lambda: _space_dofs(V, L))
        grid = doc.grid("mesh", doc.collection())
        key = (id(V.mesh), degree)
        store = self._stored.get(key)
        if store is None:
            group = doc.mesh_group(True)
            conn, coords = L.mesh_nodes()
            store = (doc.slab(group + "/topology", L.cell_offset,
                              topology_array(conn, L.n_nodes), L.n_cells),
                     doc.rows(group + "/geometry", L.rows, coords, L.n_nodes))
            conn = coords = None
            self._stored[key] = store
        doc.topology(grid, None, L.topology, ref=store[0], shape=(L.n_cells, L.per_cell),
                     wide=is_wide(L.n_nodes))
        doc.geometry(grid, None, ref=store[1], shape=(L.n_nodes, L.gdim))
        doc.time(grid, t)
        bs = V.dofmap.index_map_bs
        values = np.asarray(u.x.array).reshape(-1, bs)[dofs]
        values = _pad3(values[:, 0] if bs == 1 else values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        ref = doc.rows("VisualisationVector/{}".format(doc._next("VisualisationVector")),
                       L.rows, values, L.n_nodes, plans=L.plans)
        doc.attribute(grid, u.name, values, "Node", ref=ref, shape=(L.n_nodes, values.shape[1]))
        doc.seal()                 # the step is complete: no rank holds its XML after
        doc.flush()

    def close(self):
        if self._doc is not None:
            self._doc.close()
            self._doc = None
