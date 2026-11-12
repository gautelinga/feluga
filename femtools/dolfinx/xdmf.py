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
(`XdmfDocument.rows`): nothing is gathered.

Node numbers are DOLFINx's (global vertex and edge indices), not dolfin's; a reader
matches nodes by position (`XDMFFile2.read_checkpoint`), so it does not care. A node's
value comes from its dof through the dof layout (no point search). Scalar and
vector Lagrange Functions of degree 1 or 2 (not mixed: write their components).
"""
import numpy as np

from femtools.ufc import CELL_NAMES, edge_index
from femtools.xdmf import (QUADRATIC, XDMF_TOPOLOGY, XdmfDocument, _int32, _pad3)


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


class _Layout:
    """What writing a field of space V takes, the same every step: the owned cells'
    global node numbers in XDMF's order, the owned nodes' global numbers and
    positions, and the local dof (block) at each owned node."""

    def __init__(self, V):
        from femtools.hdf5_rows import agreed
        mesh = V.mesh
        comm = mesh.comm
        tdim, gdim = mesh.topology.dim, mesh.geometry.dim
        if tdim not in CELL_NAMES:
            raise NotImplementedError("XDMFSeries: triangles and tetrahedra only")
        e = V.ufl_element()
        if V.element.num_sub_elements > 0 and V.dofmap.index_map_bs == 1:
            raise NotImplementedError("XDMFSeries: a mixed space; write its components")
        degree = e.embedded_superdegree
        if degree not in (1, 2) or getattr(e, "discontinuous", False):
            raise NotImplementedError("XDMFSeries: continuous Lagrange of degree 1 or 2")
        mesh.topology.create_connectivity(tdim, 0)
        mesh.topology.create_connectivity(0, tdim)
        vmap = mesh.topology.index_map(0)
        n_cells = mesh.topology.index_map(tdim).size_local
        c2v = mesh.topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)  # all local
        corners = c2v[:n_cells]
        n_vloc = vmap.size_local + vmap.num_ghosts
        gvert = vmap.local_to_global(np.arange(n_vloc, dtype=np.int32)).astype(np.int64)
        # vertex positions, through the geometry (affine cells: a node a vertex)
        import dolfinx.mesh
        vx = mesh.geometry.x[dolfinx.mesh.entities_to_geometry(
            mesh, 0, np.arange(n_vloc, dtype=np.int32)).reshape(-1)]
        n_vert = vmap.size_global
        conn = gvert[corners]
        rows = [gvert[:vmap.size_local]]
        coords = [vx[:vmap.size_local]]
        # each local node's dof (block) through the dof layout's entity dofs, over every
        # local cell (an owned node may lie in ghost cells alone)
        dm = V.dofmap
        cell_dofs = np.asarray(dm.list)
        vdof = [dm.dof_layout.entity_dofs(0, j)[0] for j in range(tdim + 1)]
        node_dofs = np.full(n_vloc, -1, dtype=np.int64)
        node_dofs[c2v] = cell_dofs[:, vdof]
        dofs = [node_dofs[:vmap.size_local]]
        if degree == 2:
            emap = mesh.topology.index_map(1)
            pairs = EDGES_PER_CELL[tdim]
            edges = _edges_of_cells(mesh, n_cells, pairs)
            n_eloc = emap.size_local + emap.num_ghosts
            gedge = emap.local_to_global(np.arange(n_eloc, dtype=np.int32)).astype(np.int64)
            conn = np.hstack([conn, gedge[edges] + n_vert])
            mesh.topology.create_connectivity(1, 0)
            e2v = mesh.topology.connectivity(1, 0).array.reshape(-1, 2)
            rows.append(gedge[:emap.size_local] + n_vert)
            own_e = e2v[:emap.size_local]
            coords.append(0.5 * (vx[own_e[:, 0]] + vx[own_e[:, 1]]))
            n_edges = len(EDGES_PER_CELL[tdim])
            c2e = mesh.topology.connectivity(tdim, 1).array.reshape(-1, n_edges)
            edof = [dm.dof_layout.entity_dofs(1, k)[0] for k in range(n_edges)]
            edge_dofs = np.full(n_eloc, -1, dtype=np.int64)
            edge_dofs[c2e] = cell_dofs[:, edof]
            dofs.append(edge_dofs[:emap.size_local])
            self.n_nodes = n_vert + emap.size_global
        else:
            self.n_nodes = n_vert
        self.conn = conn
        self.n_cells = mesh.topology.index_map(tdim).size_global
        self.cell_offset = comm.exscan(len(conn)) or 0
        self.rows = np.concatenate(rows)
        self.coords = np.vstack(coords)[:, :gdim]
        self.topology = XDMF_TOPOLOGY[(CELL_NAMES[tdim], conn.shape[1])]
        self.gdim = gdim
        self.dofs = np.concatenate(dofs)
        if agreed(comm, bool(np.any(self.dofs < 0))):
            raise RuntimeError("XDMFSeries: an owned node in no local cell")
        self.bs = V.dofmap.index_map_bs


class XDMFSeries:
    """A time series of one or more DOLFINx Functions in XDMFFile2's layout (see the
    module's docstring); `write(u, t)` adds a step."""

    def __init__(self, comm, path):
        self.comm, self.path = comm, path
        self._doc = None
        self._layouts = {}
        self._stored = {}          # (space id) -> (topology ref, geometry ref)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _layout(self, V):
        key = id(V)
        entry = self._layouts.get(key)
        if entry is None or entry[0] is not V:
            entry = self._layouts[key] = (V, _Layout(V))
        return entry[1]

    def write(self, u, t):
        """A step at time t: u (a Function, its values in DOLFINx's array, ghosts
        included) on a grid of its own in the temporal collection. Collective."""
        if self._doc is None:
            self._doc = XdmfDocument(self.path, comm=self.comm if self.comm.size > 1 else None)
        doc, V = self._doc, u.function_space
        L = self._layout(V)
        grid = doc.grid("mesh", doc.collection())
        store = self._stored.get(id(V))
        if store is None:
            group = doc.mesh_group(True)
            store = (doc.slab(group + "/topology", L.cell_offset, _int32(L.conn), L.n_cells),
                     doc.rows(group + "/geometry", L.rows, L.coords, L.n_nodes))
            self._stored[id(V)] = store
        doc.topology(grid, L.conn, L.topology, ref=store[0], shape=(L.n_cells, L.conn.shape[1]))
        doc.geometry(grid, None, ref=store[1], shape=(L.n_nodes, L.gdim))
        doc.time(grid, t)
        values = np.asarray(u.x.array).reshape(-1, L.bs)[L.dofs]
        values = _pad3(values[:, 0] if L.bs == 1 else values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        ref = doc.rows("VisualisationVector/{}".format(doc._next("VisualisationVector")),
                       L.rows, values, L.n_nodes)
        doc.attribute(grid, u.name, values, "Node", ref=ref, shape=(L.n_nodes, values.shape[1]))
        doc.flush()

    def close(self):
        if self._doc is not None:
            self._doc.close()
            self._doc = None
