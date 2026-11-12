"""DOLFINx meshes from the HDF5 files dolfin writes (`HDF5File.write(mesh, name)`,
femtools.dolfin_layout.write_mesh, meshtools' generators): `name/coordinates` (a vertex a
row, by global vertex index), `name/topology` (a cell a row, as global vertex indices,
with the `celltype` attribute) and `name/cell_indices` (each row's global cell id).

    mesh = read_mesh("mesh.h5", comm)

The mesh's numbering is the file's: a cell's original index (`topology.original_cell_index`)
is its global id in the file, a vertex's original index
(`geometry.input_global_indices`) its row -- which is how dolfin numbers the same file,
so a checkpoint names the same cells on either backend (femtools.dolfinx.checkpoint).
Every rank reads its block of the cells, in order of their global ids, and its block of
the coordinates; nothing is read whole. Affine triangles and tetrahedra.
"""
import numpy as np


def _block(n, comm):
    r, size = comm.rank, comm.size
    return n * r // size, n * (r + 1) // size


def read_mesh(path, comm, name="mesh"):
    """The mesh `name` in the HDF5 file `path`, distributed over `comm`. Collective."""
    import h5py
    import basix.ufl
    import dolfinx.mesh
    from femtools.hdf5_rows import fetch_rows, rows_of_labels

    with h5py.File(path, "r") as h5:
        group = h5[name]
        n_cells = int(group["topology"].shape[0])
        n_x, gdim = (int(v) for v in group["coordinates"].shape)
        celltype = group["topology"].attrs["celltype"]
        celltype = celltype.decode() if isinstance(celltype, bytes) else str(celltype)
        has_ids = "cell_indices" in group
        lo, hi = _block(n_x, comm)
        x = np.asarray(group["coordinates"][lo:hi], dtype=np.float64)
    if celltype not in ("triangle", "tetrahedron"):
        raise NotImplementedError("read_mesh: triangles and tetrahedra, not " + celltype)
    lo, hi = _block(n_cells, comm)
    ids = np.arange(lo, hi, dtype=np.int64)
    rows = (rows_of_labels(path, name + "/cell_indices", ids, comm if comm.size > 1 else None)
            if has_ids else ids)
    cells = fetch_rows(path, name + "/topology", rows, comm if comm.size > 1 else None)
    element = basix.ufl.element("Lagrange", celltype, 1, shape=(gdim,))
    return dolfinx.mesh.create_mesh(comm, np.asarray(cells, dtype=np.int64), element, x)


def write_mesh(path, mesh, name="mesh", mode="w"):
    """The mesh as dolfin's `HDF5File(path, mode).write(mesh, name)` writes one
    (femtools.dolfin_layout.write_mesh): `coordinates` by original vertex index,
    `topology` the own cells rank after rank (each cell's vertices in increasing order,
    as dolfin orders them), `cell_indices` their original indices. Collective."""
    from femtools.dolfin_layout import _offsets
    from femtools.hdf5_rows import write_ordered
    comm = mesh.comm
    tdim, gdim = mesh.topology.dim, mesh.geometry.dim
    gmap = mesh.geometry.index_map()
    gv = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
    n_own_x = gmap.size_local
    own = mesh.topology.index_map(tdim).size_local
    cells = np.sort(gv[mesh.geometry.dofmaps[0][:own]], axis=1)
    first, partition = _offsets(comm, own)
    rows = first + np.arange(own, dtype=np.int64)
    n_c = mesh.topology.index_map(tdim).size_global
    cellname = {2: "triangle", 3: "tetrahedron"}[tdim]
    write_ordered(path, name + "/coordinates", gv[:n_own_x], mesh.geometry.x[:n_own_x, :gdim],
                  gmap.size_global, comm, mode=mode)
    write_ordered(path, name + "/topology", rows, cells, n_c, comm,
                  attrs={"celltype": np.bytes_(cellname), "partition": partition})
    write_ordered(path, name + "/cell_indices", rows,
                  np.asarray(mesh.topology.original_cell_index[:own], dtype=np.int64), n_c, comm)


def write_boundary_facets(path, mesh, values):
    """`path` (.xdmf) and its .h5 as femtools.boundary_markers.BoundaryMarkers.write
    writes a dolfin facet MeshFunction: the exterior facets with a nonzero marker
    (`values`, one per local facet), their vertices numbered as the mesh's original
    vertices. Collective; returns `(facets, points)` counts."""
    import dolfinx.mesh
    from femtools import boundaries, hdf5_rows
    comm = mesh.comm if mesh.comm.size > 1 else None
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    exterior = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    n_own_f = mesh.topology.index_map(tdim - 1).size_local
    values = np.asarray(values)
    keep = exterior[(exterior < n_own_f) & (values[exterior] != 0)].astype(np.int32)
    nodes = dolfinx.mesh.entities_to_geometry(mesh, tdim - 1, keep).reshape(len(keep), -1)
    gv = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
    used, inverse = np.unique(nodes, return_inverse=True)
    glob = gv[used]
    x = mesh.geometry.x[used, :mesh.geometry.dim]
    if comm is None:
        order = np.argsort(glob)
        points, index = x[order], np.empty(len(glob), np.int64)
        index[order] = np.arange(len(glob))
    else:
        # each vertex once, at the rank whose block of original indices holds it
        n_vertices = mesh.geometry.index_map().size_global
        edges = hdf5_rows.block_edges(n_vertices, comm.Get_size())
        routed = hdf5_rows.route(comm, hdf5_rows.owner_of(edges, glob, comm), glob, x)
        got, got_x = routed.received
        mine, first, where = np.unique(got, return_index=True, return_inverse=True)
        points = got_x[first]
        start = comm.exscan(len(mine)) or 0
        (index,) = hdf5_rows.reply(comm, routed, start + where.astype(np.int64))
    return boundaries.write(path, points, index[inverse].reshape(nodes.shape),
                            values[keep].astype(np.int64), comm)
