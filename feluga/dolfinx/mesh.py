"""DOLFINx meshes from the HDF5 files dolfin writes (`HDF5File.write(mesh, name)`,
feluga.dolfin_layout.write_mesh, meshtools' generators): `name/coordinates` (a vertex a
row, by global vertex index), `name/topology` (a cell a row, as global vertex indices,
with the `celltype` attribute) and `name/cell_indices` (each row's global cell id).

    mesh = read_mesh("mesh.h5", comm)

The mesh's numbering is the file's: a cell's original index (`topology.original_cell_index`)
is its global id in the file, a vertex's original index
(`geometry.input_global_indices`) its row -- which is how dolfin numbers the same file,
so a checkpoint names the same cells on either backend (feluga.dolfinx.checkpoint).
Every rank reads its block of the cells, in order of their global ids, and its block of
the coordinates; nothing is read whole. Affine triangles and tetrahedra.
"""
import numpy as np


def _block(n, comm):
    r, size = comm.rank, comm.size
    return n * r // size, n * (r + 1) // size


def read_mesh(path, comm, name="mesh"):
    """The mesh `name` in the HDF5 file `path` (dolfin's layout), or the mesh of an XDMF
    file `path` (`*.xdmf`: the meshers', through meshio; a cell's id is its row, as dolfin
    numbers it), distributed over `comm`. Collective."""
    import h5py
    import basix.ufl
    import dolfinx.mesh
    from feluga.hdf5_rows import _mesh_datasets, _xdmf_celltype, fetch_rows, rows_of_labels

    if path.endswith(".xdmf"):
        h5path, topo_name, coords_name, ids_name = _mesh_datasets(path)
        celltype = _xdmf_celltype(path)
    else:
        h5path = path
        topo_name, coords_name, ids_name = (name + "/topology", name + "/coordinates",
                                            name + "/cell_indices")
        celltype = None
    with h5py.File(h5path, "r") as h5:
        n_cells = int(h5[topo_name].shape[0])
        n_x, gdim = (int(v) for v in h5[coords_name].shape)
        if celltype is None:
            celltype = h5[topo_name].attrs["celltype"]
            celltype = celltype.decode() if isinstance(celltype, bytes) else str(celltype)
        has_ids = ids_name is not None and ids_name in h5
        lo, hi = _block(n_x, comm)
        x = np.asarray(h5[coords_name][lo:hi], dtype=np.float64)
        if not has_ids:            # a cell's id is its row: this rank's block, read here
            lo, hi = _block(n_cells, comm)
            cells = np.asarray(h5[topo_name][lo:hi], dtype=np.int64)
    if celltype not in ("triangle", "tetrahedron"):
        raise NotImplementedError("read_mesh: triangles and tetrahedra, not " + celltype)
    if has_ids:
        lo, hi = _block(n_cells, comm)
        ids = np.arange(lo, hi, dtype=np.int64)
        rows = rows_of_labels(h5path, ids_name, ids, comm if comm.size > 1 else None)
        cells = fetch_rows(h5path, topo_name, rows, comm if comm.size > 1 else None)
    element = basix.ufl.element("Lagrange", celltype, 1, shape=(gdim,))
    return dolfinx.mesh.create_mesh(comm, np.asarray(cells, dtype=np.int64), element, x)


def write_mesh(path, mesh, name="mesh", mode="w"):
    """The mesh as dolfin's `HDF5File(path, mode).write(mesh, name)` writes one
    (feluga.dolfin_layout.write_mesh): `coordinates` by original vertex index,
    `topology` the own cells rank after rank (each cell's vertices in increasing order,
    as dolfin orders them), `cell_indices` their original indices. Collective."""
    from feluga.dolfin_layout import write_mesh_rows
    tdim, gdim = mesh.topology.dim, mesh.geometry.dim
    gmap = mesh.geometry.index_map()
    gv = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
    n_own_x = gmap.size_local
    cmap = mesh.topology.index_map(tdim)
    own = cmap.size_local
    write_mesh_rows(path, name, mesh.comm, gv[:n_own_x], mesh.geometry.x[:n_own_x, :gdim],
                    gmap.size_global, np.sort(gv[mesh.geometry.dofmaps[0][:own]], axis=1),
                    np.asarray(mesh.topology.original_cell_index[:own], dtype=np.int64),
                    cmap.size_global, {2: "triangle", 3: "tetrahedron"}[tdim], mode)


def write_boundary_facets(path, mesh, values):
    """`path` (.xdmf) and its .h5 as feluga.boundary_markers.BoundaryMarkers.write
    writes a dolfin facet MeshFunction: the exterior facets with a nonzero marker
    (`values`, one per local facet), their vertices numbered as the mesh's original
    vertices. Collective; returns `(facets, points)` counts."""
    import dolfinx.mesh
    from feluga import boundaries
    from feluga.dolfinx.boundary_markers import local_facets
    comm = mesh.comm if mesh.comm.size > 1 else None
    tdim = mesh.topology.dim
    _, exterior = local_facets(mesh)
    n_own_f = mesh.topology.index_map(tdim - 1).size_local
    values = np.asarray(values)
    keep = exterior[(exterior < n_own_f) & (values[exterior] != 0)].astype(np.int32)
    nodes = dolfinx.mesh.entities_to_geometry(mesh, tdim - 1, keep).reshape(len(keep), tdim)
    return boundaries.write_exterior(
        path, nodes, mesh.geometry.input_global_indices, mesh.geometry.x[:, :mesh.geometry.dim],
        values[keep], mesh.geometry.index_map().size_global, comm)
