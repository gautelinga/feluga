"""A P1 field stored as its vertex values, in the mesh file's vertex order.

    <field>/vertex_values    float64, one per row of the mesh file's mesh/coordinates
    <field>.attrs            layout = "vertex", n_vertices, n_cells

    write_function(path, f)  read_function(path, f)     # a P1 Function, either backend
    write_values(path, values, n_cells)                 # serial, numpy
    read_values(path, mesh_path)                        # serial, numpy; either layout

dolfin's own layout (`HDF5File.write(f, name)`) stores the dofmap beside the vector --
`cell_dofs`, `cells`, `x_cell_dofs` -- 33.5 bytes a cell, of which the values are 1.4: for
a P1 field on a mesh already on disk the dofmap only restates the topology. Here only the
values are kept, 24x smaller, bit for bit the same numbers, and a view is a node-centred
attribute over the mesh file. From stokes-beadpack (2026-09-29), whose refinement rungs'
`delta.h5` hold it.

The order is the one thing it rests on: dolfin gives a vertex read from a mesh file the
global index of its row in `mesh/coordinates`, through the plain read and through
`periodic_partition.read_mesh` alike, on any number of ranks; femtools' DOLFINx reader
keeps that row as the vertex's `geometry.input_global_indices` (femtools.dolfinx.mesh).
So writing is by that index (`hdf5_rows.write_ordered`: no rank holds the field), and
reading a fetch by it.

Every reader here also takes dolfin's layout: `read_values` through
`femtools.layout_reader`, `read_function` through `HDF5File.read` (legacy dolfin).
"""
import os

import numpy as np

from femtools.backend import is_dolfinx

LAYOUT = "vertex"
VALUES = "vertex_values"


def is_vertex_layout(path, field="delta"):
    import h5py
    with h5py.File(path, "r") as h5:
        return VALUES in h5[field]


def counts(path, field="delta"):
    """(n_cells, n_vertices) of the mesh the field lives on, from either layout."""
    import h5py
    with h5py.File(path, "r") as h5:
        g = h5[field]
        if VALUES in g:
            return int(g.attrs["n_cells"]), int(g.attrs["n_vertices"])
        return int(g["cells"].shape[0]), int(g["vector_0"].shape[0])


def mark_layout(group, n_vertices, n_cells):
    """The attributes that say a group holds a field by vertex (every writer's)."""
    group.attrs["layout"] = LAYOUT
    group.attrs["n_vertices"] = int(n_vertices)
    group.attrs["n_cells"] = int(n_cells)


def write_values(path, values, n_cells, field="delta"):
    """Vertex values, in mesh-file order, as `path`. Serial; written beside and renamed,
    so a reader never meets half a file."""
    import h5py
    from femtools.compress import put
    values = np.ascontiguousarray(values, dtype=np.float64)
    tmp = path + ".writing"
    try:
        with h5py.File(tmp, "w") as h5:
            g = h5.create_group(field)
            put(g, VALUES, values)
            mark_layout(g, len(values), n_cells)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_function(path, f, field="delta"):
    """A P1 Function, as `path` (legacy dolfin, or femtools.backend.dolfinx on a mesh
    read from a file: its vertices' file rows known). Collective.

    Each rank's vertex values go by global vertex index to the rank whose block of the
    dataset holds them, which writes that block (`hdf5_rows.write_ordered`: no rank
    holds the field; it refuses a vertex nobody covered). Written beside and renamed,
    so a reader never meets half a file.
    """
    if is_dolfinx(f):
        return _write_dolfinx(path, f, field)
    import dolfin as df
    V = f.function_space()
    mesh = V.mesh()
    comm = mesh.mpi_comm()
    gvi = np.asarray(mesh.topology().global_indices(0), dtype=np.int64)
    n_vertices = mesh.num_entities_global(0)
    if V.dim() == n_vertices and V.num_sub_spaces() == 0:
        # a plain P1 scalar: a vertex's value is its dof, and each vertex is owned by one
        # rank, so each sends its owned ones. `compute_vertex_values` evaluates cell by
        # cell instead -- 0.23 s of 0.26 on a 2M-cell rung -- and sends shared ones twice
        v2d = df.vertex_to_dof_map(V)
        local = f.vector().get_local()
        owned = (v2d >= 0) & (v2d < len(local))
        rows, values = gvi[owned], local[v2d[owned]]
    else:                               # a periodic space, say: dolfin's own evaluation
        rows, values = gvi, f.compute_vertex_values(mesh)
    n_cells = mesh.num_entities_global(mesh.topology().dim())
    if df.MPI.size(comm) == 1:
        full = np.full(n_vertices, np.nan)
        full[rows] = values
        if np.isnan(full).any():
            raise ValueError("writing {}: {} vertices have no value".format(
                path, int(np.isnan(full).sum())))
        write_values(path, full, n_cells, field)
        return
    _write_rows(path, field, rows, values, n_vertices, n_cells, comm)


def read_values(path, mesh_path, field="delta"):
    """The field's value at every row of the mesh file's coordinates. Serial, numpy."""
    import h5py
    with h5py.File(path, "r") as h5:
        if VALUES in h5[field]:
            return np.asarray(h5[field][VALUES])
    from femtools.layout_reader import LayoutReader
    return LayoutReader(mesh_path, path).read_p1(field)[2]


def read_function(path, f, field="delta"):
    """Into a P1 Function on a mesh read from the field's mesh file (legacy dolfin, or
    femtools.backend.dolfinx: a vertex-layout file only). Collective."""
    if is_dolfinx(f):
        return _read_dolfinx(path, f, field)
    import dolfin as df
    mesh = f.function_space().mesh()
    comm = mesh.mpi_comm()
    if not is_vertex_layout(path, field):
        with df.HDF5File(comm, path, "r") as h5:
            h5.read(f, field)
        return f
    from femtools.xdmf import scatter_vertex_values
    from femtools.hdf5_rows import fetch_rows
    n_cells, n_vertices = counts(path, field)
    if (n_cells, n_vertices) != (mesh.num_entities_global(mesh.topology().dim()),
                                 mesh.num_entities_global(0)):
        raise ValueError("{} holds a field on {} cells and {} vertices; the mesh has {} "
                         "and {}".format(path, n_cells, n_vertices,
                                         mesh.num_entities_global(mesh.topology().dim()),
                                         mesh.num_entities_global(0)))
    gvi = np.asarray(mesh.topology().global_indices(0), dtype=np.int64)
    return scatter_vertex_values(f, fetch_rows(path, field + "/" + VALUES, gvi, comm))


def _p1_nodes(V):
    """A DOLFINx P1 scalar space's local dofs and the geometry nodes they sit on, one
    pair a local cell's vertex (the P1 dofmap and the geometry's share a cell's vertex
    order); and the nodes' file rows."""
    mesh = V.mesh
    if V.dofmap.index_map_bs != 1 or V.num_sub_spaces or V.element.space_dimension != \
            mesh.geometry.dofmap.shape[1]:
        raise NotImplementedError("vertex_field: a P1 scalar space on an affine mesh")
    dofs = np.asarray(V.dofmap.list).ravel()
    nodes = np.asarray(mesh.geometry.dofmap).ravel()
    rows = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
    return dofs, nodes, rows


def _counts_dolfinx(mesh):
    return (int(mesh.topology.index_map(mesh.topology.dim).size_global),
            int(mesh.geometry.index_map().size_global))


def _write_rows(path, field, rows, values, n_vertices, n_cells, comm):
    """Both backends' write: each value to the file row `rows` gives it
    (`hdf5_rows.write_ordered`, which refuses a vertex nobody covered), written beside
    `path` and renamed by rank 0 once the layout is marked; an error there raised on
    every rank. Collective."""
    from femtools.hdf5_rows import write_ordered
    tmp = path + ".writing"
    write_ordered(tmp, field + "/" + VALUES, rows, np.asarray(values, np.float64),
                  n_vertices, comm if comm.size > 1 else None, mode="w")
    err = None
    if comm.rank == 0:
        try:
            import h5py
            with h5py.File(tmp, "a") as h5:
                mark_layout(h5[field], n_vertices, n_cells)
            os.replace(tmp, path)
        except Exception as e:                             # noqa: BLE001 -- shared below
            err = "{}: {}".format(type(e).__name__, e)
    err = comm.bcast(err, root=0)
    if err:
        raise RuntimeError("writing {}: {}".format(path, err))


def _write_dolfinx(path, f, field):
    V = f.function_space
    f.sync()
    dofs, nodes, rows = _p1_nodes(V)
    # each node once (it is in ~20 cells), and from the rank that owns it
    nodes, first = np.unique(nodes, return_index=True)
    own = nodes < V.mesh.geometry.index_map().size_local
    n_cells, n_vertices = _counts_dolfinx(V.mesh)
    comm = V.mesh.comm
    vals = np.asarray(f.x.array, dtype=np.float64)[dofs[first[own]]]
    _write_rows(path, field, rows[nodes[own]], vals, n_vertices, n_cells, comm)


def _read_dolfinx(path, f, field):
    from femtools.hdf5_rows import fetch_rows
    V = f.function_space
    comm = V.mesh.comm
    if not is_vertex_layout(path, field):
        raise NotImplementedError("{}: dolfin's own layout on DOLFINx (read it with "
                                  "femtools.dolfin_layout.read_function)".format(path))
    have = _counts_dolfinx(V.mesh)
    if counts(path, field) != have:
        raise ValueError("{} holds a field on {} cells and {} vertices; the mesh has {} "
                         "and {}".format(path, *counts(path, field), *have))
    dofs, nodes, rows = _p1_nodes(V)
    n_nodes = len(rows)
    values = fetch_rows(path, field + "/" + VALUES, rows, comm if comm.size > 1 else None)
    at = np.full(n_nodes, -1, dtype=np.int64)
    at[nodes] = dofs
    f.x.array[at[at >= 0]] = values[at >= 0]
    return f.restrict()
