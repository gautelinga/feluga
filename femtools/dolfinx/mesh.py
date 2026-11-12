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
