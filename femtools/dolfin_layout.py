"""A dolfin Mesh or Function written in the layout `dolfin.HDF5File.write` gives it,
compressed (`compress`; dolfin's own writer has no compression).

    write_mesh(path, mesh)                 # <name>/coordinates, topology, cell_indices
    write_function(path, u, "u")           # <name>/vector_0, cell_dofs, x_cell_dofs, cells
    write_dofs(path, V, "V")               # <name>/cell_dofs, x_cell_dofs, cells, once
    write_function(path, u, "u", dofs=("tables.h5", "V"))   # vector_0, the tables linked
    read_function(path, u, "u")            # any number of ranks, either backend's file
    write_mesh_arrays(path, x, cells)      # the mesh from arrays, serially, without dolfin
    write_mesh_rows(path, name, comm, ...) # a mesh's datasets from either backend's arrays

The same datasets, dtypes, values and attributes as dolfin writes on the same partition
(`tests/test_dolfin_layout.py` holds the two to every bit on 1 and 3 ranks), so every
reader -- dolfin's `HDF5File.read`, partrac, h5py -- reads either. What dolfin writes, and
so this:

- `coordinates` by global vertex index; `topology` each rank's own cells in local order,
  rank after rank, as global vertex indices, with `celltype` and `partition` (each rank's
  first row); `cell_indices` those cells' global ids.
- `x_cell_dofs` (uint64) each own cell's first entry in `cell_dofs` (the dofmap's
  integer type) plus the total; `cells` (uint64) the cells' global ids; `vector_0` the
  vector's local ranges with `partition` (each rank's first row); the group's `signature`,
  the element's.

Every dataset goes through `hdf5_rows.write_ordered`: no rank holds one whole, and every
rank writes its own block (MPI-IO). dolfin and mpi4py are imported where they are used.

One writer serves both backends: `write_dofs_of` and `write_values_of` take a space's
`SpaceLayout` (its communicator, signature, cell count, and on demand its dof tables and
their fingerprint); `write_dofs`/`write_function` here and femtools.dolfinx.checkpoint's
make one and call them.

The dof tables are most of a Function's file, and the same for every Function on a space
until the mesh is partitioned anew; `write_dofs` writes them once, and
`write_function(..., dofs=...)` links them (an HDF5 external link a reader follows) next
to each vector -- a checkpoint then costs its vectors alone.
"""
import numpy as np

from femtools.hdf5_rows import write_ordered


def _first(comm, n):
    """This rank's first row for n rows a rank, rank order: one buffer Exscan."""
    from mpi4py import MPI
    before = np.zeros(1, dtype=np.int64)
    comm.Exscan(np.array([int(n)], dtype=np.int64), before, op=MPI.SUM)
    return 0 if comm.Get_rank() == 0 else int(before[0])


def _total(comm, n):
    """n summed over the ranks: one buffer Allreduce."""
    from mpi4py import MPI
    out = np.zeros(1, dtype=np.int64)
    comm.Allreduce(np.array([int(n)], dtype=np.int64), out, op=MPI.SUM)
    return int(out[0])


def _offsets(comm, n):
    """(this rank's first row, every rank's first row) for n rows a rank, rank order: the
    second is dolfin's `partition` attribute, which every rank writes alike."""
    first = _first(comm, n)
    every = np.empty(comm.Get_size(), dtype=np.int64)
    comm.Allgather(np.array([first], dtype=np.int64), every)
    return first, every.astype(np.uint64)


def write_mesh_rows(path, name, comm, vertex_rows, x, n_vertices, cells, cell_ids, n_cells,
                    cellname, mode="w"):
    """A mesh's datasets in the group `name`: `coordinates` (rows `vertex_rows` of
    `n_vertices`, values `x`), and this rank's own `cells` (a row a cell, as global vertex
    indices) as `topology` (with `celltype` and `partition`) and their global ids
    `cell_ids` as `cell_indices`, rank after rank in `n_cells` rows. Collective."""
    first, partition = _offsets(comm, len(cells))
    rows = first + np.arange(len(cells), dtype=np.int64)
    write_ordered(path, name + "/coordinates", vertex_rows, x, n_vertices, comm, mode=mode)
    write_ordered(path, name + "/topology", rows, cells, n_cells, comm,
                  attrs={"celltype": np.bytes_(cellname), "partition": partition})
    write_ordered(path, name + "/cell_indices", rows, cell_ids, n_cells, comm)


def write_mesh(path, mesh, name="mesh", mode="w"):
    """The mesh as `HDF5File(path, mode).write(mesh, name)` writes it. Collective."""
    tdim = mesh.topology().dim()
    own = mesh.topology().ghost_offset(tdim)
    gv = np.asarray(mesh.topology().global_indices(0), dtype=np.int64)
    gc = np.asarray(mesh.topology().global_indices(tdim), dtype=np.int64)[:own]
    write_mesh_rows(path, name, mesh.mpi_comm(), gv, np.asarray(mesh.coordinates()),
                    mesh.num_entities_global(0), gv[np.asarray(mesh.cells())[:own]], gc,
                    mesh.num_entities_global(tdim), mesh.ufl_cell().cellname(), mode)


def _dof_tables(V):
    """`(gc, dofs)`: this rank's own cells' global ids (uint64), and their global dofs, a
    row a cell, in the dofmap's integer type (femtools.tables)."""
    from femtools.tables import space_tables
    t = space_tables(V)
    return t.gc, t.dofs


def read_function(path, u, name):
    """u from the Function `name` in path (dolfin's layout, by either backend, on any
    number of ranks): femtools.tables.read_values, the reader DOLFINx's checkpoints use
    too. Collective."""
    from femtools.tables import read_values, space_tables
    x = u.vector()
    x.set_local(read_values(path, name, space_tables(u.function_space())))
    x.apply("insert")
    return u


def _mix(x):
    """splitmix64's finalizer, elementwise on uint64 (wrapping)."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _fingerprint(comm, gc, dofs):
    """A 64-bit hash of which global dofs each cell (by global id) has, in which local
    position: a sum over the cells, so the same on any partition of the same numbering,
    and another for another numbering (another number of ranks, or partition)."""
    n = dofs.shape[1] if dofs.ndim == 2 else 0
    key = gc[:, None] * np.uint64(n + 1) + np.arange(n, dtype=np.uint64)
    from mpi4py import MPI
    mine = np.array([_mix(_mix(key) ^ dofs.astype(np.uint64)).sum(dtype=np.uint64)],
                    dtype=np.uint64)
    total = np.zeros(1, dtype=np.uint64)
    comm.Allreduce(mine, total, op=MPI.SUM)        # unsigned: wraps modulo 2^64
    return np.uint64(total[0])


def _write_tables(path, name, comm, n_c, gc, dofs, mode):
    """`x_cell_dofs`, `cell_dofs` and `cells` of the group `name`, as dolfin writes them."""
    own, n = dofs.shape if dofs.ndim == 2 else (0, 0)
    local = dofs.reshape(-1)
    lengths = np.full(own, n, dtype=np.uint64)
    first = _first(comm, own)
    dof_first = _first(comm, len(local))
    n_dofs = _total(comm, len(local))
    x_rows = first + np.arange(own, dtype=np.int64)
    # in uint64 throughout: an int64 [0] beside a uint64 sum promotes to float64
    x_vals = np.uint64(dof_first) + np.concatenate(
        [np.zeros(1, np.uint64), np.cumsum(lengths, dtype=np.uint64)[:-1]])[:own]
    if comm.Get_rank() == comm.Get_size() - 1:      # the total, by the last rank
        x_rows = np.r_[x_rows, n_c]
        x_vals = np.r_[x_vals, np.uint64(n_dofs)]
    write_ordered(path, name + "/x_cell_dofs", x_rows, x_vals, n_c + 1, comm, mode=mode)
    write_ordered(path, name + "/cell_dofs", dof_first + np.arange(len(local)), local,
                  n_dofs, comm)
    write_ordered(path, name + "/cells", x_rows[:own], gc, n_c, comm)


class SpaceLayout:
    """What writing a space's Functions in dolfin's layout takes of either backend's
    space, so that one writer serves both (`write_dofs_of`, `write_values_of`; dolfin's
    `write_dofs`/`write_function` here and femtools.dolfinx.checkpoint's wrap them):

    - `comm` (mpi4py), `signature` (dolfin's of the element), `n_cells` (over all ranks);
    - `tables()`: `(gc, dofs)`, this rank's own cells' global ids (uint64) and global
      dofs (a row a cell, the dofmap's integer type) -- asked only when written, and
      once per SpaceLayout: a Function written with linked tables needs none;
    - `fingerprint()`: the numbering's fingerprint, through `cache(compute)`, the
      backend's store of it per space (it costs a pass over every cell's dofs)."""

    def __init__(self, comm, signature, n_cells, tables, cache):
        self.comm, self.signature, self.n_cells = comm, signature, int(n_cells)
        self._build, self._cache, self._tables = tables, cache, None

    def tables(self):
        if self._tables is None:
            self._tables = self._build()
        return self._tables

    def fingerprint(self):
        return self._cache(lambda: _fingerprint(self.comm, *self.tables()))


def write_dofs_of(path, layout, name="dofs", mode="w"):
    """The dof tables of a space (its SpaceLayout) alone -- `x_cell_dofs`, `cell_dofs` and
    `cells`, as `write_values_of` writes them -- with the element's `signature` and the
    numbering's `fingerprint` on the group. Collective."""
    from femtools.hdf5_rows import opened
    gc, dofs = layout.tables()
    _write_tables(path, name, layout.comm, layout.n_cells, gc, dofs, mode)
    fingerprint = layout.fingerprint()
    with opened(path, "a", layout.comm) as h5:
        if h5 is not None:
            h5[name].attrs["signature"] = np.bytes_(layout.signature)
            h5[name].attrs["fingerprint"] = fingerprint


def _check_linked(path, dofs, layout):
    """Why the tables `dofs = (file, group)` are not the layout's space's on this
    partition, or None."""
    import os
    import h5py
    comm = layout.comm
    target, group = dofs
    where = os.path.join(os.path.dirname(os.path.abspath(path)), target)
    found = None
    if comm.Get_rank() == 0:
        try:
            with h5py.File(where, "r") as h5:
                a = h5[group].attrs
                found = (bytes(a["signature"]), np.uint64(a["fingerprint"]))
        except (OSError, KeyError) as e:
            found = str(e)
    found = comm.bcast(found, root=0)
    fingerprint = layout.fingerprint()
    if os.path.realpath(where) == os.path.realpath(path):
        return ("{}: the dof tables must be in another file than the Function's, which "
                "mode 'w' would truncate".format(where))
    if isinstance(found, str):
        return "{}:{}: no dof tables ({})".format(where, group, found)
    if found[0] != np.bytes_(layout.signature):
        return "{}:{}: the dof tables of another element".format(where, group)
    if found[1] != fingerprint:
        return ("{}:{}: the dof tables of another numbering (another number of ranks, or "
                "partition); write this one's with write_dofs".format(where, group))
    return None


def write_values_of(path, layout, name, values, lo, n_global, mode="a", dofs=None):
    """A Function -- this rank's owned vector entries `values`, the first being entry
    `lo` of `n_global` -- on the space of `layout`, as dolfin's `HDF5File(path,
    mode).write(u, name)` writes one. Collective.

    With `dofs = (file, group)`, the dof tables `write_dofs_of` wrote for the space on
    this partition (`file` relative to `path`'s directory, as HDF5 resolves it) are
    linked into the group (external links) rather than written again: dolfin, h5py and
    every reader follow the links, so the group reads as before while only `vector_0` is
    written. The tables are checked to be the space's, numbered as now (refused on every
    rank)."""
    import h5py
    from femtools.hdf5_rows import agreed, opened
    comm = layout.comm
    values = np.asarray(values)
    _, partition = _offsets(comm, len(values))
    if dofs is None:
        gc, tables = layout.tables()
        _write_tables(path, name, comm, layout.n_cells, gc, tables, mode)
        mode = "a"
    else:
        bad = _check_linked(path, dofs, layout)
        if agreed(comm, bad is not None):
            raise ValueError(bad or "another rank refused the dof tables")
    write_ordered(path, name + "/vector_0", lo + np.arange(len(values), dtype=np.int64),
                  values, n_global, comm, mode=mode, attrs={"partition": partition})
    with opened(path, "a", comm) as h5:
        if h5 is not None:
            if dofs is not None:
                for k in ("x_cell_dofs", "cell_dofs", "cells"):
                    h5[name + "/" + k] = h5py.ExternalLink(dofs[0], dofs[1] + "/" + k)
            h5[name].attrs["signature"] = np.bytes_(layout.signature)


#: each dolfin space's fingerprint, by the space's id (dolfin never reuses one in a
#: process): a space's numbering is fixed
_FINGERPRINTS = {}


def _layout(V):
    """V's SpaceLayout (dolfin)."""
    mesh = V.mesh()
    key = V.id()

    def cache(compute):
        if key not in _FINGERPRINTS:
            _FINGERPRINTS[key] = compute()
        return _FINGERPRINTS[key]

    return SpaceLayout(mesh.mpi_comm(), V.element().signature(),
                       mesh.num_entities_global(mesh.topology().dim()),
                       lambda: _dof_tables(V), cache)


def write_dofs(path, V, name="dofs", mode="w"):
    """The dof tables of V alone -- `x_cell_dofs`, `cell_dofs` and `cells`, as
    `write_function` writes them -- with the element's `signature` and the numbering's
    `fingerprint` on the group. Written once for a space and a partition, they serve every
    `write_function(..., dofs=(path, name))` of a Function on it. Collective."""
    write_dofs_of(path, _layout(V), name, mode)


def write_function(path, u, name, mode="a", dofs=None):
    """The Function as `HDF5File(path, mode).write(u, name)` writes it, or with `dofs =
    (file, group)` its tables linked to those `write_dofs` wrote (write_values_of).
    Collective."""
    vec = u.vector()
    lo, _ = vec.local_range()
    write_values_of(path, _layout(u.function_space()), name, vec.get_local(), lo,
                    vec.size(), mode=mode, dofs=dofs)


def write_mesh_arrays(path, x, cells, name="mesh", mode="w"):
    """A serial mesh from its arrays, as dolfin writes one on one rank: `x` the vertex
    coordinates, `cells` (triangles or tetrahedra) indices into them."""
    import h5py
    from femtools.compress import put
    x = np.ascontiguousarray(x, dtype=np.float64)
    cells = np.ascontiguousarray(cells, dtype=np.int64)
    with h5py.File(path, mode) as h5:
        put(h5, name + "/coordinates", x)
        t = put(h5, name + "/topology", cells)
        t.attrs["celltype"] = np.bytes_("triangle" if cells.shape[1] == 3 else "tetrahedron")
        t.attrs["partition"] = np.zeros(1, dtype=np.uint64)
        put(h5, name + "/cell_indices", np.arange(len(cells), dtype=np.int64))
