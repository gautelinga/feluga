"""A mesh and its functions in the layout dolfin's `HDF5File` writes, read without dolfin.

    reader = LayoutReader("mesh.h5", "up_0.h5")
    coords, conn10, values, names = reader.read(["u", "p"])  # P2 nodes, XDMF order
    vx, topo, p = reader.read_p1("p")                        # a P1 scalar at the vertices
    volumes, v_fluid, box = reader.label_volumes()           # by global cell id
    for vol, nodal in reader.cell_blocks("u", volumes): ...  # streamed, cell by cell

No function space is built: the stored dofmap (`cell_dofs`, `x_cell_dofs`, `cells`) and
the mesh topology are enough, so numpy and h5py are the whole dependency. Four
conventions it depends on, each measured, none assumed; the last is what a hand-written
permutation would most likely get wrong:

- `x_cell_dofs` has a uniform stride, so `cell_dofs` reshapes to a row per cell;
- a vector element's cell dofs are **component major** -- ten node dofs for u0, then ten
  for u1, then u2 -- not interleaved node by node;
- `cells` and `mesh/cell_indices` each label their rows by global cell id, and neither
  file stores them in that order, so the two have to be composed;
- dolfin numbers a tetrahedron's P2 nodes `v0 v1 v2 v3 e23 e13 e12 e03 e02 e01`, where
  XDMF wants the midsides in the order `e01 e12 e02 e03 e13 e23`.

And it checks itself where that matters: every cell containing a node must hand it the
same value, which fails loudly if any of the above is read wrongly. `read`'s output is
what `femtools.xdmf.write_quadratic` takes.
"""
import re

import numpy as np

from femtools.xdmf import DOLFIN_EDGES, XDMF_ORDER

#: cells `LayoutReader.label_volumes` and `cell_blocks` take at a time
CELL_CHUNK = 1 << 20


def _element(signature, per_cell):
    """`(degree, components, nodes per cell)` of a stored Lagrange field on tetrahedra, from
    its element signature and its dofs a cell."""
    degree = int(re.search(r'tetrahedron,\s*(\d)', signature).group(1))
    nodes_per_cell = 10 if degree == 2 else 4
    comps = per_cell // nodes_per_cell
    if comps * nodes_per_cell != per_cell:
        raise ValueError('{} dofs per cell does not divide into {} nodes'
                         .format(per_cell, nodes_per_cell))
    return degree, comps, nodes_per_cell


class LayoutReader(object):
    """A mesh file and a function file in dolfin's layout, read without dolfin.

    `mesh_path` holds `mesh/coordinates`, `topology`, `cell_indices`; `field_path`
    the functions, each a group (`vector_0`, `cell_dofs`, `x_cell_dofs`, `cells` and
    the element's `signature`). The composition of the two files' row labels through
    `mesh/cell_indices` lives here alone.
    """

    #: a dataset in a field's group that holds its values in vertex order already,
    #: which `read_p1` returns as they are (None: no such dataset is looked for)
    VERTEX_VALUES = None

    def __init__(self, mesh_path, field_path=None):
        self.mesh_path, self.field_path = mesh_path, field_path

    def read_mesh(self):
        """Vertices and tetrahedra, with each row's global cell id kept aside.

        Topology rows come in the writer's own order, labelled by global cell id in
        `mesh/cell_indices`; the function file labels its rows the same way and the
        two orders are not the same. Coordinates are indexed by global vertex id.
        Both halves of that are pinned by `tests/test_hierarchical_transfer.py`, across
        rank counts.
        """
        import h5py
        with h5py.File(self.mesh_path, 'r') as h5:
            vx = np.asarray(h5['mesh/coordinates'])
            topo = np.asarray(h5['mesh/topology'])
            self.cell_id = np.asarray(h5['mesh/cell_indices'])
        return vx, topo

    def read(self, fields):
        """Returns (node coordinates, XDMF connectivity, values, field names)."""
        vx, topo = self.read_mesh()
        n_vert, n_cells = len(vx), len(topo)

        # Number the nodes: the mesh vertices, then one per edge. Edges are collected
        # in dolfin's local order so a cell's ten nodes come out in dolfin's order.
        pairs = np.sort(topo[:, np.asarray(DOLFIN_EDGES)], axis=2)
        key = pairs[..., 0].astype(np.int64) * n_vert + pairs[..., 1]
        uniq, inv = np.unique(key.ravel(), return_inverse=True)
        ends = np.column_stack((uniq // n_vert, uniq % n_vert))
        coords = np.vstack((vx, 0.5 * (vx[ends[:, 0]] + vx[ends[:, 1]])))
        local = np.hstack((topo, inv.reshape(n_cells, 6) + n_vert))

        values, names = [], []
        for field in fields:
            values.append(self._gather(field, local, len(coords), ends, n_vert))
            names.append(field)
        return coords, local[:, XDMF_ORDER], values, names

    def cell_rows(self, field, n_cells):
        """The stored dofmap of `field`, one row per mesh row, and its vector.

        Both files label their rows by global cell id, and neither is in that order:
        composing the two is the whole subtlety of this route, so it lives here alone
        and the P2 and P1 readers cannot come to read the same file differently.

        Whole, for the P2 and P1 readers, which number the nodes over the whole mesh
        anyway; `cell_blocks` is the same composition streamed, for a caller that only
        needs each cell's values.
        """
        import h5py
        with h5py.File(self.field_path, 'r') as h5:
            group = h5[field]
            signature = group.attrs['signature'].decode()
            flat = np.asarray(group['cell_dofs'])
            offsets = np.asarray(group['x_cell_dofs'])
            cells = np.asarray(group['cells'])
            vec = np.asarray(group['vector_0'])

        stride = np.diff(offsets)
        if not np.all(stride == stride[0]):
            raise ValueError('cell_dofs is ragged; this route cannot reshape it')
        per_cell = int(stride[0])
        where = np.full(int(max(self.cell_id.max(), cells.max())) + 1, -1,
                        dtype=np.int64)
        where[self.cell_id] = np.arange(n_cells)
        if np.any(where[cells] < 0):
            raise ValueError('the function file names cells the mesh file does not')
        # every mesh row filled exactly once: a label named twice would leave another row
        # as whatever np.empty held, and the integral over it garbage
        if len(cells) != n_cells or len(np.unique(where[cells])) != n_cells:
            raise ValueError('the function file does not label each mesh cell exactly once')
        rows = np.empty((n_cells, per_cell), dtype=flat.dtype)
        rows[where[cells]] = flat.reshape(n_cells, per_cell)
        degree, comps, nodes_per_cell = _element(signature, per_cell)
        return rows, vec, degree, comps, nodes_per_cell

    def label_volumes(self, chunk=CELL_CHUNK):
        """`(volumes, v_fluid, box)`: every cell's volume indexed by its global cell id
        (`mesh/cell_indices`; NaN at an id no row carries), their sum and the coordinates'
        bounding box, the topology read `chunk` rows at a time. What `cell_blocks`
        pairs a function file's rows with, so several fields on one mesh read it once."""
        import h5py
        with h5py.File(self.mesh_path, 'r') as h5:
            vx = np.asarray(h5['mesh/coordinates'])
            topo, labels = h5['mesh/topology'], h5['mesh/cell_indices']
            n = topo.shape[0]
            top = max((int(np.max(labels[k:k + chunk])) for k in range(0, n, chunk)),
                      default=-1)
            volumes = np.full(top + 1, np.nan)
            v_fluid = 0.0
            for k in range(0, n, chunk):
                t = np.asarray(topo[k:k + chunk], dtype=np.int64)
                ids = np.asarray(labels[k:k + len(t)], dtype=np.int64)
                if (ids.min() < 0 or not np.isnan(volumes[ids]).all()
                        or len(np.unique(ids)) != len(ids)):
                    raise ValueError('{} labels a cell twice, or with a negative id'
                                     .format(self.mesh_path))
                x = vx[t]
                vol = np.abs(np.einsum('ij,ij->i', x[:, 1] - x[:, 0],
                                       np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))) / 6.0
                volumes[ids] = vol
                v_fluid += float(vol.sum())
        return volumes, v_fluid, vx.max(axis=0) - vx.min(axis=0)

    def cell_blocks(self, field, volumes, chunk=CELL_CHUNK):
        """`(volumes, values)` for the function file's cells, `chunk` at a time, with
        `volumes` from `label_volumes`: each cell's volume and its nodal values,
        (cells, components, nodes per cell) in dolfin's node order. The streamed
        `cell_rows`: the same composition through the global cell id and the same
        refusals, holding the vector and one flag a cell rather than the dofmap."""
        import h5py
        with h5py.File(self.field_path, 'r') as h5:
            group = h5[field]
            offsets, flat, cells = group['x_cell_dofs'], group['cell_dofs'], group['cells']
            per_cell = int(offsets[1] - offsets[0]) if offsets.shape[0] > 1 else 0
            _, comps, nodes = _element(group.attrs['signature'].decode(), per_cell)
            vec = np.asarray(group['vector_0'])
            seen = np.zeros(len(volumes), dtype=bool)
            for k in range(0, cells.shape[0], chunk):
                ids = np.asarray(cells[k:k + chunk], dtype=np.int64)
                if not np.array_equal(np.asarray(offsets[k:k + len(ids) + 1]),
                                      per_cell * np.arange(k, k + len(ids) + 1)):
                    raise ValueError('cell_dofs is ragged; this route cannot reshape it')
                if ids.min() < 0 or ids.max() >= len(volumes) or np.isnan(volumes[ids]).any():
                    raise ValueError('the function file names cells the mesh file does not')
                if seen[ids].any() or len(np.unique(ids)) != len(ids):
                    raise ValueError('the function file does not label each mesh cell '
                                     'exactly once')
                seen[ids] = True
                dofs = np.asarray(flat[per_cell * k:per_cell * (k + len(ids))])
                yield volumes[ids], vec[dofs.reshape(len(ids), comps, nodes)]
        if np.count_nonzero(seen) != np.count_nonzero(~np.isnan(volumes)):
            raise ValueError('the function file does not label each mesh cell exactly once')

    def _scatter(self, field, out, cell_nodes, rows, vec, comps, nodes_per_cell):
        """Cell-local dofs onto the nodes they share, checking the cells agree."""
        for c in range(comps):
            block = rows[:, c * nodes_per_cell:(c + 1) * nodes_per_cell]
            take = vec[block]
            target = out[:, c] if comps > 1 else out
            target[cell_nodes] = take                  # duplicates: last write wins
            # ...so every other cell sharing a node must have written the same thing
            clash = float(np.abs(target[cell_nodes] - take).max())
            if clash > 0.:
                raise ValueError(
                    'cells disagree by {:.3e} about a shared node of "{}", so the '
                    'stored dofmap is not being read as dolfin wrote it'
                    .format(clash, field))
        return out

    def _gather(self, field, local, n_nodes, ends, n_vert):
        """One field's nodal values, through the cell dofmap stored beside it."""
        rows, vec, degree, comps, per = self.cell_rows(field, len(local))
        cell_nodes = local if degree == 2 else local[:, :4]
        out = np.zeros((n_nodes, comps)) if comps > 1 else np.zeros(n_nodes)
        self._scatter(field, out, cell_nodes, rows, vec, comps, per)
        if degree == 1:
            # P1 carries no midside nodes; linear interpolation onto them is exact
            out[n_vert:] = 0.5 * (out[ends[:, 0]] + out[ends[:, 1]])
        return out

    def read_p1(self, field):
        """A degree-1 scalar as vertex values: (coordinates, tetrahedra, values).

        `read` numbers the edge midsides too, which for a P1 field means a `np.unique`
        over six edges per cell to produce values it then fills in by interpolation;
        this stops at the vertices, for a caller that evaluates the field in a cell it
        already knows. `topo` rows are labelled by `self.cell_id`, as `read_mesh` says.
        A field whose group holds `VERTEX_VALUES` is those values already.
        """
        import h5py
        vx, topo = self.read_mesh()
        if self.VERTEX_VALUES is not None:
            with h5py.File(self.field_path, 'r') as h5:
                if self.VERTEX_VALUES in h5[field]:
                    out = np.asarray(h5[field][self.VERTEX_VALUES])
                    if len(out) != len(vx):
                        raise ValueError('{} has {} vertex values and {} {} vertices'.format(
                            self.field_path, len(out), self.mesh_path, len(vx)))
                    return vx, topo, out
        rows, vec, degree, comps, per = self.cell_rows(field, len(topo))
        if degree != 1:
            raise ValueError('"{}" is stored at degree {}, not 1'.format(field, degree))
        if comps != 1:
            raise ValueError('"{}" has {} components; read_p1 is for a scalar'
                             .format(field, comps))
        out = np.zeros(len(vx))
        self._scatter(field, out, topo, rows, vec, comps, per)
        return vx, topo, out
