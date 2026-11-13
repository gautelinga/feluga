"""`XDMFFile2`: a drop-in for legacy dolfin's `XDMFFile` that keeps a quadratic field.

    with XDMFFile2(comm, 'u.xdmf') as f:
        f.write((u, p))

dolfin's XDMFFile interpolates a quadratic function down to one value per vertex,
silently, so the picture loses everything the midside nodes carry. XDMF can hold
quadratic tetrahedra (`Tetrahedron_10`) and quadratic triangles (`Triangle_6`),
which ParaView renders once "Nonlinear Subdivision Level" is turned up; this writes
those. Their nodes are the mesh's vertices, then one per edge at its midpoint -- the
nodes of the unconstrained P2 space, so a periodic field is written on the whole
mesh, both sides of the period included, each node with the field's value there.
That value is read off the field's own dofmap, the dof its space gives the vertex or
edge (a slave's is its master's): writing builds no other space and interpolates
nothing.

A grid's coordinates are the mesh's: `GeometryType="XY"`, two columns, for a 2D mesh and
`XYZ` for a 3D one, as dolfin writes them, so a 2D mesh reads back 2D. A 2D vector is
padded to three components, as dolfin pads it (XDMF's Vector has three).

## What `XDMFFile2` covers

It is a drop-in for `dolfin.XDMFFile`, and it writes the *same file layout* legacy
dolfin writes -- so dolfin reads back what it produces, and a reader that copes with one
copes with the other:

    write(Function)                write(Function, t)
    write(Mesh)                    write(MeshFunction)      write(MeshValueCollection)
    write(points)                  write(points, values)
    read(Mesh)                     read(MeshFunction)       read(MeshValueCollection)
    write_checkpoint / read_checkpoint
    close, rename, name, label, id, parameters, Encoding

It differs from `dolfin.XDMFFile` in exactly four ways, all of them deliberate:

- **A degree-2 Function keeps its midside nodes.** dolfin throws them away without
  saying so; this writes `Tetrahedron_10` or `Triangle_6`.
- **`write((u, p))` writes several functions onto one node set.** dolfin takes one
  function per call; several sharing a grid are then certainly registered against
  each other, and the mesh is stored once instead of once per field.
- **A MeshFunction and a Function can share a file.** `dolfin.XDMFFile` *segfaults*
  on that combination -- `write(mf)` followed by `write(u)` takes the process down,
  in that order only (`tests/test_xdmf.py`).
- **A MeshFunction can share a *grid*, as cell data.** `write((u, p, tags))` writes
  the MeshFunction as a `Center="Cell"` attribute on the one grid the fields are on,
  which is what a value per cell is; dolfin has no call for this and rejects a tuple
  outright. It has to be over the same mesh at its top dimension, and is refused
  otherwise -- a facet MeshFunction against the cells writes numbers belonging to
  other entities onto a grid that looks perfectly well formed.

The three `parameters` are dolfin's, and they mean what dolfin means by them:
`flush_output` (write as you go rather than at close), `functions_share_mesh`
(successive functions attach to the grid already written rather than starting a new
one) and `rewrite_function_mesh` (store the mesh again for each time step, or point
every step at the one copy). Like dolfin's, the mapping refuses a key it does not
know rather than silently ignoring it.

`read_checkpoint` reads dolfin's own checkpoints -- which already round-trip a P2
function exactly, so those are handed straight to dolfin -- *and* the quadratic
visualisation grids written here, which dolfin cannot read at all. That reconstruction
goes via an unconstrained space and then interpolates onto whatever space the caller's
Function lives in, so it is safe for a periodic space, where matching dof coordinates
is not.

Every dataset is shuffled and gzipped (`femtools.compress`), and written from every
rank's own rows (`hdf5_rows.write_rows`): nothing is gathered. With h5py's MPI build
every rank writes its blocks through MPI-IO; with the serial build rank zero writes the
blocks the others compress. `write_quadratic` and `write_quadratic_blocks` write the
same layout from plain arrays, the second from ranks' shares.

dolfin, mpi4py and scipy are imported only inside the functions that need them:
`XdmfDocument`, `write_quadratic` and the readers need numpy and h5py alone.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from femtools import compress
from femtools.fast_interpolation import interpolate as fast_interpolate
from femtools.ufc import UFC_EDGES

# XDMF's Tetrahedron_10: four corners, then the midsides of these edges in order.
XDMF_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
# dolfin's own P2 node order: four corners, then the midsides of these.
DOLFIN_EDGES = UFC_EDGES[3]
# XDMF slot -> dolfin local node, corners first
XDMF_ORDER = [0, 1, 2, 3] + [4 + DOLFIN_EDGES.index(e) for e in XDMF_EDGES]

# The same three for Triangle_6: three corners, then the midsides of (0, 1), (1, 2)
# and (2, 0) in XDMF's order (VTK's), and of the edges opposite corners 0, 1, 2 in
# dolfin's -- its local edge i is the one not touching vertex i, as on a tetrahedron.
XDMF_TRIANGLE_EDGES = ((0, 1), (1, 2), (0, 2))
DOLFIN_TRIANGLE_EDGES = ((1, 2), (0, 2), (0, 1))
XDMF_TRIANGLE_ORDER = [0, 1, 2] + [3 + DOLFIN_TRIANGLE_EDGES.index(e)
                                   for e in XDMF_TRIANGLE_EDGES]

# cell name -> (corners, XDMF's midside edges, XDMF slot -> dolfin local node), for
# the cells a degree-2 Function is written on
QUADRATIC = {
    'tetrahedron': (4, XDMF_EDGES, XDMF_ORDER),
    'triangle': (3, XDMF_TRIANGLE_EDGES, XDMF_TRIANGLE_ORDER),
}

XI = 'http://www.w3.org/2001/XInclude'
DOCTYPE = '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>'

# (cell name, nodes per cell) -> XDMF TopologyType. Only the simplices dolfin meshes
# here are made of, plus the quadratic forms XDMF can hold.
XDMF_TOPOLOGY = {
    ('point', 1): 'Polyvertex',
    ('interval', 2): 'Polyline',
    ('triangle', 3): 'Triangle',
    ('tetrahedron', 4): 'Tetrahedron',
    ('interval', 3): 'Edge_3',
    ('triangle', 6): 'Triangle_6',
    ('tetrahedron', 10): 'Tetrahedron_10',
}
# and back, for reading
TOPOLOGY_CELL = {v: k for k, v in XDMF_TOPOLOGY.items()}
CELL_DIM = {'point': 0, 'interval': 1, 'triangle': 2, 'tetrahedron': 3}
# meshio's names, which `write_quadratic` still takes for its callers' sake
MESHIO_TOPOLOGY = {'tetra': 'Tetrahedron', 'tetra10': 'Tetrahedron_10',
                   'triangle': 'Triangle', 'triangle6': 'Triangle_6',
                   'line': 'Polyline', 'vertex': 'Polyvertex'}


class Encoding(object):
    """`XDMFFile.Encoding` -- the arrays go to HDF5 beside the XML.

    dolfin also has an ASCII encoding, inlining the arrays into the XML. It is
    refused rather than quietly written as HDF5.
    """
    HDF5 = 'HDF5'

    @staticmethod
    def of(value):
        """dolfin's enum, this one, or the string; anything else is an error."""
        if value is None:
            return Encoding.HDF5
        name = str(getattr(value, 'name', value)).upper()
        if name == 'ASCII':
            raise ValueError('ASCII encoding is not supported; '
                             'the arrays go to HDF5')
        if name != 'HDF5':
            raise ValueError('{!r} is not an XDMF encoding'.format(value))
        return name


class XDMFParameters(object):
    """dolfin's three parameters, and its habit of refusing a key it does not know.

    A plain dict would accept `flush_ouput` and quietly do nothing with it, which is
    the class of silence this whole module exists to remove.
    """

    _DEFAULTS = (('flush_output', False),
                 ('functions_share_mesh', False),
                 ('rewrite_function_mesh', True))

    def __init__(self):
        self._values = dict(self._DEFAULTS)

    def __getitem__(self, key):
        self._check(key)
        return self._values[key]

    def __setitem__(self, key, value):
        self._check(key)
        self._values[key] = bool(value)

    def __contains__(self, key):
        return key in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()

    def values(self):
        return self._values.values()

    def update(self, other):
        for key, value in dict(other).items():
            self[key] = value

    def _check(self, key):
        if key not in self._values:
            raise KeyError(
                '{!r} is not an XDMF parameter; dolfin has {}'.format(
                    key, ', '.join(sorted(self._values))))

    def str(self, verbose=False):
        return '\n'.join('{:<22} | {}'.format(k, v)
                         for k, v in sorted(self._values.items()))

    def __repr__(self):
        return 'XDMFParameters({})'.format(self._values)


#: how much of a dolfin checkpoint's file is uncompressed before a close rewrites it
#: (`XDMFFile2.close`, `compress.rewrite`'s `share`)
CHECKPOINT_SHARE = 0.5


class XdmfDocument(object):
    """The XDMF XML and its HDF5 sidecar, in the layout legacy dolfin writes.

    Deliberately free of dolfin: the standalone route has no dolfin at all, and the
    two producers must not drift into writing different files for the same data.
    Everything dolfin-shaped -- gathering a distributed mesh, tabulating a dofmap --
    happens in `XDMFFile2` above this and arrives here as plain arrays.
    """

    def __init__(self, path, comm=None):
        self.path = path
        stem = os.path.splitext(os.path.basename(path))[0]
        self.h5_name = stem + '.h5'
        self.h5_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                    self.h5_name)
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0
        self.size = comm.Get_size() if comm is not None else 1
        # Whether the HDF5 sidecar is opened by every rank and written through MPI-IO
        # ('mpio'), or by rank zero writing the blocks the others compress ('serial').
        # Recorded rather than inferred, so a test can insist which path ran.
        self.driver = None
        self._parallel = None       # decided once; it cannot change under us
        self._h5 = None
        self.root = ET.Element('Xdmf', {'Version': '3.0', 'xmlns:xi': XI})
        self.domain = ET.SubElement(self.root, 'Domain')
        self.series = None          # the temporal collection, once a time is given
        self._counts = {}
        self._groups = set()        # HDF5 groups already spoken for
        # The XML is saved after every write, so it is serialised a grid at a time
        # (`save`): every element maps to the grid at the top of the document it is
        # part of (a child of the Domain or a time step), whose text is kept until
        # something is added to it.
        self._top = {}              # element -> its top-level grid
        self._text = {}             # top-level grid -> its XML, while unchanged
        self._changed = set()       # top-level grids added to since the last save
        self._step = {}             # time step -> its place in the series
        self._saved = None          # what the file on disk holds, for appending

    # -- HDF5 ------------------------------------------------------------------

    def _next(self, kind):
        n = self._counts.get(kind, 0)
        self._counts[kind] = n + 1
        return n

    def free_group(self, base):
        """An HDF5 group name not already spoken for.

        Every other writer here numbers its groups -- `Mesh/0`, `MeshFunction/1`,
        `VisualisationVector/2` -- and the points path did not, so a second
        `write(points)` deleted the first one's arrays (`dataset` recreates a group
        that exists) while the first grid's XML went on pointing at them.
        """
        name = base
        while name in self._groups:
            name = '{}/{}'.format(base, self._next(base))
        self._groups.add(name)
        return name

    def mesh_group(self, timed):
        """A free HDF5 group for one grid's mesh, named the way dolfin names it.

        dolfin calls the first untimed one `Mesh/mesh` and numbers the timed ones.
        The numbering matters here in a way it does not there: this file can hold a
        linear mesh grid and a quadratic function grid at once, and if the second
        took the first's group name it would replace the linear topology with the
        quadratic one and leave the mesh grid pointing at cells of the wrong size.
        """
        name = 'Mesh/mesh' if not timed else None
        while name is None or name in self._groups:
            name = 'Mesh/{}/mesh'.format(self._next('Mesh'))
        self._groups.add(name)
        return name

    def parallel(self):
        """Whether the HDF5 sidecar is written through MPI-IO, every rank its own blocks
        (`hdf5_rows.mpio`: more than one rank, and h5py's MPI build). Otherwise rank
        zero writes the blocks the other ranks compress and send it."""
        if self._parallel is None:
            from femtools.hdf5_rows import mpio
            self._parallel = mpio(self.comm)
        return self._parallel

    def _h5file(self):
        """The sidecar, opened once: by every rank under MPI-IO, else by rank zero alone
        (None on the others), as `hdf5_rows.write_rows` takes it."""
        if self._h5 is None and (self.parallel() or self.rank == 0):
            import h5py
            if self.parallel():
                self._h5 = h5py.File(self.h5_path, 'w', driver='mpio', comm=self.comm)
                self.driver = 'mpio'
            else:
                self._h5 = h5py.File(self.h5_path, 'w')
                self.driver = 'serial'
        return self._h5

    def rows(self, group, rows, values, n_global):
        """One global dataset: row `rows[i]` is `values[i]`, from every rank's rows
        (`hdf5_rows.write_rows`: moved to chunk-aligned blocks, each written by its
        rank). The file's rows are in global order, so the file is the same however many
        ranks wrote it, as `dolfin.XDMFFile`'s node coordinates are. Collective.

        The `partition` attribute is dolfin's, and is written so that dolfin's own
        reader finds a file of ours exactly as it finds one of its own. Returns the
        reference a DataItem points at."""
        from femtools.hdf5_rows import write_rows
        h5 = self._h5file()
        if h5 is not None and group in h5:
            del h5[group]
        write_rows(h5, group, rows, np.ascontiguousarray(values), int(n_global), self.comm,
                   attrs={'partition': np.array([0], dtype=np.uint64)})
        return '{}:/{}'.format(self.h5_name, group)

    def slab(self, group, offset, values, n_global):
        """`rows` for rows whose order in the file is arbitrary -- cells, and the entities
        a MeshFunction is defined over: each rank's next block, from `offset`.
        `dolfin.XDMFFile` reorders its cells between rank counts for the same reason."""
        values = np.asarray(values)
        return self.rows(group, int(offset) + np.arange(len(values)), values, n_global)

    def dataset(self, group, array):
        """`rows` for an array rank zero holds whole (the others pass nothing): points
        given alike to every rank, or a serial writer's arrays."""
        array = np.asarray(array)
        n = len(array) if self.comm is None else self.comm.bcast(len(array), root=0)
        mine = array if self.rank == 0 else array[:0]
        return self.slab(group, 0, mine, n)

    # -- XML -------------------------------------------------------------------

    def _sub(self, parent, tag, attrib=None):
        """A new element under parent, its top-level grid's text discarded."""
        node = ET.SubElement(parent, tag, attrib or {})
        top = self._top[parent]
        self._top[node] = top
        self._text.pop(top, None)
        self._changed.add(top)
        return node

    def _data_item(self, parent, ref, shape, unsigned=False, wide=False):
        item = self._sub(parent, 'DataItem')
        item.set('Dimensions', ' '.join(str(int(n)) for n in shape))
        if wide:
            item.set('NumberType', 'Int')
            item.set('Precision', '8')
        elif unsigned:
            item.set('NumberType', 'UInt')
        item.set('Format', 'HDF')
        item.text = ref
        return item

    def grid(self, name, parent=None):
        attrib = {'Name': name, 'GridType': 'Uniform'}
        if parent is None or parent is self.series:     # a grid at the top
            node = ET.SubElement(self.domain if parent is None else parent, 'Grid', attrib)
            self._top[node] = node
            self._changed.add(node)
            if parent is not None:
                self._step[node] = len(parent) - 1
            return node
        return self._sub(parent, 'Grid', attrib)

    def topology(self, grid, cells, topology_type, ref=None, shape=None, wide=False):
        # `shape` is the *global* one: writing in parallel, a rank holds a slice, and
        # the XML has to describe the whole dataset rather than that rank's part.
        # `wide`: the node numbers are stored as 64-bit integers (`topology_array`)
        n_cells, per_cell = shape if shape is not None else cells.shape
        node = self._sub(grid, 'Topology', {
            'NumberOfElements': str(n_cells), 'TopologyType': topology_type,
            'NodesPerElement': str(per_cell)})
        self._data_item(node, ref if ref is not None else cells,
                        (n_cells, per_cell), unsigned=True, wide=wide)
        return node

    def geometry(self, grid, coords, ref=None, shape=None):
        shape = shape if shape is not None else coords.shape
        node = self._sub(grid, 'Geometry', {'GeometryType': 'XY' if shape[1] == 2 else 'XYZ'})
        self._data_item(node, ref if ref is not None else coords, shape)
        return node

    def attribute(self, grid, name, values, center='Node', ref=None, shape=None):
        values = np.asarray(values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        shape = shape if shape is not None else values.shape
        kind = {1: 'Scalar', 3: 'Vector', 9: 'Tensor'}.get(shape[1])
        if kind is None:
            raise ValueError('{} components is not a Scalar, Vector or Tensor'
                             .format(shape[1]))
        node = self._sub(grid, 'Attribute', {
            'Name': name, 'AttributeType': kind, 'Center': center})
        self._data_item(node, ref if ref is not None else values, shape)
        return node

    def time(self, grid, value):
        """A `<Time>`, and the temporal collection it has to live in."""
        self._sub(grid, 'Time', {'Value': '{:.15e}'.format(float(value))})

    def collection(self):
        """The temporal collection dolfin puts every timed grid inside."""
        if self.series is None:
            self.series = ET.SubElement(self.domain, 'Grid', {
                'Name': 'TimeSeries', 'GridType': 'Collection',
                'CollectionType': 'Temporal'})
            self._changed.add(self.series)      # not a step: the file is rewritten
        # A grid written before the first timed one stays where it is: a grid with no
        # <Time> inside a temporal collection would be ambiguous to a reader and make a
        # static mesh a time step. (dolfin, given `write(mesh)` then `write(u, t)`,
        # drops the mesh grid.)
        return self.series

    # -- output ----------------------------------------------------------------

    def save(self):
        """The XML, written by rank zero (every rank builds the tree -- that is what
        keeps the dataset names in step -- but they must not all write the file).

        The same text `ET.indent` and `ET.tostring` give the whole tree, built from each
        top-level grid's text, kept until something is added to the grid. When only time
        steps changed since the last save -- new ones, or the last ones added to -- the
        file on disk is rewritten in place from the first of them on, in one write: a
        save costs what changed, not the whole series. Otherwise it is written beside
        and renamed over, so a job killed while saving leaves the last complete XML."""
        changed, self._changed = self._changed, set()
        if self.rank != 0:
            return
        saved = self._saved
        first = None
        if (saved is not None and changed and all(g in self._step for g in changed)
                and _file_size(self.path) == saved['size']):
            first = min(self._step[g] for g in changed)
        if first is not None:
            at = saved['steps_at'][first] if first < len(saved['steps_at']) else saved['at']
            texts = [self._grid_text(g, 3) for g in self.series[first:]]
            data = ''.join(texts).encode()
            with open(self.path, 'r+b') as handle:
                handle.seek(at)
                handle.write(data + saved['tail'])
                handle.truncate()
            del saved['steps_at'][first:]
            for t in texts:
                saved['steps_at'].append(at)
                at += len(t.encode())
            saved['at'], saved['size'] = at, at + len(saved['tail'])
            return
        head, steps, tail = self._parts()
        data = head.encode()
        steps_at = []
        for t in steps:
            steps_at.append(len(data))
            data += t.encode()
        tail = tail.encode()
        tmp = self.path + '.tmp'
        with open(tmp, 'wb') as handle:
            handle.write(data + tail)
        os.replace(tmp, self.path)
        self._saved = dict(steps_at=steps_at, at=len(data), tail=tail,
                           size=len(data) + len(tail))

    def _grid_text(self, node, level):
        """A top-level grid's XML at its level of indentation, kept while unchanged."""
        if node not in self._text:
            ET.indent(node, space='  ', level=level)
            tail, node.tail = node.tail, None
            self._text[node] = '  ' * level + ET.tostring(node, encoding='unicode') + '\n'
            node.tail = tail
        return self._text[node]

    def _parts(self):
        """`(head, step texts, tail)` of the XML: everything before the time series'
        steps, each step, and everything after them (the whole document in head when
        there is no series)."""
        def shell(node, level):
            """The open and close tags of node, indented."""
            bare = ET.Element(node.tag, node.attrib)
            bare.text = '\x00'
            a, b = ET.tostring(bare, encoding='unicode').split('\x00')
            return '  ' * level + a + '\n', '  ' * level + b + '\n'

        root_open, root_close = shell(self.root, 0)
        head = '<?xml version="1.0"?>\n' + DOCTYPE + '\n' + root_open
        if not len(self.domain):
            return head + '  <Domain />\n' + root_close, [], ''
        head += '  <Domain>\n'
        tail, steps, before = '', [], True
        for child in self.domain:
            if child is self.series:
                series_open, series_close = shell(child, 2)
                head += series_open
                steps = [self._grid_text(step, 3) for step in child]
                tail += series_close
                before = False
            elif before:
                head += self._grid_text(child, 2)
            else:
                tail += self._grid_text(child, 2)
        tail += '  </Domain>\n' + root_close
        return head, steps, tail

    def flush(self):
        # the HDF5 flush is collective under mpio, so every rank makes it
        if self._h5 is not None:
            self._h5.flush()
        self.save()

    def close(self):
        self.save()
        if self._h5 is not None:
            self._h5.close()          # collective under mpio
            self._h5 = None


def _geometry(x):
    """A mesh's coordinates as a grid stores them: XY for 2D, XYZ for 3D, and a 1D mesh's
    padded to XY (XDMF has no one-column geometry)."""
    x = np.asarray(x, dtype=float)
    return np.column_stack((x, np.zeros(len(x)))) if x.shape[1] == 1 else x


def _gwidth(mesh):
    """The columns `_geometry` stores for `mesh`."""
    return max(mesh.geometry().dim(), 2)


def _file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


#: the most nodes a grid's topology numbers in 32-bit integers
INT32_NODES = 2 ** 31


def is_wide(n_nodes):
    """Whether a topology over `n_nodes` nodes (all ranks') is stored in 64-bit integers:
    past `INT32_NODES` (a P2 mesh has ~1.33 nodes a cell, so ~1.6B cells), its XML saying
    so (`NumberType="Int" Precision="8"`). Smaller ones stay 32-bit."""
    return int(n_nodes) > INT32_NODES


def topology_array(a, n_nodes):
    """Topology as the XDMF files here store it (`is_wide`): 32-bit integers, or 64-bit past
    `INT32_NODES` nodes; a node number past the 32-bit range is refused rather than
    wrapped."""
    a = np.asarray(a)
    if is_wide(n_nodes):
        return a.astype(np.int64)
    if a.size and int(a.max()) >= INT32_NODES:
        raise ValueError("XDMF topology: node {} of {} nodes, past the 32-bit range"
                         .format(int(a.max()), int(n_nodes)))
    return a.astype(np.int32)


def _pad3(values):
    """XDMF's Vector is three-component; dolfin pads a 2D field the same way."""
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 2:
        return np.column_stack((values, np.zeros(len(values))))
    return values


# The degree of the nodal values a visualisation grid holds, by topology.
_NODAL_DEGREE = {'PolyLine': 1, 'Triangle': 1, 'Tetrahedron': 1,
                 'Edge_3': 2, 'Triangle_6': 2, 'Tetrahedron_10': 2}


class XDMFFile2(object):
    """A stand-in for `dolfin.XDMFFile` that does not throw the quadratic part away.

        with XDMFFile2(comm, 'u.xdmf') as f:
            f.write(u_)

    dolfin's own XDMFFile interpolates any Function down to one value per vertex
    before writing, so a P2 velocity comes out linear and the picture loses the
    midside nodes. This writes `Tetrahedron_10` cells on a tetrahedral mesh and
    `Triangle_6` cells on a triangle mesh instead, whose nodes are the P2 degrees of
    freedom, which ParaView renders once "Nonlinear Subdivision Level" is raised. A
    P1 Function goes on the linear cells, as dolfin writes it. The coordinates are
    the mesh's, XY or XYZ, as dolfin writes them (a 1D mesh's padded to XY).

    Everything else dolfin's XDMFFile does, this does too, in dolfin's own file
    layout -- see the module docstring for the surface and for the three deliberate
    differences. Anything it genuinely cannot write raises and says why, rather than
    accepting the call and doing nothing with it.
    """

    Encoding = Encoding

    def __init__(self, comm_or_name, name=None):
        if name is None:
            self.comm, self.path = None, comm_or_name
        else:
            self.comm, self.path = comm_or_name, name
        self.parameters = XDMFParameters()
        self._doc = None
        self._name, self._label = '', ''
        self._grids = {}            # (mesh id, degree, time) -> grid element
        self._stored = {}           # (mesh id, degree) -> (topology ref, geometry ref)
        self._nodes = {}            # what a field's write takes of its mesh and space
        self._delegate = None       # a real dolfin.XDMFFile, for checkpoints
        self._mesh = None           # the last mesh seen, for a MeshValueCollection

    # -- dolfin's Variable surface --------------------------------------------

    def name(self):
        return self._name

    def label(self):
        return self._label

    def rename(self, name, label=''):
        self._name, self._label = name, label

    def id(self):
        return id(self)

    # -- lifetime --------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close(compress_checkpoint=exc[0] is None)    # never mask the exception
        return False

    def close(self, compress_checkpoint=True):
        if self._doc is not None:
            self._doc.close()
            self._doc = None
        _release_plain(self._nodes)
        if self._delegate is not None:
            self._delegate.close()
            self._delegate = None
            if not compress_checkpoint:
                return
            # dolfin's checkpoint cannot compress: its HDF5 file rewritten, as every
            # output is (`compress.rewrite`), by every rank, its earlier steps copied as
            # their stored chunks. An appended series closed after each step would be
            # copied whole at each close, so it is rewritten only once its plain part
            # is CHECKPOINT_SHARE of the file: the copying stays a few times the final
            # size, and at most that share of it is uncompressed between rewrites
            h5 = os.path.splitext(self.path)[0] + '.h5'
            code, what = compress.rewrite(h5, self._mpi_comm(), share=CHECKPOINT_SHARE)
            if code == 1:
                raise RuntimeError('compressing the checkpoint {}: {}'.format(h5, what))

    def _mpi_comm(self):
        import dolfin as df
        return self.comm if self.comm is not None else df.MPI.comm_world

    def _document(self, encoding):
        Encoding.of(encoding)       # only HDF5 survives; anything else raises
        if self._doc is None:
            if self._delegate is not None:
                raise RuntimeError(
                    '{} already holds a checkpoint, which has a file layout of its '
                    'own; write the visualisation grids to another file'
                    .format(self.path))
            comm = self._mpi_comm()
            self._doc = XdmfDocument(self.path,
                                     comm=comm if _size(comm) > 1 else None)
        return self._doc

    def _finish(self):
        if self._doc is None:
            return
        if self.parameters['flush_output']:
            self._doc.flush()
        else:
            # the XML is kept complete after every write, close or no close: a
            # caller without a context manager otherwise ends with datasets in
            # the .h5 and no .xdmf naming them at all. `flush_output` only adds
            # the HDF5 flush; `save` is rank zero writing one small file.
            self._doc.save()

    def _rank(self):
        import dolfin as df
        comm = self.comm if self.comm is not None else df.MPI.comm_world
        return df.MPI.rank(comm)

    # -- write -----------------------------------------------------------------

    def write(self, obj, *args, time=None, encoding=None):
        """Writes a Function, a Mesh, a MeshFunction, a MeshValueCollection or points.

            f.write(u)                 f.write(u, t)
            f.write((u, p))            # several fields onto one node set
            f.write((u, p, tags))      # ... and a per-cell field beside them
            f.write(mesh)              f.write(facet_markers)
            f.write(points)            f.write(points, values)

        The trailing `encoding` argument dolfin takes is accepted in the same place,
        and `time=`/`encoding=` keywords as well. A time on anything but a Function
        is refused the way dolfin refuses it: there is no grid to hang it on.
        Several functions written together share one grid, the highest degree among
        them, so a P1 pressure beside a P2 velocity is carried onto the quadratic
        nodes. That is exact -- linear interpolation of a linear field -- but it does
        mean the file cannot tell you the pressure was only ever linear.

        A MeshFunction in the same tuple is written as a **cell-centred** attribute
        on that one grid rather than as a grid of its own: one value per cell, which
        is what a MeshFunction over the mesh's top dimension is. It must be over the
        same mesh and at that dimension, and is refused otherwise -- a facet
        MeshFunction has nothing to do with the cells of this grid, and writing it
        against them would put plausible numbers on the wrong entities.
        """
        import dolfin as df
        if _is_points(obj):
            # `write(points, values)` -- the values are an array, not the trailing
            # encoding, so this cannot go through the (time, encoding) parse below
            if time is not None:
                raise TypeError('points take no time; dolfin.XDMFFile refuses '
                                'this too')
            values = None
            for arg in args:
                if isinstance(arg, (list, tuple, np.ndarray)):
                    values = arg
                else:
                    encoding = _merge_write_arg(encoding, arg, 'encoding')
            return self._write_points(obj, values, encoding)

        t, e = _split_write_args(args)
        if t is not None:
            time = _merge_write_arg(time, t, 'time')
        if e is not None:
            encoding = _merge_write_arg(encoding, e, 'encoding')
        if isinstance(obj, (tuple, list)) and obj and any(
                isinstance(o, df.Function) for o in obj):
            # a Function in any position: the tuple's order does not matter
            functions, tags = _split_cell_tags(list(obj))
            return self._write_functions(functions, time, encoding, tags)
        if isinstance(obj, df.Function):
            return self._write_functions([obj], time, encoding)
        if time is not None:
            # dolfin has no write(Mesh, t) overload and says so
            raise TypeError('write({}, t) takes no time; only a Function does'
                            .format(type(obj).__name__))
        if isinstance(obj, df.Mesh):
            return self._write_mesh_only(obj, encoding)
        if _is_mesh_function(obj):
            return self._write_mesh_function(obj, encoding)
        if _is_mesh_value_collection(obj):
            return self._write_mesh_value_collection(obj, encoding)
        raise TypeError('XDMFFile2 does not write a {}; dolfin.XDMFFile writes '
                        'Function, Mesh, MeshFunction, MeshValueCollection and '
                        'points'.format(type(obj).__name__))

    # -- functions -------------------------------------------------------------

    def _write_functions(self, functions, time, encoding, tags=()):
        """The functions onto one node set, every rank writing the nodes whose vertices
        and edges it owns and its own cells (`XdmfDocument.rows`): nothing is gathered.

        Every rank runs every line: the XML is built on all of them and thrown away on
        all but rank zero, which costs nothing and keeps the dataset counters --
        `Mesh/0`, `VisualisationVector/3` -- in step across ranks. They have to be,
        because creating a dataset is a collective operation and the ranks must agree
        on its name.
        """
        if not functions:
            raise ValueError('nothing to write')
        doc = self._document(encoding)
        info = _function_nodes_distributed(functions, self._nodes)
        self._mesh = info['mesh']
        mesh, degree = info['mesh'], info['degree']
        conn, n_nodes, n_cells = info['conn'], info['n_nodes'], info['n_cells']
        topology_type = XDMF_TOPOLOGY[(mesh.ufl_cell().cellname(), conn.shape[1])]

        key = (mesh.id(), degree, time)
        grid = self._grids.get(key) if self.parameters['functions_share_mesh'] else None
        if grid is None:
            parent = doc.collection() if time is not None else None
            grid = doc.grid('mesh', parent)
            store = self._stored.get((mesh.id(), degree))
            if store is None or self.parameters['rewrite_function_mesh']:
                group = doc.mesh_group(time is not None)
                coords = node_positions(_geometry(mesh.coordinates()), info['layout'])
                store = (doc.slab(group + '/topology', info['cell_offset'],
                                  topology_array(conn, n_nodes), n_cells),
                         doc.rows(group + '/geometry', info['layout']['rows'], coords,
                                  n_nodes))
                self._stored[(mesh.id(), degree)] = store
            doc.topology(grid, conn, topology_type, ref=store[0],
                         shape=(n_cells, conn.shape[1]), wide=is_wide(n_nodes))
            doc.geometry(grid, None, ref=store[1],
                         shape=(n_nodes, _gwidth(mesh)))
            if time is not None:
                doc.time(grid, time)
            self._grids[key] = grid

        for name, rows, values in info['fields']:
            values = _pad3(values)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            ref = doc.rows('VisualisationVector/{}'.format(doc._next('VisualisationVector')),
                           rows, values, n_nodes)
            doc.attribute(grid, name, values, 'Node', ref=ref,
                          shape=(n_nodes, values.shape[1]))
        # A cell attribute takes the same rank-major block the topology took, at the
        # same offset -- the file's row order for a cell is the order `conn` was
        # written in, and `mf.array()` is over the same local cells in the same
        # order. Cells are not shared between ranks, so unlike the node fields there
        # is no ownership question to settle. They live under VisualisationVector
        # too: that group is where an attribute of a function grid goes, whatever it
        # is centred on and whatever its rank -- dolfin puts scalars there as readily
        # as vectors.
        for name, values in _local_cell_tags(tags, mesh, len(conn)):
            ref = doc.slab('VisualisationVector/{}'.format(doc._next('VisualisationVector')),
                           info['cell_offset'], values, n_cells)
            doc.attribute(grid, name, values, 'Cell', ref=ref, shape=(n_cells, 1))
        self._finish()

    # -- mesh ------------------------------------------------------------------

    def _write_mesh_only(self, mesh, encoding):
        """The mesh: vertices by global index, each rank's owned cells rank-major."""
        self._mesh = mesh
        doc = self._document(encoding)
        gvert = np.asarray(mesh.topology().global_indices(0))
        owned = owned_entities(mesh, 0)
        n_vert = mesh.num_entities_global(0)
        n_cell = mesh.num_entities_global(mesh.topology().dim())
        cells = gvert[_owned_cells(mesh)]
        offset = _exscan(mesh.mpi_comm(), len(cells))

        grid = doc.grid('mesh')
        group = doc.mesh_group(False)
        store = (doc.slab(group + '/topology', offset, topology_array(cells, n_vert), n_cell),
                 doc.rows(group + '/geometry', gvert[owned],
                          _geometry(mesh.coordinates())[owned], n_vert))
        self._stored[(mesh.id(), 1)] = store
        doc.topology(grid, cells,
                     XDMF_TOPOLOGY[(mesh.ufl_cell().cellname(), cells.shape[1])],
                     ref=store[0], shape=(n_cell, cells.shape[1]), wide=is_wide(n_vert))
        doc.geometry(grid, None, ref=store[1], shape=(n_vert, _gwidth(mesh)))
        self._grids[(mesh.id(), 1, None)] = grid
        self._finish()

    # -- mesh functions --------------------------------------------------------

    def _write_mesh_function(self, mf, encoding):
        """Vertices by global index; entities and their values rank-major together.

        The two must go to the *same* block on each rank or a value would end up
        against another entity's row -- which is why they share one offset rather than
        each taking their own.
        """
        mesh, dim = mf.mesh(), mf.dim()
        self._mesh = mesh
        doc = self._document(encoding)
        mesh.init(dim, 0)
        mesh.init_global(dim)       # the entities' global count, which is 0 before
        gvert = np.asarray(mesh.topology().global_indices(0))
        owned_v = owned_entities(mesh, 0)
        owned_e = owned_entities(mesh, dim)
        n_vert = mesh.num_entities_global(0)
        n_ent = mesh.num_entities_global(dim)
        rows = gvert[_connectivity(mesh, dim)][owned_e]
        values = np.asarray(mf.array())[owned_e].reshape(-1, 1)
        offset = _exscan(mesh.mpi_comm(), int(owned_e.sum()))

        group = 'MeshFunction/{}'.format(doc._next('MeshFunction'))
        grid = doc.grid('mesh')
        topo = doc.slab(group + '/mesh/topology', offset, topology_array(rows, n_vert), n_ent)
        geom = doc.rows(group + '/mesh/geometry', gvert[owned_v],
                        _geometry(mesh.coordinates())[owned_v], n_vert)
        doc.topology(grid, rows, _entity_topology(mesh, dim, rows.shape[1]),
                     ref=topo, shape=(n_ent, rows.shape[1]), wide=is_wide(n_vert))
        doc.geometry(grid, None, ref=geom, shape=(n_vert, _gwidth(mesh)))
        name = mf.name() if mf.name() and mf.name() != 'f' else 'meshfunction'
        doc.attribute(grid, name, values, 'Cell',
                      ref=doc.slab(group + '/values', offset, values, n_ent),
                      shape=(n_ent, 1))
        self._finish()

    def _write_mesh_value_collection(self, mvc, encoding):
        # A MeshValueCollection holds a mesh in C++ and does not expose it to Python
        # in this dolfin -- `dir(mvc)` has dim, size, values and nothing else -- so
        # unlike every other object here it cannot say what it is defined over. The
        # mesh already written to this file is the one it belongs to in any use that
        # makes sense; with no mesh written, there is nothing to guess from.
        mesh = self._mesh
        if mesh is None:
            raise ValueError(
                'a MeshValueCollection does not carry its mesh in this dolfin, so '
                'write the mesh (or a MeshFunction, or a Function) to {} first'
                .format(self.path))
        # dolfin visualises it as the entities it marks, which is what a MeshFunction
        # over the same dimension holds -- but only the marked ones, so it is built
        # by hand rather than by widening it to every entity.
        dim = mvc.dim()
        tdim = mesh.topology().dim()
        mesh.init(tdim, dim)
        mesh.init(dim, 0)
        doc = self._document(encoding)
        # global vertex numbers and the ownership filter: each mark is written once,
        # by the rank owning its entity, against vertices by global index
        owned = owned_entities(mesh, dim)
        gvert = np.asarray(mesh.topology().global_indices(0))
        triples = None
        try:
            triples = _cpp().mvc_values(mvc)
        except TypeError:       # a value type the helper has no overload for
            pass                # (only size_t has one; the loop below takes the rest)
        if triples is not None:
            # three aligned arrays in map order, then everything is indexing:
            # the entity each pair names, the ownership filter, its vertices
            cell_of, local_of, values = (np.asarray(a) for a in triples)
            entity = (_connectivity(mesh, tdim, dim)[cell_of, local_of]
                      if len(cell_of) else np.zeros(0, dtype=np.int64))
            keep = owned[entity]
            entity, values = entity[keep], values[keep]
            rows = gvert[_connectivity(mesh, dim, 0)[entity]]
        else:
            cell_entities = mesh.topology()(tdim, dim)
            entity_vertices = mesh.topology()(dim, 0)
            rows, values = [], []
            for (cell, local), value in mvc.values().items():
                entity = cell_entities(cell)[local]
                if not owned[entity]:
                    continue          # another rank writes this one
                rows.append(gvert[np.asarray(entity_vertices(entity))])
                values.append(value)

        comm = mesh.mpi_comm()
        width = dim + 1
        entities = np.array(rows, dtype=np.int64).reshape(len(rows), width) \
            if len(rows) else np.zeros((0, width), dtype=np.int64)
        marks = np.asarray(values, dtype=_mvc_dtype(mvc)).reshape(-1, 1)
        n_marked = comm.allreduce(len(entities))
        offset = _exscan(comm, len(entities))
        owned_v = owned_entities(mesh, 0)
        n_vert = mesh.num_entities_global(0)

        group = 'MeshValueCollection/{}'.format(doc._next('MeshValueCollection'))
        name = mvc.name() or 'm'
        grid = doc.grid(name)
        doc.topology(grid, entities, _entity_topology(mesh, dim, width),
                     ref=doc.slab(group + '/topology', offset,
                                  topology_array(entities, n_vert), n_marked),
                     shape=(n_marked, width), wide=is_wide(n_vert))
        doc.geometry(grid, None, shape=(n_vert, _gwidth(mesh)),
                     ref=doc.rows(group + '/geometry', gvert[owned_v],
                                  _geometry(mesh.coordinates())[owned_v], n_vert))
        doc.attribute(grid, name, marks, 'Cell', shape=(n_marked, 1),
                      ref=doc.slab(group + '/values', offset, marks, n_marked))
        self._finish()

    # -- points ----------------------------------------------------------------

    def _write_points(self, points, values, encoding):
        try:
            coords = np.asarray(_cpp().points_array(points))
        except TypeError:           # not dolfin Points; the caster refused them
            coords = np.array([p.array() if hasattr(p, 'array') else list(p)
                               for p in points], dtype=float)
        coords = _pad_coords(coords)
        doc = self._document(encoding)
        if values is not None:
            values = np.asarray(values, dtype=float).reshape(-1, 1)
        # Points are given to every rank, not partitioned across them, so rank zero
        # writes the lot and the others take part with nothing (`XdmfDocument.dataset`).
        # That keeps the meaning it has in serial rather than concatenating the same
        # list once per rank.
        n = len(coords)
        conn = np.arange(n, dtype=np.int64).reshape(-1, 1)
        group = doc.free_group('Points')
        grid = doc.grid('points')
        doc.topology(grid, conn, 'Polyvertex', shape=(n, 1), wide=is_wide(n),
                     ref=doc.dataset(group + '/topology', topology_array(conn, n)))
        doc.geometry(grid, None, shape=(n, 3), ref=doc.dataset(group + '/geometry', coords))
        if values is not None:
            doc.attribute(grid, 'values', values, 'Node', shape=(n, 1),
                          ref=doc.dataset(group + '/values', values))
        self._finish()

    # -- checkpoints -----------------------------------------------------------

    def write_checkpoint(self, u, function_name, time_step=0.0, encoding=None,
                         append=False):
        """dolfin's checkpoint, unchanged -- it already round-trips P2 exactly.

        The checkpoint stores the dof vector and the element beside it, so nothing is
        lost and there is nothing here to improve on. It has a file layout of its own,
        though, so a file cannot hold both a checkpoint and visualisation grids;
        asking for both says so. The parameters are dolfin's, names included
        (`function_name`, `time_step`), and `append` is forwarded: dolfin truncates the
        file on every call that does not append.
        """
        import dolfin as df
        if self._doc is not None:
            raise RuntimeError(
                '{} already holds visualisation grids, and a checkpoint has a file '
                'layout of its own; use another file'.format(self.path))
        if self._delegate is None:
            comm = self.comm if self.comm is not None else df.MPI.comm_world
            self._delegate = df.XDMFFile(comm, self.path)
            for key, value in self.parameters.items():
                self._delegate.parameters[key] = value
        Encoding.of(encoding)   # only HDF5 survives, which is dolfin's default
        self._delegate.write_checkpoint(u, function_name, float(time_step),
                                        df.XDMFFile.Encoding.HDF5, bool(append))

    def read_checkpoint(self, u, function_name, counter=-1):
        """Reads dolfin's checkpoint, or a time series of nodal values written here.

        dolfin's reader knows only its own checkpoint format. If the file instead
        holds visualisation grids -- P1 (`Triangle`, `Tetrahedron`, `PolyLine`) or P2
        (`Triangle_6`, `Tetrahedron_10`, `Edge_3`), the field at the nodes, which
        dolfin cannot read at all -- the field is rebuilt from the nodes onto an
        unconstrained space of the grid's degree, and then interpolated onto
        whatever space `u` lives in, of any degree (fast_interpolation). Going
        through the unconstrained space is what makes it safe for a periodic one,
        where matching dof coordinates does not line up node for node.

        `counter` defaults to -1 -- the last stored step -- because that is dolfin's
        default; a drop-in defaulting to the first step would hand back the wrong
        field with no error.
        """
        import dolfin as df
        grids = _scan(self.path)
        if any(g['topology'] in _NODAL_DEGREE and
               (_NODAL_DEGREE[g['topology']] == 2
                or g['centers'].get(function_name) == 'Node') for g in grids):
            return self._read_nodal(u, function_name, counter, grids)
        comm = self.comm if self.comm is not None else df.MPI.comm_world
        with df.XDMFFile(comm, self.path) as f:
            f.read_checkpoint(u, function_name, counter)

    def _read_nodal(self, u, name, counter, grids):
        import dolfin as df
        picked = [g for g in grids if name in g['attributes']]
        if not picked:
            raise ValueError('no field named {!r} in {}; the file has {}'.format(
                name, self.path, ', '.join(sorted(
                    {a for g in grids for a in g['attributes']})) or 'none'))
        if not -len(picked) <= counter < len(picked):
            raise ValueError('{} has {} step(s) of {!r}; counter {} is out of '
                             'range'.format(self.path, len(picked), name, counter))
        grid = picked[counter]

        V = u.function_space()
        degree = _NODAL_DEGREE[grid['topology']]
        width = int(np.prod(u.value_shape())) if len(u.value_shape()) else 1
        mesh = V.mesh()
        # The nodes are read into an unconstrained space and interpolated onto V,
        # which is right for a periodic V too, where matching dof coordinates is
        # not. When V is that unconstrained space they are read into it directly.
        if _is_plain(V, mesh, degree, width):
            _scatter_nodes(u, V, self.path, grid, name, width, degree)
            return u
        plain = _plain_space(mesh, degree, width, self._nodes)
        target = df.Function(plain)
        _scatter_nodes(target, plain, self.path, grid, name, width, degree)
        fast_interpolate(target, V, out=u)
        return u

    # -- read ------------------------------------------------------------------

    def read(self, obj, name=''):
        """Reads a Mesh, a MeshFunction or a MeshValueCollection.

        A file with no quadratic grid in it is handed to dolfin, which reads its own
        layout -- and ours, which is the same one -- with no help from here. Only a
        file holding `Tetrahedron_10` or `Triangle_6`, which dolfin's reader cannot
        parse, is read here instead.
        """
        import dolfin as df
        grids = _scan(self.path)
        quadratic = any(g['topology'] in ('Tetrahedron_10', 'Triangle_6', 'Edge_3')
                        for g in grids)
        if not quadratic:
            comm = self.comm if self.comm is not None else df.MPI.comm_world
            with df.XDMFFile(comm, self.path) as f:
                return f.read(obj) if not name else f.read(obj, name)
        if isinstance(obj, df.Mesh):
            return self._read_mesh(obj, grids)
        raise ValueError(
            '{} holds quadratic cells, which this reads back only as a Mesh (its '
            'corners) or through read_checkpoint (the field)'.format(self.path))

    def _read_mesh(self, mesh, grids):
        import dolfin as df
        if df.MPI.size(mesh.mpi_comm()) > 1:
            raise RuntimeError('reading a quadratic XDMF is serial only; dolfin '
                               'partitions its own files, this does not')
        # the mesh is the grid of the highest-dimensional cells: a facet MeshFunction's
        # grid, or points, may come first in the file
        cells = [g for g in grids
                 if TOPOLOGY_CELL.get(g['topology'], ('point',))[0] != 'point']
        if not cells:
            raise ValueError('{} holds no grid of cells'.format(self.path))
        grid = max(cells, key=lambda g: CELL_DIM[TOPOLOGY_CELL[g['topology']][0]])
        coords, conn = _read_grid_geometry(self.path, grid)
        cellname, per_cell = TOPOLOGY_CELL[grid['topology']]
        corners = conn[:, :{'tetrahedron': 4, 'triangle': 3,
                            'interval': 2}[cellname]]
        used, conn = np.unique(corners, return_inverse=True)
        conn = conn.reshape(corners.shape)
        coords = coords[used]
        _cpp().build_mesh(mesh, cellname, CELL_DIM[cellname], coords,
                          conn.astype(np.uintp))
        return mesh


# ---------------------------------------------------------------------------
# the dolfin-shaped helpers: everything that has to touch a Function or a Mesh
# ---------------------------------------------------------------------------

def _size(comm):
    if comm is None:
        return 1
    return comm.Get_size() if hasattr(comm, 'Get_size') else 1


def _split_write_args(args):
    """dolfin's trailing `(t, encoding)`, either, both or neither.

    A time can arrive as a numpy scalar -- `times[i]` off a float32 array is an
    `np.float32`, which is not a `float` subclass -- so the test is numpy-aware.
    Booleans are neither: `True` routed to time would be silently accepted as 1.0.
    """
    time, encoding = None, None
    for arg in args:
        if isinstance(arg, (bool, np.bool_)):
            encoding = arg              # let Encoding.of refuse it by name
        elif isinstance(arg, (float, int, np.floating, np.integer)):
            time = float(arg)
        else:
            encoding = arg
    return time, encoding


def _merge_write_arg(keyword, positional, what):
    """One value for `time` or `encoding`, given positionally or by keyword."""
    if keyword is not None and positional is not None:
        raise TypeError('{} given twice, positionally and by keyword'.format(what))
    return positional if positional is not None else keyword


def _is_points(obj):
    """A list of dolfin Points, as `write(points)` takes -- not a list of Functions."""
    if not isinstance(obj, (list, tuple)) or not obj:
        return False
    first = obj[0]
    return hasattr(first, 'x') and hasattr(first, 'y') and hasattr(first, 'z')


def _mvc_dtype(mvc):
    """The dtype a MeshValueCollection's values really have, off its class name.

    dolfin writes each value type as it is -- a uint64 would floor a double
    collection's values and wrap a -1 'unmarked' int to 2**64-1 -- and so does this.
    dolfin's classes are MeshValueCollection_sizet, _double, _int and _bool.
    """
    name = type(mvc).__name__
    if name.endswith('_double'):
        return np.float64
    if name.endswith('_int'):
        return np.int32
    if name.endswith('_bool'):
        return np.uint8
    return np.uint64


def _is_mesh_function(obj):
    return type(obj).__name__.startswith('MeshFunction')


def _is_mesh_value_collection(obj):
    return type(obj).__name__.startswith('MeshValueCollection')


def _pad_coords(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[1] < 3:
        coords = np.column_stack(
            [coords] + [np.zeros(len(coords))] * (3 - coords.shape[1]))
    return coords


_CPP_CODE = r'''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshTopology.h>
#include <dolfin/mesh/MeshConnectivity.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/MeshEditor.h>
namespace py = pybind11;

// MeshConnectivity stores the whole table as one contiguous vector, which the
// Python layer does not expose (its MeshConnectivity has `size` and nothing
// else); this hands it to numpy in one copy.
py::array_t<unsigned int> connectivity(const dolfin::Mesh& mesh,
                                       std::size_t d0, std::size_t d1)
{
  const std::vector<unsigned int>& c = mesh.topology()(d0, d1)();
  return py::array_t<unsigned int>(c.size(), c.data());
}

// A list of Points costs three Python method calls per point to read from
// Python; the stl caster converts the whole list at C++ speed instead.
py::array_t<double> points_array(const std::vector<dolfin::Point>& pts)
{
  py::array_t<double> out({pts.size(), std::size_t(3)});
  double* p = out.mutable_data();
  for (const auto& pt : pts) { *p++ = pt.x(); *p++ = pt.y(); *p++ = pt.z(); }
  return out;
}

// The (cell, local_entity) -> value map of a MeshValueCollection, as three
// aligned arrays in map order. From Python the map arrives as a dict costing
// per entry; from here it is three pointer walks.
py::tuple mvc_values(const dolfin::MeshValueCollection<std::size_t>& mvc)
{
  const auto& map = mvc.values();
  const std::size_t n = map.size();
  py::array_t<std::int64_t> cells(n), locals(n);
  py::array_t<std::size_t> vals(n);
  auto* c = cells.mutable_data();
  auto* l = locals.mutable_data();
  auto* v = vals.mutable_data();
  for (const auto& kv : map) {
    *c++ = kv.first.first;
    *l++ = kv.first.second;
    *v++ = kv.second;
  }
  return py::make_tuple(cells, locals, vals);
}

// A serial mesh from its arrays, through MeshEditor at C++ speed: from Python it is
// two calls a vertex and a cell, minutes at ten million cells.
void build_mesh(dolfin::Mesh& mesh, const std::string& cellname, std::size_t tdim,
                py::array_t<double, py::array::c_style | py::array::forcecast> x,
                py::array_t<std::size_t, py::array::c_style | py::array::forcecast> cells)
{
  const std::size_t gdim = x.shape(1), nv = cells.shape(1);
  dolfin::MeshEditor editor;
  editor.open(mesh, cellname, tdim, gdim);
  editor.init_vertices(x.shape(0));
  editor.init_cells(cells.shape(0));
  const double* px = x.data();
  for (std::size_t i = 0; i < std::size_t(x.shape(0)); ++i)
    editor.add_vertex(i, dolfin::Point(gdim, px + i*gdim));
  const std::size_t* pc = cells.data();
  std::vector<std::size_t> v(nv);
  for (std::size_t i = 0; i < std::size_t(cells.shape(0)); ++i)
  {
    std::copy(pc + i*nv, pc + (i + 1)*nv, v.begin());
    editor.add_cell(i, v);
  }
  editor.close();
}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("connectivity", &connectivity);
  m.def("points_array", &points_array);
  m.def("mvc_values", &mvc_values);
  m.def("build_mesh", &build_mesh);
}
'''
_CPP_MODULE = False             # False: not tried yet; the module afterwards


def _cpp():
    """The JIT-compiled C++ helpers. Raises if they will not build.

    `dolfin.compile_cpp_code` needs a compiler and the dolfin headers
    (`libdolfin-dev`); the first call compiles and dijitso caches the result for
    later processes. Both are assumed present, as they are by dolfin itself, which
    JIT-compiles every variational form. The fallbacks below are about argument types
    the pybind casters refuse, not about the compiler.

    This is reached only from the dolfin-facing paths; the document core reads HDF5
    with numpy and h5py and never comes here, so the module is usable with no dolfin
    at all.
    """
    global _CPP_MODULE
    if _CPP_MODULE is False:
        import dolfin as df
        _CPP_MODULE = df.compile_cpp_code(_CPP_CODE)
    return _CPP_MODULE


def _connectivity(mesh, dim, of_dim=0):
    """The entity-to-vertex table of `dim`, as an array rather than a Python list.

    `mesh.topology()(d, 0)()` hands back a Python list of ints, slow to walk on a large
    mesh. The C++ storage behind it is one contiguous vector, which the JIT helper
    above reads in one copy. The two dimensions that have a real array behind them
    avoid the question altogether.
    """
    tdim = mesh.topology().dim()
    if of_dim == 0 and dim == tdim:
        return np.asarray(mesh.cells(), dtype=np.int64)
    if of_dim == 0 and dim == 0:
        return np.arange(mesh.num_entities(0), dtype=np.int64).reshape(-1, 1)
    mesh.init(dim, of_dim)
    conn = mesh.topology()(dim, of_dim)
    n_entities = mesh.num_entities(dim)
    per = conn.size(0) if n_entities else 0
    flat = _cpp().connectivity(mesh, dim, of_dim).astype(np.int64)
    return flat.reshape(n_entities, per)


def _entity_topology(mesh, dim, per_cell):
    cellname = {0: 'point', 1: 'interval', 2: 'triangle', 3: 'tetrahedron'}[dim]
    return XDMF_TOPOLOGY[(cellname, per_cell)]


def scatter_vertex_values(f, per_vertex, width=1):
    """Per-vertex values into a Function's owned dofs, ghosts left to `apply`.

    `vertex_to_dof_map` covers this rank's
    ghost vertices too, and their dof indices run past the end of `get_local()`,
    which is over the ownership range alone -- assigning through them would either
    raise or, with a negative index, write a value onto some other node's dof.

    A blocked dofmap keeps a node's components together, so testing the first
    component covers all `width` of them. Works on a periodic space as well:
    `vertex_to_dof_map` sends a slave vertex to its master's dof, so both faces
    write the same dof with the same value.
    """
    import dolfin as df
    v2d = df.vertex_to_dof_map(f.function_space()).reshape(-1, width)
    values = f.vector().get_local()
    owned = (v2d[:, 0] >= 0) & (v2d[:, 0] < len(values))
    values[v2d[owned]] = np.asarray(per_vertex).reshape(len(v2d), width)[owned]
    f.vector().set_local(values)
    f.vector().apply("insert")
    return f


def _plain_space(mesh, degree, width, cache):
    """An unconstrained Lagrange space, built once per mesh, degree and width and kept in
    `cache`, which is a file's (`XDMFFile2._nodes`) and goes when the file closes.

    Reading needs one (`_read_nodal`): the file's nodes are matched to its distinct dof
    coordinates and the field interpolated from it onto the caller's space. Writing
    reads a field's own dofs (`_node_dofs`) and builds none. A series read step by step
    builds it once; it goes with the file rather than with the process, since the space
    and its transfers (`fast_interpolation`) are as large as the field's.
    """
    import dolfin as df
    key = ('plain', mesh.id(), degree, width)
    space = cache.get(key)
    if space is None:
        cell = mesh.ufl_cell()
        element = (df.VectorElement('Lagrange', cell, degree, dim=width) if width > 1
                   else df.FiniteElement('Lagrange', cell, degree))
        space = df.FunctionSpace(mesh, element)
        cache[key] = space
    return space


def _release_plain(cache):
    """Let go of what `cache` holds -- its unconstrained spaces, the transfers onto and
    from them, the node layouts and node dofs -- and hand the heap back
    (`femtools.memory`) whenever it held anything: a write's own transients (the dofmap's
    entity lists, the compressed chunks) otherwise stay resident."""
    from femtools import fast_interpolation
    from femtools.memory import drop
    held = [v for k, v in cache.items() if isinstance(k, tuple) and k[:1] == ('plain',)]
    for space in held:
        fast_interpolation.forget(space)
    anything = bool(cache)
    cache.clear()
    if anything:
        drop()


def _is_plain(V, mesh, degree, width):
    """Whether a space is already unconstrained Lagrange of this degree and width.

    A periodic space identifies its slave dofs with masters, so it is strictly
    smaller than the unconstrained one; comparing the dimensions is therefore an
    exact test and not a heuristic. When it holds, a file's nodes can be read straight
    into the field's own dofs (`_read_nodal`) and neither a new space nor an
    interpolation is needed.
    """
    element = V.ufl_element()
    if element.family() not in ('Lagrange', 'P', 'CG') or element.degree() != degree:
        return False
    n = mesh.num_entities_global(0)
    if degree == 2:
        mesh.init(1)
        mesh.init_global(1)
        n += mesh.num_entities_global(1)
    return V.dim() == n * width


def _split_cell_tags(objects):
    """A `write((u, p, tags))` tuple split into its Functions and its MeshFunctions.

    Order does not matter -- the nodal fields are written first regardless -- but
    anything that is neither is refused by name rather than failing later inside the
    node assembly, where the message would be about a missing `function_space`.
    """
    import dolfin as df
    functions, tags = [], []
    for obj in objects:
        if isinstance(obj, df.Function):
            functions.append(obj)
        elif _is_mesh_function(obj):
            tags.append(obj)
        else:
            raise TypeError(
                'a tuple holds Functions to write on one node set and MeshFunctions '
                'to write as cell data beside them, not a {}'
                .format(type(obj).__name__))
    if not functions:
        raise ValueError('cell data needs a grid to sit on: a tuple of MeshFunctions '
                         'alone has no node set. Write one on its own -- f.write(mf) '
                         '-- for a grid of its own')
    return functions, tags


def _check_cell_tags(tags, mesh):
    """Every MeshFunction in a `write((u, ..., mf))` tuple is cell data for that grid.

    Two things have to hold and neither is safe to assume. It must be over the *same*
    mesh, or its values are being written against another mesh's cells. And it must be
    over the mesh's **top** dimension, which is the only dimension with one value per
    cell of the grid: a facet MeshFunction has a different length and a different
    meaning, and pairing it with the cells would put entirely plausible numbers on the
    wrong entities -- the sort of file that looks right in ParaView and is not.
    """
    tdim = mesh.topology().dim()
    for mf in tags:
        if mf.mesh().id() != mesh.id():
            raise ValueError('the MeshFunction is over a different mesh from the '
                             'functions; it cannot be their cell data')
        if mf.dim() != tdim:
            raise ValueError(
                'a MeshFunction written beside functions is cell data, so it must be '
                'over dimension {} (the cells); this one is over dimension {}. Write '
                'it on its own -- f.write(mf) -- for a grid of its own'
                .format(tdim, mf.dim()))
    return tdim


def _tag_name(mf, index):
    """A MeshFunction's name, or one that says which of them it was.

    'f' is dolfin's placeholder for an unnamed MeshFunction, and `_write_mesh_function`
    already treats it as no name at all; this agrees with it.
    """
    name = mf.name()
    return name if name and name != 'f' else 'meshfunction{}'.format(
        '' if index == 0 else index)


def _local_cell_tags(tags, mesh, n_local):
    """(name, values) per MeshFunction, each rank keeping only its own cells.

    `n_local` is the number of rows this rank contributed to the topology, its owned
    cells, which come first: a ghost cell's value (ghost_mode other than "none") is
    its owner's to write.
    """
    if not tags:
        return []
    _check_cell_tags(tags, mesh)
    return [(_tag_name(mf, index), np.asarray(mf.array())[:n_local].reshape(-1, 1))
            for index, mf in enumerate(tags)]


def _check_functions(functions):
    """What this can write: refused here, before anything is written, rather than a
    DG function quietly interpolated onto a Lagrange space and written as though it
    had been continuous all along.

    Degree 1 has no midsides and is written on any simplex. Degree 2 needs the
    cell's midside numbering, which `QUADRATIC` holds for triangles and tetrahedra.
    """
    spaces = [f.function_space() for f in functions]
    mesh = spaces[0].mesh()
    for V in spaces[1:]:
        if V.mesh().id() != mesh.id():
            raise ValueError('the functions are on different meshes')
    for V in spaces:
        element = V.ufl_element()
        if element.family() not in ('Lagrange', 'P', 'CG'):
            raise ValueError('{} is not a Lagrange element; use dolfin.XDMFFile'
                             .format(element.family()))
        if element.degree() not in (1, 2):
            raise ValueError('degree {} is not something this writes; use '
                             'dolfin.XDMFFile'.format(element.degree()))
    degree = max(V.ufl_element().degree() for V in spaces)
    cellname = mesh.ufl_cell().cellname()
    if cellname not in ('interval', 'triangle', 'tetrahedron'):
        raise ValueError('{} cells are not something this writes'.format(cellname))
    if degree == 2 and cellname not in QUADRATIC:
        raise ValueError('quadratic functions are written on triangles and '
                         'tetrahedra, not {}; use dolfin.XDMFFile (which drops the '
                         'midsides)'.format(cellname))
    widths = [int(np.prod(f.value_shape())) if len(f.value_shape()) else 1
              for f in functions]
    return mesh, degree, widths


def _entity_dofs(dofmap, mesh, dim):
    """`dofmap.entity_dofs` as an integer array, empty or not: it hands back a list, and
    `np.asarray([])` is float64, which cannot index -- on a rank with no entities of the
    dimension (a mesh built by hand can leave one so) that would raise there alone."""
    return np.asarray(dofmap.entity_dofs(mesh, dim), dtype=np.int64)


def owned_entities(mesh, dim):
    """Which of this rank's entities of `dim` it is the one responsible for writing.

    It decides a Function's nodes as well as a Mesh's or a MeshFunction's entities
    (`_node_layout`): a periodic space gives a slave entity its master's dof, so dof
    ownership cannot say who writes the nodes on both sides of the period, and the
    value is read from the vector's ghosts where the dof is another rank's
    (`_node_dofs`). The rule is the lowest rank among those sharing the entity, read off
    `mesh.topology().shared_entities(dim)`. It needs no communication -- each rank
    already knows who else holds the entity -- and it partitions the entities exactly,
    which `hdf5_rows.write_rows` checks on every write.
    """
    import dolfin as df
    from femtools.hdf5_rows import agreed
    comm = mesh.mpi_comm()
    tdim = mesh.topology().dim()
    n_own = mesh.topology().ghost_offset(tdim)
    if agreed(comm, n_own < mesh.num_cells()):
        return _owned_with_ghosts(mesh, dim, n_own)
    rank = df.MPI.rank(comm)
    shared = mesh.topology().shared_entities(dim)
    owned = np.ones(mesh.num_entities(dim), dtype=bool)
    for local, others in shared.items():
        if others and min(others) < rank:
            owned[local] = False
    return owned


def _owned_with_ghosts(mesh, dim, n_own):
    """`owned_entities` on a mesh with ghost cells (ghost_mode other than "none"): an
    entity a rank holds through ghost cells alone need not be in `shared_entities`, so
    the rule there would have it written twice. Here an entity is written by the lowest
    rank holding it in one of its own cells: each rank sends the global ids of its own
    cells' entities to the rank the id hashes to, which answers whether the sender is
    the lowest. Collective."""
    from femtools.hdf5_rows import rank_of, reply, route, unique_inverse
    comm = mesh.mpi_comm()
    tdim = mesh.topology().dim()
    mesh.init(tdim, dim)
    mesh.init_global(dim)
    held = np.zeros(mesh.num_entities(dim), dtype=bool)
    held[_connectivity(mesh, tdim, dim)[:n_own].ravel()] = True
    mine = np.nonzero(held)[0]
    gid = np.asarray(mesh.topology().global_indices(dim), dtype=np.int64)[mine]
    routed = route(comm, rank_of(gid, comm.Get_size()), gid)
    got = routed.received[0]
    source = np.repeat(np.arange(comm.Get_size()), routed.recv_counts)
    _, inverse = unique_inverse(got)
    lowest = np.full(inverse.max() + 1 if len(inverse) else 0, comm.Get_size())
    np.minimum.at(lowest, inverse, source)
    (first,) = reply(comm, routed, (source == lowest[inverse]).astype(np.int8))
    owned = np.zeros(mesh.num_entities(dim), dtype=bool)
    owned[mine] = first.astype(bool)
    return owned


def _exscan(comm, value):
    """This rank's offset into a rank-major concatenation."""
    offset = comm.exscan(int(value))
    return 0 if offset is None else int(offset)


def _owned_cells(mesh):
    """This rank's cells as local vertex indices, ghost cells (ghost_mode other than
    "none", numbered after dolfin's ghost offset) left to the ranks owning them."""
    tdim = mesh.topology().dim()
    return np.asarray(mesh.cells())[:mesh.topology().ghost_offset(tdim)]


def node_layout(comm, gvert, cells, owned_vertices, n_vertices, n_cells, edges=None):
    """What a write of a degree-1 or 2 field takes of the mesh's topology, from either
    backend's arrays: `gvert` each local vertex's global id, `cells` this rank's own cells
    as local vertices, `owned_vertices` which local vertices this rank writes, and the
    global vertex and cell counts. At degree 2 `edges` holds `gedge` (each local edge's
    global id), `cells` (the own cells' local edges), `ends` (each local edge's two local
    vertices), `owned` (which local edges this rank writes), `n` (the global edge count)
    and `order` (a cell's vertex then edge columns put in XDMF's order, or None).

    Returns a dict: `conn` the own cells' global node numbers (a node's number is its
    vertex's global id, or `n_vertices` plus its edge's), `rows` the numbers of the nodes
    this rank writes, `owned` their masks (vertices, then edges), `ends`, the counts
    `n_nodes` and `n_cells`, and `cell_offset`, the cells' first row in the rank-major
    topology."""
    gvert = np.asarray(gvert)
    conn = gvert[cells]
    owned, rows = [owned_vertices], [gvert[owned_vertices]]
    n_nodes, ends = n_vertices, None
    if edges is not None:
        gedge = np.asarray(edges['gedge'])
        conn = np.hstack((conn, gedge[edges['cells']] + n_vertices))
        if edges.get('order') is not None:
            conn = conn[:, edges['order']]
        owned.append(edges['owned'])
        rows.append(gedge[edges['owned']] + n_vertices)
        n_nodes += edges['n']
        ends = edges['ends']
    return dict(conn=conn, rows=np.concatenate(rows), owned=owned, ends=ends,
                n_nodes=n_nodes, n_cells=n_cells, cell_offset=_exscan(comm, len(conn)))


def node_positions(x, layout):
    """The positions of the nodes a rank writes (`node_layout`): its vertices' of `x`
    (the local vertices' coordinates), then its edges' midpoints."""
    owned = layout['owned']
    out = [x[owned[0]]]
    if len(owned) > 1:
        ends = layout['ends'][owned[1]]
        out.append(0.5 * (x[ends[:, 0]] + x[ends[:, 1]]))
    return np.vstack(out)


def node_dofs(layout, vertex_dofs, edge_dofs=None):
    """`(direct, ends)`: the dofs of the nodes a rank writes (`node_layout`), from each
    local vertex's dofs (`vertex_dofs`, a row a vertex) and at degree 2 each local edge's
    (`edge_dofs`). `direct` holds a row a node with dofs of its own; a linear field on
    the quadratic nodes (`edge_dofs` None) has none at the edges, which take the mean of
    their ends' dofs, `ends` (two arrays of rows, else None)."""
    owned = layout['owned']
    direct, ends = [vertex_dofs[owned[0]]], None
    if len(owned) > 1 and edge_dofs is not None:
        direct.append(edge_dofs[owned[1]])
    elif len(owned) > 1:
        pairs = layout['ends'][owned[1]]
        ends = (vertex_dofs[pairs[:, 0]], vertex_dofs[pairs[:, 1]])
    return np.concatenate(direct), ends


def _node_layout(mesh, degree, cache):
    """`node_layout` of a dolfin mesh, kept in `cache` by (mesh id, degree); the
    coordinates are not, a mesh may move between writes (`node_positions`). dolfin
    numbers a cell's edges by the vertex pairs QUADRATIC names, which is checked on the
    local cells (`_check_midsides`). Collective (`owned_entities`)."""
    key = ('mesh', mesh.id(), degree)
    if key in cache:
        return cache[key]
    tdim = mesh.topology().dim()
    cells = _owned_cells(mesh)
    owned = owned_entities(mesh, 0)
    edges = None
    if degree == 2:
        mesh.init(1)
        mesh.init(tdim, 1)
        mesh.init(1, 0)
        mesh.init_global(1)
        ends = _connectivity(mesh, 1)
        cellname = mesh.ufl_cell().cellname()
        order = QUADRATIC[cellname][2]
        cell_edges = _connectivity(mesh, tdim, 1)[:len(cells)]
        if len(cells):
            vertices = np.asarray(mesh.coordinates())
            mid = 0.5 * (vertices[ends[:, 0]] + vertices[ends[:, 1]])
            _check_midsides(np.vstack((vertices, mid)),
                            np.hstack((cells, cell_edges + len(vertices)))[:, order],
                            cellname)
        edges = dict(gedge=mesh.topology().global_indices(1), cells=cell_edges, ends=ends,
                     owned=owned_entities(mesh, 1), n=mesh.num_entities_global(1),
                     order=order)
    layout = node_layout(mesh.mpi_comm(), mesh.topology().global_indices(0), cells, owned,
                         mesh.num_entities_global(0), mesh.num_entities_global(tdim), edges)
    cache[key] = layout
    return layout


def _node_dofs(V, mesh, degree, width, layout, cache):
    """Where each component's values at this rank's nodes sit in the local form of a V
    Function's vector -- owned dofs, then ghosts -- as a pair per component (`node_dofs`):
    the local dof of each node that has one of its own, and, for a linear field on the
    quadratic node set, the dofs at the two ends of each edge (its midside value is their
    mean, which is exact for a linear field), None otherwise.

    The dof is the one V's dofmap gives the entity, so a periodic V needs nothing
    else: a slave vertex or edge is mapped to its master's dof, whose value is the
    field's on either side of the period. That dof is in this rank's dofmap because
    the entity is in one of its cells (`owned_entities` picks among the ranks holding
    it in their own cells), so its value is either owned or one of the vector's ghosts.

    The components come from V's own dofmap, which lists an entity's dofs component by
    component -- its `width` columns are what `V.sub(c).dofmap()` gives -- rather than
    from the sub-dofmaps, each of which V keeps for its own lifetime with a copy of the
    cell table. A component of a mixed Function (`Function.split()`) is read the same
    way, its dofmap indexing the mixed vector. Kept in `cache` by (V's id, degree): a
    time series writes the same space every step."""
    key = ('space', V.id(), degree)
    if key in cache:
        return cache[key]
    linear = degree == 2 and V.ufl_element().degree() == 1

    def entity_dofs(dim):
        dofs = _entity_dofs(V.dofmap(), mesh, dim)
        if len(dofs) != width * mesh.num_entities(dim):
            raise ValueError('{} dofs on {} entities of dimension {}; a Lagrange field of '
                             '{} component(s) has one a component'.format(
                                 len(dofs), mesh.num_entities(dim), dim, width))
        return dofs.reshape(-1, width)

    direct, ends = node_dofs(layout, entity_dofs(0),
                             entity_dofs(1) if degree == 2 and not linear else None)
    takes = [(np.ascontiguousarray(direct[:, c]),
              None if ends is None else tuple(np.ascontiguousarray(e[:, c]) for e in ends))
             for c in range(width)]
    cache[key] = takes
    return takes


def _node_values(func, takes):
    """`func`'s values at this rank's nodes, from the local form of its vector (owned
    entries, then ghosts) at the dofs `_node_dofs` found: an array of one column per
    component, or a flat one for a scalar. Collective (the ghost update).

    The ghosts are brought up to date first -- a vector solved into by PETSc directly
    need not have them -- which sets each to its owner's value and changes nothing
    else. A vector with no ghosts at all (one assembled, then wrapped in a Function)
    cannot answer for a dof another rank owns; its values are then copied into a
    Function of the same space, which has them, rather than read wrong. Every rank
    takes the same branch (`agreed`), the copy being collective."""
    import dolfin as df
    from femtools.hdf5_rows import agreed
    vector = df.as_backend_type(func.vector())
    vector.update_ghost_values()
    reach = max([int(t.max()) for direct, ends in takes
                 for t in (direct,) + (ends or ()) if len(t)] + [-1])
    with vector.vec().localForm() as local:
        # a vector PETSc built without ghosts has no local form at all (a null handle,
        # on which getSize segfaults), dolfin's own have one, empty of ghosts or not
        short = not local.handle or reach >= local.getSize()
        if not agreed(func.function_space().mesh().mpi_comm(), short):
            values = local.getArray(readonly=True)
            out = [values[direct] if ends is None else np.concatenate(
                       (values[direct], 0.5 * (values[ends[0]] + values[ends[1]])))
                   for direct, ends in takes]
            return np.column_stack(out) if len(out) > 1 else out[0]
    V = func.function_space()
    if len(V.component()):
        raise ValueError('a component of a mixed Function whose vector holds no ghost '
                         'values; write its collapsed copy (split(deepcopy=True))')
    copy = df.Function(V)
    if agreed(V.mesh().mpi_comm(),
              func.vector().local_size() != copy.vector().local_size()):
        raise ValueError("the Function's vector does not hold its space's own dofs on "
                         'every rank, so its values cannot be matched to the dofmap')
    copy.vector().set_local(func.vector().get_local())
    copy.vector().apply('insert')
    return _node_values(copy, takes)


def _function_nodes_distributed(functions, cache=None):
    """The functions' node set -- the mesh's vertices, then (at degree 2) one per edge
    -- left where it already is: every rank returns only the nodes it owns, so nothing
    is gathered. Node ids are the global entity ids -- vertex index, then `n_vert +`
    edge index -- whose order is the file's row order (`XdmfDocument.rows`).

    A node is written by the rank that owns its entity (`owned_entities`), and its value
    is read straight off the field's own dofmap: the dof the space gives that vertex or
    edge, from the vector's local form, ghosts included (`_node_dofs`, `_node_values`).
    A linear field on the quadratic node set takes its midsides as the mean of the
    edge's ends. A periodic field needs nothing more: its slave entities share their
    masters' dofs, so both sides of the period are written from the same values, and
    nothing the size of a space is built.

    What depends on the mesh and the spaces alone is kept in `cache` (`_node_layout`,
    `_node_dofs`), so a step of a time series costs its values.
    """
    cache = {} if cache is None else cache
    mesh, degree, widths = _check_functions(functions)
    layout = _node_layout(mesh, degree, cache)
    fields = []
    for index, (func, width) in enumerate(zip(functions, widths)):
        takes = _node_dofs(func.function_space(), mesh, degree, width, layout, cache)
        fields.append((func.name() if func.name() else 'f{}'.format(index),
                       layout['rows'], _node_values(func, takes)))
    return {'n_nodes': layout['n_nodes'], 'n_cells': layout['n_cells'],
            'degree': degree, 'mesh': mesh, 'layout': layout,
            'conn': layout['conn'], 'cell_offset': layout['cell_offset'],
            'fields': fields}


def _block_of(path, item, comm):
    """This rank's contiguous block of the rows of the HDF5 array a DataItem names (all
    of them on one rank), as rows of its stated shape."""
    import h5py
    from femtools.hdf5_rows import block_edges
    fmt, ref, shape = item
    if fmt != 'HDF':
        return _load(path, item)        # refused there
    filename, _, group = ref.partition(':')
    full = os.path.join(os.path.dirname(os.path.abspath(path)), filename)
    rows = int(shape[0])
    width = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    size, rank = (1, 0) if comm is None else (comm.Get_size(), comm.Get_rank())
    edges = block_edges(rows, size)
    a, b = int(edges[rank]), int(edges[rank + 1])
    with h5py.File(full, 'r') as handle:
        d = handle[group.lstrip('/')]
        block = d[a:b] if d.ndim > 1 or width == 1 else d[a * width:b * width]
    return np.asarray(block).reshape((b - a,) + tuple(shape[1:]))


def read_nodes_at(path, grid, name, x, comm):
    """The values of the node attribute `name` of `grid` (a grid of `path`, `grids`) at
    the points `x` (rows of gdim coordinates), a row a point: each point's value is the
    one at the file's node within `1e-9` of the nodes' extent (at least 1) of it. Each
    rank reads one block of the file's nodes, and nodes and points meet by position
    (`hdf5_rows.match_nodes`), so nothing is read whole. Refused on every rank when the
    file's nodes leave the points' space or a point has no node. Collective on comm
    (None for one rank)."""
    from femtools.hdf5_rows import agreed, as_rows, extent, match_nodes
    x = as_rows(np.asarray(x, dtype=float))
    gdim = x.shape[1]
    coords = as_rows(np.asarray(_block_of(path, grid['geometry_item'], comm), dtype=float))
    values = as_rows(_block_of(path, grid['attributes'][name], comm))
    # a file of three coordinates on a 2D mesh: its z is zero, or it is another mesh
    extra = (np.abs(coords[:, gdim:]).max() if len(coords) and coords.shape[1] > gdim
             else 0.0)
    if agreed(comm, extra > 0.0):
        raise ValueError(
            "{}: the file's nodes leave the plane of this {}D mesh, so it is not the mesh "
            'the field was written on'.format(path, gdim))
    if coords.shape[1] < gdim:
        coords = np.hstack((coords, np.zeros((len(coords), gdim - coords.shape[1]))))
    coords = np.ascontiguousarray(coords[:, :gdim])
    lo, hi = extent(coords, comm, gdim)
    tol = 1e-9 * max(float(np.max(hi - lo)) if len(lo) else 0.0, 1.0)
    found, distance = match_nodes(comm, coords, values, x, tol, lo)
    if agreed(comm, bool(len(distance)) and float(distance.max()) > tol):
        raise ValueError(
            '{}: a point sits farther than {:.1e} from every node, so the mesh is not the '
            'one the field was written on'.format(path, tol))
    return found


def _scatter_nodes(target, space, path, grid, name, width, degree):
    """Puts the nodal values of grid's attribute `name` onto an unconstrained space's
    dofs, matched by position (`read_nodes_at`).

    The space is unconstrained, so its dof coordinates are distinct and a nearest
    match is unambiguous -- which is exactly what is *not* true of a periodic space,
    and why the caller interpolates from here rather than writing into one directly.
    Collective.
    """
    mesh = space.mesh()
    comm = mesh.mpi_comm() if mesh.mpi_comm().Get_size() > 1 else None
    # `dofmap().dofs()` numbers dofs globally, while `tabulate_dof_coordinates` and
    # `get_local` hold this rank's owned rows: the global numbers are made local
    first = space.dofmap().ownership_range()[0]
    gdim = mesh.geometry().dim()
    if width > 1:
        sub_dofs = [np.asarray(space.sub(c).dofmap().dofs()) - first
                    for c in range(width)]
        x = space.tabulate_dof_coordinates().reshape(-1, gdim)[sub_dofs[0]]
    else:
        sub_dofs = [np.asarray(space.dofmap().dofs()) - first]
        x = space.tabulate_dof_coordinates().reshape(-1, gdim)
    found = read_nodes_at(path, grid, name, x, comm)
    vector = target.vector()
    local = vector.get_local()
    for c, dofs in enumerate(sub_dofs):
        local[dofs] = found[:, c]
    vector.set_local(local)
    vector.apply('insert')


def _check_midsides(coords, conn, cellname='tetrahedron', sample=200):
    """Every midside node must lie on the edge it is named for.

    The node order is a convention -- dolfin's differs from XDMF's -- and getting it
    wrong writes cells that look plausible and are wrong. This is cheap and catches
    it, including after a distributed gather, where the numbering has been rebuilt.
    """
    corners, edges, _ = QUADRATIC[cellname]
    rng = np.random.RandomState(0)
    pick = conn[rng.choice(len(conn), min(sample, len(conn)), replace=False)]
    for k, (a, b) in enumerate(edges):
        off = np.abs(coords[pick[:, corners + k]]
                     - 0.5 * (coords[pick[:, a]] + coords[pick[:, b]])).max()
        edge = np.linalg.norm(coords[pick[:, a]] - coords[pick[:, b]], axis=1).max()
        if off > 1e-9 * edge:
            raise ValueError(
                'midside node {} sits {:.2e} off the edge it belongs to, so the '
                'node ordering is wrong'.format(corners + k, off))


# ---------------------------------------------------------------------------
# reading an XDMF back, with no dolfin
# ---------------------------------------------------------------------------

def _scan(path):
    """Every uniform grid of an XDMF file, in order, as dicts: 'topology' its type,
    'topology_item' and 'geometry_item' its arrays as (format, reference, shape),
    'attributes' each attribute's, 'centers' its Center, and 'time' (None if it has
    none). A grid that takes its mesh by xi:include, as the steps of dolfin's time series
    do, has the mesh of the grid before it; times listed once for a collection (older
    dolfin) go to its grids in order. The one parser of the XML here (`grids` is its
    view as HDF5 paths)."""
    root = ET.parse(path).getroot()
    parent = {c: p for p in root.iter() for c in p}
    out = []
    mesh = {'topology': None, 'topology_item': None, 'geometry_item': None}
    for grid in root.iter('Grid'):
        if grid.get('GridType') == 'Collection':
            continue
        # a Topology or Geometry holding no DataItem (an xi:include, or dolfin's
        # Reference="XML" to the first grid's) keeps the mesh before it
        topology, geometry = grid.find('Topology'), grid.find('Geometry')
        if topology is not None and topology.find('DataItem') is not None:
            mesh['topology'] = topology.get('TopologyType')
            mesh['topology_item'] = _item(topology.find('DataItem'))
        elif topology is not None and topology.get('TopologyType'):
            mesh['topology'] = topology.get('TopologyType')
        if geometry is not None and geometry.find('DataItem') is not None:
            mesh['geometry_item'] = _item(geometry.find('DataItem'))
        if mesh['topology_item'] is None or mesh['geometry_item'] is None:
            continue
        time = grid.find('Time')
        t = None if time is None or time.get('Value') is None else float(time.get('Value'))
        up = parent.get(grid)
        listed = None if up is None else up.find('Time/DataItem')
        if t is None and listed is not None:
            t = float(listed.text.split()[[g for g in up if g.tag == 'Grid'].index(grid)])
        attributes = grid.findall('Attribute')
        out.append(dict(mesh, time=t,
                        attributes={a.get('Name'): _item(a.find('DataItem'))
                                    for a in attributes},
                        centers={a.get('Name'): a.get('Center') for a in attributes}))
    return out


def grids(path):
    """Every uniform grid of an XDMF file whose arrays are in HDF5, in order, as dicts:
    'time' (None if it has none), 'geometry' and 'topology' as (HDF5 path, dataset),
    'attributes' each attribute's name to the same, 'centers' its Center (`_scan`).
    Raises ValueError for data held in the XML."""
    base = os.path.dirname(os.path.abspath(path))

    def h5(item, what):
        fmt, ref, _ = item
        if fmt != 'HDF':
            raise ValueError('{}: {} is not in an HDF5 file'.format(path, what))
        name, _, dataset = ref.partition(':')
        return os.path.join(base, name), dataset

    return [dict(time=g['time'], geometry=h5(g['geometry_item'], 'geometry'),
                 topology=h5(g['topology_item'], 'topology'),
                 attributes={n: h5(i, n) for n, i in g['attributes'].items()},
                 centers=g['centers'])
            for g in _scan(path)]


def continued(series, time):
    """Per-file lists of steps (each starting with its earliest), in the order the
    files were written, merged into one: a later list replaces the steps from its
    first time on, as a run restarted from a checkpoint continues the one before it,
    whose steps after the checkpoint belong to an abandoned history. `time(step)`
    gives a step's time."""
    steps = []
    for later in (s for s in series if s):
        t0 = time(later[0])
        steps = [s for s in steps if time(s) < t0 - 1e-9 * max(1., abs(t0))] + list(later)
    return steps


def merge_series(files):
    """Merge XDMF time series, given in the order they were written, into the first
    file: each file's steps replace the steps before it from its first time on
    (`continued`). The other .xdmf files are deleted; the .h5 files they refer to,
    which the merged series refers to, are kept. The files must refer to their HDF5
    data by paths that are valid from the first file, as files in one folder do."""
    ET.register_namespace('xi', XI)
    trees = [ET.parse(f) for f in files]
    collections = [tree.getroot().find('Domain').find('Grid') for tree in trees]
    steps = [[g for g in c if g.tag == 'Grid'] for c in collections]
    merged = continued(steps, lambda g: float(g.find('Time').get('Value')))
    for g in list(collections[0]):
        collections[0].remove(g)
    for g in merged:
        collections[0].append(g)
    trees[0].write(files[0], xml_declaration=True)
    for f in files[1:]:
        os.remove(f)


def parse_xdmf(xml_file, get_mesh_address=False):
    """partractools' `parse_xdmf` on `grids`: each grid's (time, [HDF5 file, dataset]) of
    its last attribute, and with get_mesh_address the [HDF5 file, dataset] of the first
    topology and geometry too. Files are joined onto the XDMF file's folder as given; a
    file of one grid, not a series, gives one step, at time None."""
    folder = os.path.dirname(xml_file)
    base = os.path.dirname(os.path.abspath(xml_file))

    def address(a):
        return None if a is None else [os.path.join(folder, os.path.relpath(a[0], base)), a[1]]

    steps = grids(xml_file)
    dsets = [(g['time'], address(list(g['attributes'].values())[-1] if g['attributes'] else None))
             for g in steps]
    if not get_mesh_address:
        return dsets
    topology = next((g['topology'] for g in steps if g['topology']), None)
    geometry = next((g['geometry'] for g in steps if g['geometry']), None)
    if topology is None or geometry is None:
        return dsets
    return dsets, address(topology), address(geometry)


def _item(node):
    """One DataItem as (format, reference, shape)."""
    if node is None:
        return None
    shape = tuple(int(n) for n in node.get('Dimensions').split())
    return (node.get('Format'), (node.text or '').strip(), shape)


def _load(path, item):
    """The array a DataItem names, from the sidecar HDF5 (the only place this module
    writes arrays: dolfin's ASCII encoding, which inlines them, is refused)."""
    import h5py
    fmt, ref, shape = item
    if fmt != 'HDF':
        raise ValueError('{}: an array held in the XML, which is not read here'.format(path))
    filename, _, group = ref.partition(':')
    full = os.path.join(os.path.dirname(os.path.abspath(path)), filename)
    with h5py.File(full, 'r') as handle:
        return np.asarray(handle[group.lstrip('/')])


def _read_grid_geometry(path, grid):
    """`(coords, conn)` of a grid: its nodes, two or three columns as stored (XY or
    XYZ), and its cells."""
    item = grid['geometry_item']
    coords = _load(path, item)
    conn = _load(path, grid['topology_item']).astype(np.int64)
    width = int(item[2][1]) if len(item[2]) > 1 else 3
    return coords.reshape(-1, width), conn.reshape(grid['topology_item'][2])


def _read_grid_attribute(path, grid, name):
    values = _load(path, grid['attributes'][name])
    return values.reshape(grid['attributes'][name][2])


def write_quadratic(path, coords, conn, values, names, cell_type='tetra10'):
    """Writes one XDMF holding the given cells and nodal fields.

    Goes through the same writer `XDMFFile2` uses, so the standalone route and the
    solver produce the same file layout for the same data rather than two dialects.
    """
    doc = XdmfDocument(path)
    grid = doc.grid('mesh')
    topology_type = MESHIO_TOPOLOGY.get(cell_type, cell_type)
    n_nodes = len(np.asarray(coords))
    doc.topology(grid, conn, topology_type, wide=is_wide(n_nodes),
                 ref=doc.dataset('Mesh/mesh/topology',
                                 topology_array(np.asarray(conn), n_nodes)))
    doc.geometry(grid, coords,
                 ref=doc.dataset('Mesh/mesh/geometry', np.asarray(coords, float)))
    for index, (name, value) in enumerate(zip(names, values)):
        value = _pad3(value)
        if np.asarray(value).ndim == 1:
            value = np.asarray(value).reshape(-1, 1)
        doc.attribute(grid, name, value, 'Node',
                      ref=doc.dataset('VisualisationVector/{}'.format(index), value))
    doc.close()


def write_quadratic_blocks(path, comm, cell_rows, conn, n_cells, node_rows, coords,
                           values, names, n_nodes, cell_type='tetra10'):
    """`write_quadratic` from ranks' shares, no rank holding the mesh: each dataset by
    `hdf5_rows.write_ordered` (the ranks' rows to the ranks whose blocks hold them, rank
    zero writing the compressed bytes -- serial h5py is enough), then rank zero writes the
    XML over them. `cell_rows`, `node_rows`: this rank's global rows of its cells and of
    the nodes it owns. The same file `write_quadratic` writes. Collective; what is
    refused -- a cell type, a field neither scalar, vector nor tensor, or rank zero
    failing to write the XML -- is refused on every rank."""
    from femtools.hdf5_rows import as_rows, write_ordered
    topology_type = MESHIO_TOPOLOGY.get(cell_type, cell_type)
    if topology_type not in TOPOLOGY_CELL:
        raise ValueError('{}: no XDMF topology {!r}'.format(path, cell_type))
    h5_name = os.path.splitext(os.path.basename(path))[0] + '.h5'
    h5_path = os.path.join(os.path.dirname(os.path.abspath(path)), h5_name)
    zero = {'partition': np.array([0], dtype=np.uint64)}
    write_ordered(h5_path, 'Mesh/mesh/topology', cell_rows, topology_array(conn, n_nodes),
                  n_cells, comm, mode='w', attrs=zero)
    write_ordered(h5_path, 'Mesh/mesh/geometry', node_rows, np.asarray(coords, float),
                  n_nodes, comm, attrs=zero)
    coords = np.asarray(coords, float)
    gdim = comm.allreduce(coords.shape[1] if coords.ndim == 2 and len(coords) else 0,
                          op=_mpi_max())
    values = [as_rows(np.asarray(_pad3(v), float)) for v in values]
    widths = [comm.allreduce(v.shape[1] if len(v) else 0, op=_mpi_max()) for v in values]
    odd = [w for w in widths if w not in (1, 3, 9)]
    if odd:
        raise ValueError('{}: {} components is not a Scalar, Vector or Tensor'.format(
            path, odd[0]))
    for index, value in enumerate(values):
        write_ordered(h5_path, 'VisualisationVector/{}'.format(index), node_rows, value,
                      n_nodes, comm, attrs=zero)
    failure = None
    if comm.Get_rank() == 0:
        try:
            doc = XdmfDocument(path)
            grid = doc.grid('mesh')
            doc.topology(grid, None, topology_type, wide=is_wide(n_nodes),
                         ref='{}:/Mesh/mesh/topology'.format(h5_name),
                         shape=(n_cells, TOPOLOGY_CELL[topology_type][1]))
            doc.geometry(grid, None, ref='{}:/Mesh/mesh/geometry'.format(h5_name),
                         shape=(n_nodes, gdim))
            for index, (name, width) in enumerate(zip(names, widths)):
                doc.attribute(grid, name, np.zeros((0, width)), 'Node',
                              ref='{}:/VisualisationVector/{}'.format(h5_name, index),
                              shape=(n_nodes, width))
            doc.close()
        except Exception as e:      # noqa: BLE001 -- every rank raises it below
            failure = '{}: {}: {}'.format(path, type(e).__name__, e)
    failure = comm.bcast(failure, root=0)
    if failure:
        raise RuntimeError('writing the XML failed on rank zero: ' + failure)


def _mpi_max():
    from mpi4py import MPI
    return MPI.MAX

