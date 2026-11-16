"""A boundary facet file: facets, each with an integer marker, on the points they use.

    write(path, points, facets, markers, comm=None)
    points, facets, markers = read(path)

`path` (.xdmf) and the .h5 beside it, meshio's layout: one uniform grid, `points`
(float64, gdim a row), `facets` (indices into points, int32 where they fit, else int64:
triangles in 3D, segments in 2D) and `markers` (int64) as a 1-D cell attribute named
`markers`, every dataset compressed (`femtools.compress`). The layout partrac's
`wall_facets` reads, and `femtools-cut-facets` writes. numpy and h5py; mpi4py only where
a communicator is passed, dolfin never.

    write_exterior(path, facet_vertices, gvert, x, values, n_vertices, comm)
                                          a mesh's facets, numbered by global vertex
    facet_index, marks = match_exterior(path, x, facet_vertices, owned, comm)
                                          a file's markers on a mesh's exterior facets
    counts = written_marker_counts(path)  how many facets carry each marker, the markers
                                          read a block at a time
    BoundaryMarkers(mesh)                 a facet MeshFunction written and read as this file
                                          (femtools.boundary_markers; imports dolfin)
"""
import os

import numpy as np

XDMF = """<?xml version="1.0"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="Grid" GridType="Uniform">
      <Topology TopologyType="{top}" NumberOfElements="{nf}" NodesPerElement="{nv}">
        <DataItem Dimensions="{nf} {nv}" NumberType="Int" Precision="{ip}" Format="HDF">{h5}:/facets</DataItem>
      </Topology>
      <Geometry GeometryType="{geo}">
        <DataItem Dimensions="{np} {gd}" NumberType="Float" Precision="8" Format="HDF">{h5}:/points</DataItem>
      </Geometry>
      <Attribute Name="markers" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{nf}" NumberType="Int" Precision="8" Format="HDF">{h5}:/markers</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
TOPOLOGY = {3: "Triangle", 2: "Polyline"}
GEOMETRY = {3: "XYZ", 2: "XY"}


def h5_path(path):
    """The .h5 beside an .xdmf."""
    return os.path.splitext(path)[0] + ".h5"


def write(path, points, facets, markers, comm=None):
    """`path` and its .h5 from `points`, `facets` and `markers`. Under MPI each rank
    passes a block of rows of each array (any of them empty), and the file holds the
    blocks in rank order; `facets` index the whole points array. Collective; returns
    `(facets, points)` counts."""
    import h5py
    from femtools import compress, hdf5_rows
    points = np.asarray(points, dtype=np.float64)
    facets, markers = np.asarray(facets), np.asarray(markers, dtype=np.int64).ravel()
    size = 1 if comm is None else comm.Get_size()
    # an empty block may be 1-D: its width is any other rank's
    mine = (len(points), len(facets), facets.shape[1] if facets.ndim == 2 else 0,
            points.shape[1] if points.ndim == 2 else 0)
    shares = np.array([mine] if size == 1 else comm.allgather(mine), dtype=np.int64)
    n_points, n_facets = (int(v) for v in shares[:, :2].sum(axis=0))
    nv, gd = (int(v) for v in shares[:, 2:].max(axis=0))
    # each rank's first row of points and of facets
    first = np.cumsum(shares[:, :2], axis=0) - shares[:, :2]
    bad = len(facets) != len(markers) or nv not in TOPOLOGY or gd not in GEOMETRY
    if hdf5_rows.agreed(comm, bad):
        raise ValueError("{}: {} facets and {} markers, facets of {} points in {}D"
                         .format(path, len(facets), len(markers), nv, gd))
    wide = n_points > np.iinfo(np.int32).max
    facets = facets.astype(np.int64 if wide else np.int32)
    out = h5_path(path)
    if size == 1:
        with h5py.File(out, "w") as h5:
            compress.put(h5, "points", points)
            compress.put(h5, "facets", facets)
            compress.put(h5, "markers", markers)
    else:
        rank = comm.Get_rank()
        for k, (name, a, n, at) in enumerate((("points", points, n_points, first[rank, 0]),
                                              ("facets", facets, n_facets, first[rank, 1]),
                                              ("markers", markers, n_facets, first[rank, 1]))):
            hdf5_rows.write_ordered(out, name, at + np.arange(len(a)), a, n, comm,
                                    mode="w" if k == 0 else "a")
    from femtools.jit import on_rank0

    def write_xml():                   # beside and renamed: never a half-written file
        tmp = path + ".writing"
        with open(tmp, "w") as f:
            f.write(XDMF.format(top=TOPOLOGY[nv], nf=n_facets, nv=nv, ip=8 if wide else 4,
                                geo=GEOMETRY[gd], np=n_points, gd=gd,
                                h5=os.path.basename(out)))
        os.replace(tmp, path)
    # by rank 0, its error raised on every rank (and every rank waits for the file)
    on_rank0(comm if size > 1 else None, write_xml, "writing " + path)
    return n_facets, n_points


def write_exterior(path, facet_vertices, gvert, x, values, n_vertices, comm=None):
    """`path` and its .h5 (`write`) from a mesh's facets: `facet_vertices` each facet's
    local vertices (a row a facet, this rank's share of the facets), `gvert` and `x` every
    local vertex's global index and coordinates, `values` the facets' markers,
    `n_vertices` the global vertex count. The file's points are the facets' vertices, each
    once, in order of global index. Collective; returns `(facets, points)` counts."""
    from femtools import hdf5_rows
    facet_vertices = np.asarray(facet_vertices)
    used, inverse = np.unique(facet_vertices, return_inverse=True)
    glob = np.asarray(gvert, dtype=np.int64)[used]
    x = np.asarray(x)[used]
    if comm is None:
        order = np.argsort(glob)
        points, index = x[order], np.empty(len(glob), np.int64)
        index[order] = np.arange(len(glob))
    else:
        # each vertex once, at the rank whose block of global indices holds it
        edges = hdf5_rows.block_edges(n_vertices, comm.Get_size())
        routed = hdf5_rows.route(comm, hdf5_rows.owner_of(edges, glob, comm), glob, x)
        got, got_x = routed.received
        mine, first, where = np.unique(got, return_index=True, return_inverse=True)
        points = got_x[first]
        start = comm.exscan(len(mine)) or 0
        (index,) = hdf5_rows.reply(comm, routed, start + where.astype(np.int64))
    return write(path, points, index[inverse].reshape(facet_vertices.shape),
                 np.asarray(values, dtype=np.int64), comm)

# a file's point matches a mesh vertex within this much of the shortest exterior edge
REL_TOL = 1e-3


def _row_keys(rows):
    """Each row's sorted entries as one structured value, for sorting and matching."""
    rows = np.ascontiguousarray(np.sort(rows, axis=1).astype(np.int64))
    return rows.view([("", np.int64)] * rows.shape[1]).ravel()


def _in_boxes(pts, lo, hi):
    """`(facet, box)`: every pair of a facet `pts` (n, nv, gdim) and a box `lo[r]`,
    `hi[r]` (rows of (P, gdim); an empty box has lo > hi) holding all its points,
    ordered by box and then facet. The boxes are binned on a grid of about P cells over
    them, and each facet tested against the boxes of the cell its lowest corner is in
    alone: a box holding the facet holds that corner, so it is among them."""
    n, gdim = len(pts), pts.shape[2]
    full = np.flatnonzero((lo <= hi).all(axis=1))
    if not n or not len(full):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    low, high = pts.min(axis=1), pts.max(axis=1)        # each facet's bounding box
    g = max(1, int(round(len(full) ** (1.0 / gdim))))   # cells an axis
    origin = lo[full].min(axis=0)
    width = (hi[full].max(axis=0) - origin) / g
    width[width <= 0] = 1.0

    def cell(p):
        return np.clip(np.floor((p - origin) / width), 0, g - 1).astype(np.int64)
    first, last = cell(lo[full]), cell(hi[full])
    span = last - first + 1
    count = np.prod(span, axis=1)
    # each box in every cell its range covers, as (cell, box) sorted by cell
    box = np.repeat(full, count)
    k = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    at = np.zeros(len(box), np.int64)
    for a in range(gdim):
        s = np.repeat(span[:, a], count)
        at = at * g + np.repeat(first[:, a], count) + k % s
        k //= s
    order = np.argsort(at, kind="stable")
    at, box = at[order], box[order]
    starts = np.searchsorted(at, np.arange(g ** gdim + 1))
    # each facet against its corner's cell's boxes
    c = cell(low) @ (g ** np.arange(gdim)[::-1])
    many = starts[c + 1] - starts[c]
    facet = np.repeat(np.arange(n), many)
    cand = box[np.arange(int(many.sum())) - np.repeat(np.cumsum(many) - many, many)
               + np.repeat(starts[c], many)]
    inside = ((low[facet] >= lo[cand]) & (high[facet] <= hi[cand])).all(axis=1)
    facet, cand = facet[inside], cand[inside]
    order = np.lexsort((facet, cand))
    return facet[order].astype(np.int64), cand[order].astype(np.int64)


def match_exterior(path, x, facet_vertices, owned, comm=None):
    """`(facet_index, marks)`: which of a mesh's local exterior facets the facet file
    `path` holds, matched by vertex coordinates, and their markers. `x` are the local
    vertices' coordinates (a row each), `facet_vertices` the local exterior facets (ghosts
    included) as rows of local vertices, `owned` whether this rank answers for each;
    `facet_index` indexes their rows.

    No rank reads the file whole: each reads one block of the facets and their markers,
    fetches those facets' points (`hdf5_rows.fetch_rows`), and sends each facet to every
    rank whose exterior vertices' bounding box holds all its points (the boxes binned,
    `_in_boxes`), which matches it there. Refuses, on every rank, a negative marker and
    a facet that matches no owned exterior facet. Collective on comm (None for one
    rank)."""
    import h5py
    from scipy.spatial import cKDTree
    from femtools import hdf5_rows
    size, rank = (1, 0) if comm is None else (comm.Get_size(), comm.Get_rank())
    x = np.asarray(x, dtype=float)
    vertices = np.asarray(facet_vertices, dtype=np.int64)
    owned = np.asarray(owned, dtype=bool)
    nv = vertices.shape[1]
    gdim = x.shape[1]
    near = np.inf
    if len(vertices):
        near = min(np.linalg.norm(x[vertices[:, a]] - x[vertices[:, b]], axis=1).min()
                   for a in range(nv) for b in range(a + 1, nv))
    from femtools.hdf5_rows import allreduce
    tol = REL_TOL * allreduce(comm, float(near), "min")       # one tolerance, every rank's
    local = np.unique(vertices)
    where = datasets(path)
    if where[2] is None:
        raise ValueError(path + ": no markers")
    with h5py.File(where[1][0], "r") as h5:
        n_facets, n_file = h5[where[1][1].lstrip("/")].shape
    with h5py.File(where[0][0], "r") as h5:
        width = h5[where[0][1].lstrip("/")].shape[1]
    if hdf5_rows.agreed(comm, n_file != nv):
        raise ValueError("{}: facets of {} points, the mesh's of {}".format(path, n_file, nv))
    # this rank's block of the facets and markers, and their points' coordinates
    edges = hdf5_rows.block_edges(n_facets, size)
    a, b = int(edges[rank]), int(edges[rank + 1])
    with h5py.File(where[1][0], "r") as fh, h5py.File(where[2][0], "r") as mh:
        given = np.asarray(fh[where[1][1].lstrip("/")][a:b], dtype=np.int64).reshape(-1, nv)
        marks = np.asarray(mh[where[2][1].lstrip("/")][a:b]).ravel().astype(np.int64)
    pts = hdf5_rows.fetch_rows(where[0][0], where[0][1].lstrip("/"), given.ravel(), comm)
    pts = np.asarray(pts, float).reshape(len(given), nv, width)[:, :, :gdim]
    if comm is not None:
        box = ((x[local].min(axis=0) - tol, x[local].max(axis=0) + tol) if len(local)
               else (np.full(gdim, np.inf), np.full(gdim, -np.inf)))
        boxes = comm.allgather(box)
        take, dest = _in_boxes(pts, np.array([b[0] for b in boxes], dtype=float),
                               np.array([b[1] for b in boxes], dtype=float))
        pts, marks = hdf5_rows.route(comm, dest,
                                     pts[take].reshape(len(take), nv * gdim),
                                     marks[take]).received
        pts = pts.reshape(len(pts), nv, gdim)
    # each received facet's points matched to this rank's exterior vertices; a facet
    # all of whose points match is a candidate for one of its exterior facets
    rows = np.zeros((0, nv), np.int64)
    if len(local) and len(pts):
        dist, k = cKDTree(x[local]).query(pts.reshape(-1, gdim), distance_upper_bound=tol)
        dist, k = dist.reshape(len(pts), nv), k.reshape(len(pts), nv)
        full = np.isfinite(dist).all(axis=1)
        rows, marks = local[k[full]], marks[full]
    else:
        marks = marks[:0]
    mine = _row_keys(vertices)
    order = np.argsort(mine)
    theirs = _row_keys(rows)
    pos = np.minimum(np.searchsorted(mine[order], theirs), max(len(mine) - 1, 0))
    hit = (mine[order][pos] == theirs) if len(mine) else np.zeros(len(rows), bool)
    at = order[pos[hit]]
    if hdf5_rows.agreed(comm, bool((marks < 0).any())):
        raise ValueError(path + ": a negative marker")
    matched = int(owned[at].sum())
    if comm is not None:
        matched = comm.allreduce(matched)
    if matched != n_facets:
        raise ValueError("{}: {} of its {} facets match an exterior facet of the mesh"
                         .format(path, matched, n_facets))
    return at, marks[hit]

def __getattr__(name):
    if name == "BoundaryMarkers":
        from femtools.boundary_markers import BoundaryMarkers
        return BoundaryMarkers
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def datasets(xdmf):
    """`(points, facets, markers)` of the first grid of a facet file as (HDF5 path,
    dataset); `markers` the cell attribute named so, else the only cell attribute, else
    None."""
    from femtools.xdmf import grids
    found = grids(xdmf)
    if not found:
        raise ValueError(xdmf + ": no grid")
    grid = found[0]
    cells = [n for n in grid["attributes"] if (grid["centers"][n] or "Cell") == "Cell"]
    if "markers" in cells:
        name = "markers"
    elif len(cells) == 1:
        name = cells[0]
    elif cells:
        raise ValueError("{} cell attributes, none named markers".format(len(cells)))
    else:
        name = None
    return (grid["geometry"], grid["topology"],
            None if name is None else grid["attributes"][name])


def read(xdmf):
    """`(points, facets, markers)` of the first grid of a facet file; `markers` the cell
    attribute named so, else the only cell attribute, else None. Reads this module's
    files and any facet XDMF over HDF5, such as a facet MeshFunction dolfin writes."""
    import h5py

    def load(where):
        with h5py.File(where[0], "r") as h5:
            return np.asarray(h5[where[1].lstrip("/")])
    points, facets, markers = datasets(xdmf)
    points, facets = load(points), load(facets)
    return points, facets, None if markers is None else load(markers).ravel()



def written_marker_counts(path, chunk=1 << 24):
    """How many facets a facet file gives each marker (`np.bincount` of its markers).

    Reads the `markers` array alone, straight from its HDF5 dataset, a block at a time:
    `meshio.read` brings in the points and the facets as well, and a facet file may carry
    every vertex of the mesh (a Hele-Shaw mesher's does), so on a large mesh that is the
    volume's coordinates on one rank to count a surface. Anything but XDMF over HDF5 goes
    through meshio.
    """
    import xml.etree.ElementTree as ET
    item = None
    for attr in ET.parse(path).getroot().iter("Attribute"):
        if attr.get("Name") == "markers":
            item = attr.find("DataItem")
    if item is None or item.get("Format") != "HDF":
        import meshio
        written = np.asarray(meshio.read(path).cell_data_dict["markers"]["triangle"])
        return np.bincount(written.astype(np.int64).ravel())
    import h5py
    fname, _, dataset = item.text.strip().partition(":")
    counts = np.zeros(0, dtype=np.int64)
    with h5py.File(os.path.join(os.path.dirname(path), fname), "r") as h5:
        data = h5[dataset]
        for start in range(0, data.shape[0], chunk):
            block = np.bincount(np.asarray(data[start:start + chunk]).astype(np.int64).ravel())
            if len(block) > len(counts):
                counts = np.pad(counts, (0, len(block) - len(counts)))
            counts[:len(block)] += block
    return counts
