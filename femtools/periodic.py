"""Periodic boxes: the constraint a periodic dofmap is built with, and checks on it.

    PeriodicBox(upper, tol, periodic, lower)    a SubDomain identifying opposite faces
    periodic_box(upper, tol, periodic, lower)   the same pairing, compiled
    periodic_box_from_bounds({axis: (lo, hi)})  that box, from the periodic axes' bounds
    faces_congruent(mesh, L, axes, lower=...)   whether identified faces carry the same nodes
    extent(mesh)                                the mesh's bounding box, over its ranks
    box_lower(mesh, axes, lower)                a box's lower corner, checked against the mesh
    cell_dofs(dofmap, cells, n)                 the first n dofs of many cells, one call
    connectivity(mesh, d0, d1, rows)            a topology connectivity as one array

The box runs from `lower` (default the origin) to `upper` along each axis; `periodic`
says which axes are identified (default all of them). A master is a point on a lower
periodic face and on no upper one, so a point on an edge or corner of the box is never
both; `map` takes every coordinate on an upper periodic face down by the box, so a corner
maps to the lower corner in one step. An axis that is not periodic takes no part in
either. `tol` is how near a face a point must be, `map_tol` dolfin's tolerance for
matching mapped coordinates (default `tol`). Works on 2D and 3D meshes.

`PeriodicBox` is the readable statement; `periodic_box` is the one to build spaces with:
the dofmap builder calls `inside` and `map` once per boundary entity, and through a
Python SubDomain each call crosses the pybind boundary. The two give identical dofmaps.
`PERIODIC_BOX_SOURCE` is the C++ class alone, for a module that embeds it in its own
translation unit (it needs <dolfin/common/Array.h> and <cmath>).

The compiled helpers need a compiler and `libdolfin-dev`, and raise without them.
"""

import dolfin as df
import numpy as np

# the backend-free half, re-exported
from femtools.periodic_core import on_planes, xyz  # noqa: F401

PERIODIC_BOX_SOURCE = r"""
class CppPeriodicBox : public SubDomain
{
public:
  CppPeriodicBox(std::vector<double> lower, std::vector<double> upper,
                 std::vector<bool> periodic, double tol, double map_tol)
    : SubDomain(map_tol), _lo(lower), _hi(upper), _tol(tol), _p(periodic) {}

  bool inside(const Array<double>& x, bool on_boundary) const override
  { return _inside(x.data(), x.size(), on_boundary); }
  bool inside(Eigen::Ref<const Eigen::VectorXd> x, bool on_boundary) const override
  { return _inside(x.data(), x.size(), on_boundary); }

  void map(const Array<double>& x, Array<double>& y) const override
  { _map(x.data(), y.data(), x.size()); }
  void map(Eigen::Ref<const Eigen::VectorXd> x,
           Eigen::Ref<Eigen::VectorXd> y) const override
  { _map(x.data(), y.data(), x.size()); }

private:
  bool _inside(const double* x, std::size_t n, bool on_boundary) const
  {
    bool lower = false, upper = false;
    for (std::size_t i = 0; i < n && i < _p.size(); ++i)
    {
      if (!_p[i]) continue;
      lower |= std::abs(x[i] - _lo[i]) <= _tol;
      upper |= std::abs(x[i] - _hi[i]) <= _tol;
    }
    return on_boundary && lower && !upper;
  }
  void _map(const double* x, double* y, std::size_t n) const
  {
    for (std::size_t i = 0; i < n; ++i)
      y[i] = (i < _p.size() && _p[i] && std::abs(x[i] - _hi[i]) <= _tol)
        ? x[i] - (_hi[i] - _lo[i]) : x[i];
  }
  std::vector<double> _lo, _hi; double _tol; std::vector<bool> _p;
};
"""


def _box(upper, periodic, lower):
    """(lower, upper, periodic) as lists of one length."""
    upper = [float(v) for v in upper]
    lower = [0.0] * len(upper) if lower is None else [float(v) for v in lower]
    periodic = [True] * len(upper) if periodic is None else [bool(p) for p in periodic]
    if not len(lower) == len(upper) == len(periodic):
        raise ValueError("periodic box: upper, lower and periodic of different lengths")
    return lower, upper, periodic


class PeriodicBox(df.SubDomain):
    """Identifies the upper faces of the box with the lower ones, axis by axis."""

    def __init__(self, upper, tol=df.DOLFIN_EPS_LARGE, periodic=None, lower=None,
                 map_tol=None):
        self.lower, self.upper, self.periodic = _box(upper, periodic, lower)
        self.tol = float(tol)
        self.axes = [i for i, p in enumerate(self.periodic) if p]
        super().__init__(self.tol if map_tol is None else float(map_tol))

    def inside(self, x, on_boundary):
        lower = any(abs(x[i] - self.lower[i]) <= self.tol for i in self.axes)
        upper = any(abs(x[i] - self.upper[i]) <= self.tol for i in self.axes)
        return bool(on_boundary and lower and not upper)

    def map(self, x, y):
        for i in range(len(x)):
            y[i] = (x[i] - (self.upper[i] - self.lower[i])
                    if i in self.axes and abs(x[i] - self.upper[i]) <= self.tol
                    else x[i])


_CODE = r"""
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dolfin/common/Array.h>
#include <pybind11/numpy.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshTopology.h>
#include <dolfin/mesh/MeshConnectivity.h>
#include <algorithm>
#include <cmath>
namespace py = pybind11;
using namespace dolfin;
""" + PERIODIC_BOX_SOURCE + r"""
// The first n local dofs of each listed cell, as one (cells, n) array: the pairing
// check's per-cell `cell_dofs` calls, without a pybind crossing each
py::array_t<la_index> cell_dofs_of(const GenericDofMap& dm,
                                   py::array_t<std::int64_t> cells, std::size_t n)
{
  auto c = cells.unchecked<1>();
  py::array_t<la_index> out({(std::size_t) c.shape(0), n});
  auto o = out.mutable_unchecked<2>();
  for (py::ssize_t k = 0; k < c.shape(0); ++k)
  {
    auto d = dm.cell_dofs(c(k));
    if ((std::size_t) d.size() < n)
      throw std::runtime_error("cell_dofs_of: a cell has fewer dofs than asked for");
    for (std::size_t i = 0; i < n; ++i)
      o(k, i) = d[i];
  }
  return out;
}

// The (d0, d1) connectivity of every local entity of dimension d0, k entries each, as one
// (entities, k) array: dolfin's Python hands it back as a list, ~36 B an entry
py::array_t<unsigned int> connectivity_of(const Mesh& mesh, std::size_t d0, std::size_t d1,
                                          std::size_t k)
{
  const MeshConnectivity& c = mesh.topology()(d0, d1);
  const std::size_t n = mesh.num_entities(d0);
  if (n && c.empty())
    throw std::runtime_error("connectivity_of: the connectivity is not built");
  if (c.size() != n * k)
    throw std::runtime_error("connectivity_of: not k entries an entity");
  py::array_t<unsigned int> out({n, k});
  if (n)
    std::copy(c().begin(), c().end(), out.mutable_data());
  return out;
}

// The same for the listed entities `rows` alone, (rows, k)
py::array_t<unsigned int> connectivity_rows_of(const Mesh& mesh, std::size_t d0,
                                               std::size_t d1, std::size_t k,
                                               py::array_t<std::int64_t> rows)
{
  const MeshConnectivity& c = mesh.topology()(d0, d1);
  const std::size_t n = mesh.num_entities(d0);
  if (n && c.empty())
    throw std::runtime_error("connectivity_rows_of: the connectivity is not built");
  if (c.size() != n * k)
    throw std::runtime_error("connectivity_rows_of: not k entries an entity");
  auto r = rows.unchecked<1>();
  py::array_t<unsigned int> out({(std::size_t) r.shape(0), k});
  auto o = out.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < r.shape(0); ++i)
  {
    if (r(i) < 0 || (std::size_t) r(i) >= n)
      throw std::out_of_range("connectivity_rows_of: an entity beyond the mesh's");
    const unsigned int* e = c(r(i));
    for (std::size_t j = 0; j < k; ++j)
      o(i, j) = e[j];
  }
  return out;
}

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<CppPeriodicBox, std::shared_ptr<CppPeriodicBox>, SubDomain>
    (m, "CppPeriodicBox", py::dynamic_attr())
    .def(py::init<std::vector<double>, std::vector<double>, std::vector<bool>, double,
                  double>());
  m.def("cell_dofs_of", &cell_dofs_of);
  m.def("connectivity_of", &connectivity_of);
  m.def("connectivity_rows_of", &connectivity_rows_of);
}
"""

_MODULE = [None]                 # the compiled module, built once


def _module():
    if _MODULE[0] is None:
        _MODULE[0] = df.compile_cpp_code(_CODE)
    return _MODULE[0]


def cell_dofs(dofmap, cells, n):
    """The first `n` local dofs of every cell in `cells`, as a (len(cells), n) int64
    array -- `dofmap.cell_dofs` for many cells in one call."""
    return np.asarray(_module().cell_dofs_of(
        dofmap, np.ascontiguousarray(cells, dtype=np.int64), int(n)), dtype=np.int64)


def connectivity(mesh, d0, d1, rows=None):
    """The d1-entities of every local d0-entity of a triangle or tetrahedral mesh
    (d0 > d1), or of the d0-entities `rows` alone, as an (entities, C(d0 + 1, d1 + 1))
    uint32 array of local indices, built first if need be. `mesh.topology()(d0, d1)()`
    is a Python list of them all, about 36 B an entry where this is 4."""
    from math import comb
    if not 0 <= d1 < d0 <= 3:
        raise ValueError("connectivity: d0 > d1, not ({}, {})".format(d0, d1))
    mesh.init(d0, d1)
    k = comb(d0 + 1, d1 + 1)
    if rows is None:
        return np.asarray(_module().connectivity_of(mesh, d0, d1, k))
    rows = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1)
    return np.asarray(_module().connectivity_rows_of(mesh, d0, d1, k, rows))


def periodic_box(upper, tol=df.DOLFIN_EPS_LARGE, periodic=None, lower=None, map_tol=None):
    """`PeriodicBox`, compiled: the same pairing without a pybind crossing per call.
    Carries `lower`, `upper`, `periodic` and `tol` as attributes, as `PeriodicBox` does
    (`tol` for the repair and the pairing check to judge the same nodes on a face)."""
    lower, upper, periodic = _box(upper, periodic, lower)
    box = _module().CppPeriodicBox(lower, upper, periodic, float(tol),
                                   float(tol if map_tol is None else map_tol))
    box.lower, box.upper, box.periodic, box.tol = lower, upper, periodic, float(tol)
    return box


def periodic_box_from_bounds(bounds, tol=1e-10, map_tol=1e-10):
    """`periodic_box` for the axes in `bounds`, which maps each periodic axis to its
    (lower, upper) bounds: {0: (0, Lx)} is periodic in x alone. The default `tol` is
    above the rounding of mesh generators (gmsh leaves points ~1e-14 off a face),
    `map_tol` dolfin's default for a SubDomain."""
    lower, upper = ([bounds[i][k] if i in bounds else 0. for i in range(3)] for k in (0, 1))
    return periodic_box(upper, tol, [i in bounds for i in range(3)], lower, map_tol=map_tol)


def extent(mesh):
    """`(lower, upper)`: the corners of the mesh's bounding box over all its ranks,
    `gdim` components each (`hdf5_rows.extent`)."""
    from femtools import hdf5_rows
    return hdf5_rows.extent(mesh.coordinates(), mesh.mpi_comm(), mesh.geometry().dim())


def box_lower(mesh, axes, lower=None, tol=1e-9):
    """The periodic box's lower corner, three components: `lower`, or the origin when
    it is None -- refused, on every rank, when the mesh does not reach the origin along
    a periodic axis, since a box elsewhere then needs its lower corner given (the
    upper faces would otherwise be compared with planes inside the mesh). Collective."""
    from femtools.periodic_core import lower_corner
    return lower_corner(None if lower is not None else extent(mesh), axes, lower, tol)


def faces_congruent(mesh, L, axes, tol=1e-9, edges=False, exempt=None, lower=None):
    """Whether the faces identified along `axes` still carry the same nodes.

    Returns `(ok, axis, n_lo, n_hi)`; on failure `axis` is the first one that does
    not match and the two counts are its deduplicated node totals, which is what a
    caller wants to put in its message.

    `constrained_domain` pairs dofs **by coordinate**, and a slave dof whose master
    is missing is simply left unconstrained -- dolfin builds the space, the solve
    runs, and periodicity is quietly gone along that face.

    `L` is the box's upper corner and `lower` its lower one (`box_lower`: the origin
    when None, which a mesh elsewhere is refused). 2D and 3D. Compared across ranks,
    never per-rank: after `ParallelRefinement` redistributes, a
    rank's local share need not hold both halves of any pair, so a per-rank answer is
    about the partition rather than the mesh. Each face node goes, by its rounded
    position, to the rank that key hashes to (`hdf5_rows.route`), where both sides are
    deduplicated (a rank-boundary vertex is ghosted onto several ranks) and compared;
    only the counts are summed over the ranks, so no rank holds a face.

    An identified face with no nodes at all fails: on a real box every periodic face
    carries some, so an empty one means the box or the axis is wrong, and comparing
    two empty sets would call that congruent.

    With `edges=True` the faces' edges are compared as well, by midpoint: two faces can
    carry the same vertices triangulated differently (a "flip"), which no vertex check
    sees and which breaks the pairing the first time a flipped edge is refined. A
    refinement asks this of its output; a solve, which only needs its P1 nodes paired,
    does not.

    `exempt` is the cut exception: when a mesh is one piece of a longer periodic one
    (a slab), a periodic pair can straddle the cut, leaving face nodes with no partner
    in this mesh -- correct there, because they are on the inlet or the outlet. It is a
    bool per local vertex, followed with `edges=True` by one per local edge (dolfin's
    edge numbering), or an (m, 3) array of exempt node positions (the cut facets'
    vertices and edge midpoints), which needs no connectivity built. A node without a
    partner passes when it is exempt and fails as before otherwise; a node with one is
    compared as always.
    """
    from femtools.periodic_core import faces_congruent_arrays
    lower = box_lower(mesh, axes, lower)
    e = connectivity(mesh, 1, 0) if edges else None
    return faces_congruent_arrays(mesh.mpi_comm(), mesh.coordinates(), e, L, axes, tol,
                                  exempt, lower)
