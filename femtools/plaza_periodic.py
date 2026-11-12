"""Plaza refinement that knows about periodic faces, in serial and under MPI.

    refine_periodic(mesh, markers, box, axes)   df.refine(mesh, markers), periodic-safe
    TooManyCells                                 raised by max_cells / max_rank_cells

`constrained_domain` pairs dofs by coordinate, so if a face of a periodic mesh is refined
and the face it is identified with is not, the slave dofs with no master are silently left
unconstrained. dolfin's Python `refine` cannot say which edges get bisected, and marking
partner cells from outside does not converge: Plaza's conformity propagation reaches the
other axes from the interior. So the identification is imposed inside the propagation:
this is dolfin 2019.2.0's own `face_long_edge`, `enforce_rules` and `do_refine`,
JIT-compiled through `dolfin.compile_cpp_code`, with the periodic correction folded into
`enforce_rules`:

    for each periodic partner pair (a, b), possibly on different ranks:
        if is_marked(a) != is_marked(b): mark both; ++update_count

so propagation and symmetry reach a joint fixed point. Everything it calls on
`ParallelRefinement` and `PlazaRefinementND::get_simplices` is public API; only
`face_long_edge`, `enforce_rules` and `do_refine` are private, which is why they are
reproduced here. A face's longest edge is chosen by position on ties, the same on both
faces of a pair and on every rank (dolfin breaks ties on global vertex indices, which the
two faces do not share). In 2D a triangle is split into four similar ones when its edges'
ratio allows (as dolfin's and DOLFINx's own refinement do), the test made with a relative
1e-10 so a right isosceles triangle does not hang on rounding.

Partners are paired without gathering a face anywhere: each rank sends each owned edge on
an identified face to a rank chosen by a hash of its periodic image, partners meet there
and are told of each other; every propagation round then exchanges the marks of the pairs
over a neighbourhood communicator. At one rank that is a local copy: serial and parallel
are one code path. The final assembly calls `ParallelRefinement::partition` on more than
one rank, as `PlazaRefinementND::refine` does.

The tie rule, the 2D ratio test, the pairing and the exchange are
femtools/cpp/plaza_periodic_rules.h, inlined here and shared with the DOLFINx port
(`femtools.dolfinx.refine`): the same mesh refined on either backend gives the same cells.

Parent facet markers are dropped; the parent-cell map is returned
(`femtools.hierarchical_transfer` is built on it).

No Python fallback: a compiler and `libdolfin-dev` are needed, and this raises without them.
"""
import os

import dolfin as df

from femtools.periodic_core import TooManyCells  # noqa: F401  (raised here)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp",
                       "plaza_periodic_rules.h")) as _f:
    _RULES = _f.read()

_CODE = r"""
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <limits>
#include <vector>
#include <map>
#include <cmath>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/common/Array.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/refinement/ParallelRefinement.h>
#include <dolfin/refinement/PlazaRefinementND.h>
#include <dolfin/common/MPI.h>

namespace py = pybind11;
using namespace dolfin;

""" + _RULES + r"""

using femtools_plaza::Box;
using femtools_plaza::Partners;

// dolfin's PlazaRefinementND::face_long_edge (private there), with the choice made the
// same on the two faces of a periodic pair. dolfin takes the strictly longest edge and
// breaks an exact tie on the global index of the opposite vertex (edge i is opposite
// vertex i). Both go wrong across a periodic pair, and did on the real packing
// (2026-09-29): an exact tie is broken by indices the two faces do not share, and two
// edges equal to the last bit come out in either order, since partner coordinates agree
// only to ~2e-16 -- so the faces bisected different edges and their triangulations
// drifted apart ("flips"), invisible to a vertex check until a flipped edge was refined.
// The shared rule (femtools_plaza::long_edge) ties edges within a relative 1e-10 of the
// longest and gives a tie to the largest periodic key of the opposite vertex. Also, in
// 2D, whether each triangle may be split uniformly (femtools_plaza::uniform_ok).
void face_long_edge(std::vector<unsigned int>& long_edge, std::vector<bool>& uniform,
                    const Mesh& mesh, const std::vector<double>& vx, const Box& box)
{
  mesh.init(1); mesh.init(2); mesh.init(2, 1);
  long_edge.resize(mesh.num_faces());
  uniform.assign(mesh.topology().dim() == 2 ? mesh.num_faces() : 0, false);
  std::vector<double> edge_length(mesh.num_edges());
  for (EdgeIterator e(mesh); !e.end(); ++e)
    edge_length[e->index()] = e->length();
  for (FaceIterator f(mesh); !f.end(); ++f)
  {
    const unsigned int* face_edges = f->entities(1);
    const unsigned int* face_verts = f->entities(0);
    const double len[3] = {edge_length[face_edges[0]], edge_length[face_edges[1]],
                           edge_length[face_edges[2]]};
    const double* x[3] = {&vx[3 * face_verts[0]], &vx[3 * face_verts[1]],
                          &vx[3 * face_verts[2]]};
    long_edge[f->index()] = face_edges[femtools_plaza::long_edge(len, x, box)];
    if (!uniform.empty())
      uniform[f->index()] = femtools_plaza::uniform_ok(len);
  }
}

// Every local vertex's position, padded to three components
std::vector<double> vertex_coordinates(const Mesh& mesh)
{
  const std::size_t gdim = mesh.geometry().dim();
  const std::vector<double>& x = mesh.coordinates();
  const std::size_t nv = mesh.num_vertices();
  std::vector<double> vx(3 * nv, 0.0);
  for (std::size_t v = 0; v < nv; ++v)
    for (std::size_t d = 0; d < gdim; ++d)
      vx[3 * v + d] = x[gdim * v + d];
  return vx;
}

// The periodic partners of the edges this rank owns (no rank sharing the edge is lower,
// dolfin's ownership), by the shared pairing. Marks reach the other copies of a shared
// edge through ParallelRefinement::update_logical_edgefunction.
Partners periodic_partners(const Mesh& mesh, const std::vector<double>& vx, const Box& box)
{
  const unsigned int rank = dolfin::MPI::rank(mesh.mpi_comm());
  std::map<std::int32_t, std::set<unsigned int>> none;
  if (dolfin::MPI::size(mesh.mpi_comm()) > 1)
    DistributedMeshTools::number_entities(mesh, 1);
  const auto& shared = dolfin::MPI::size(mesh.mpi_comm()) > 1
                       ? mesh.topology().shared_entities(1) : none;
  std::vector<std::int32_t> ids;
  std::vector<double> x;
  for (EdgeIterator e(mesh); !e.end(); ++e)
  {
    const unsigned int* ev = e->entities(0);
    double p[6];
    std::copy_n(&vx[3 * ev[0]], 3, p);
    std::copy_n(&vx[3 * ev[1]], 3, p + 3);
    if (!femtools_plaza::on_identified_plane(p, 2, box))
      continue;
    auto it = shared.find(e->index());
    if (it != shared.end() and *it->second.begin() < rank)
      continue;
    ids.push_back(e->index());
    x.insert(x.end(), p, p + 6);
  }
  return femtools_plaza::pair<2>(mesh.mpi_comm(), ids, x, box);
}

// dolfin's enforce_rules, alternated with the marks of the periodic partners, to a joint
// fixed point. At one rank the exchange is a local copy: serial and parallel are one
// code path.
void enforce_rules_periodic(ParallelRefinement& p_ref, const Mesh& mesh,
                            const std::vector<unsigned int>& long_edge,
                            const femtools_plaza::Exchange& exchange)
{
  auto is_marked = [&](std::int64_t e) { return p_ref.is_marked(e); };
  auto mark = [&](std::int64_t e) { p_ref.mark(e); };
  femtools_plaza::fixed_point(mesh.mpi_comm(), [&]() -> std::int64_t
  {
    std::int64_t n = 0;
    p_ref.update_logical_edgefunction();
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      const std::size_t long_e = long_edge[f->index()];
      if (p_ref.is_marked(long_e)) continue;
      bool any_marked = false;
      for (EdgeIterator e(*f); !e.end(); ++e)
        any_marked |= p_ref.is_marked(e->index());
      if (any_marked) { p_ref.mark(long_e); ++n; }
    }
    return n + femtools_plaza::partner_marks(exchange, is_marked, mark);
  });
}

// The edges no refinement may bisect, from `seeds` (a slab's inlet edges, say),
// closed under the longest-edge rule: a face whose long edge is frozen freezes
// its other two, since bisecting either would make enforce_rules bisect the long edge as
// well. Closed under shared edges between ranks (a second ParallelRefinement is the
// storage, its update_logical_edgefunction the exchange) and under periodic partners,
// as enforce_rules is, to a joint fixed point.
void freeze_closure(ParallelRefinement& f_ref, const Mesh& mesh,
                    const std::vector<std::size_t>& seeds,
                    const std::vector<unsigned int>& long_edge,
                    const femtools_plaza::Exchange& exchange)
{
  for (auto e : seeds) f_ref.mark(e);
  const MeshConnectivity& f2e = mesh.topology()(2, 1);
  auto face_edge = [&](std::int64_t f, int k) -> std::size_t { return f2e(f)[k]; };
  auto in = [&](std::int64_t e) { return f_ref.is_marked(e); };
  auto put = [&](std::int64_t e) { f_ref.mark(e); };
  femtools_plaza::fixed_point(mesh.mpi_comm(), [&]() -> std::int64_t
  {
    f_ref.update_logical_edgefunction();
    return femtools_plaza::close_under_long_edges(mesh.num_faces(), face_edge, long_edge,
                                                  in, put)
           + femtools_plaza::partner_marks(exchange, in, put);
  });
}

// A cell's local edges and faces, as femtools_plaza's loops take them (a face is the
// cell itself in 2D)
struct CellEntities
{
  const Mesh& mesh;
  const std::size_t tdim;
  std::size_t edge(std::int64_t c, int i) const
  { return mesh.topology()(tdim, 1)(c)[i]; }
  std::size_t face(std::int64_t c, int k) const
  { return tdim == 2 ? c : mesh.topology()(tdim, 2)(c)[k]; }
};

// dolfin's do_refine, minus parent-cell/parent-facet bookkeeping, dispatching to the
// parallel assembly (redistribute) rather than build_local when running on more than
// one rank -- exactly what unmodified PlazaRefinementND::refine does.
void do_refine(Mesh& new_mesh, const Mesh& mesh, ParallelRefinement& p_ref,
               const std::vector<unsigned int>& long_edge,
               const std::vector<bool>& uniform, bool redistribute,
               std::vector<std::size_t>& parent_cell)
{
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t num_cell_edges = tdim*3 - 3;
  const std::size_t num_cell_vertices = tdim + 1;

  p_ref.create_new_vertices();
  const std::map<std::size_t, std::size_t>& new_vertex_map
    = *(p_ref.edge_to_new_vertex());

  std::vector<std::size_t> indices(num_cell_vertices + num_cell_edges);
  std::vector<bool> markers;
  std::vector<std::size_t> longest_edge;
  std::vector<std::size_t> simplex_set;
  const CellEntities ce{mesh, tdim};
  auto cell_edge = [&](std::int64_t c, int i) { return ce.edge(c, i); };
  auto cell_face = [&](std::int64_t c, int k) { return ce.face(c, k); };
  auto marked = [&](std::size_t e) { return p_ref.is_marked(e); };

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    std::size_t j = 0;
    for (VertexIterator v(*cell); !v.end(); ++v) indices[j++] = v->global_index();

    if (!femtools_plaza::cell_split_input<std::size_t>(cell->index(), tdim, cell_edge,
                                                       cell_face, long_edge, marked,
                                                       markers, longest_edge))
    {
      p_ref.new_cell(*cell);
      parent_cell.push_back(cell->index());
    }
    else
    {
      indices.resize(num_cell_vertices + num_cell_edges);
      for (std::size_t p = 0; p < num_cell_edges; ++p)
        if (markers[p])
          indices[num_cell_vertices + p] = new_vertex_map.find(cell->entities(1)[p])->second;

      PlazaRefinementND::get_simplices(simplex_set, markers, longest_edge, tdim,
                                     tdim == 2 && uniform[cell->index()]);
      const std::size_t ncells = simplex_set.size() / num_cell_vertices;
      for (std::size_t i = 0; i != ncells; ++i)
        parent_cell.push_back(cell->index());
      for (auto &it : simplex_set) it = indices[it];
      p_ref.new_cells(simplex_set);
    }
  }

  const bool serial = (dolfin::MPI::size(mesh.mpi_comm()) == 1);
  if (serial)
    p_ref.build_local(new_mesh);
  else
    p_ref.partition(new_mesh, redistribute);

  // The parent map is meaningful under exactly dolfin's own condition -- its
  // do_refine stores "parent_cell" into new_mesh.data() when serial or not
  // redistributed, because redistribution scatters the new cells across ranks and
  // the local ordering built above no longer describes them. Cleared rather than
  // returned wrong.
  if (!serial && redistribute)
    parent_cell.clear();
}

// The cells do_refine would make, from the edges enforce_rules left marked -- its own
// loop, emitting nothing -- summed over ranks and on the most loaded rank (which sets a
// job's peak). Cheap beside the refinement it predicts (no vertices, no cells, no
// partition), so a refinement too large for what follows it can be refused before it
// is paid for.
std::pair<std::size_t, std::size_t> count_children(const Mesh& mesh, ParallelRefinement& p_ref,
                                                   const std::vector<unsigned int>& long_edge,
                                                   const std::vector<bool>& uniform)
{
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t num_cell_vertices = tdim + 1;
  std::vector<std::size_t> simplex_set;
  const CellEntities ce{mesh, tdim};
  const std::int64_t n = femtools_plaza::count_children<std::size_t>(
      mesh.num_cells(), tdim,
      [&](std::int64_t c, int i) { return ce.edge(c, i); },
      [&](std::int64_t c, int k) { return ce.face(c, k); }, long_edge,
      [&](std::size_t e) { return p_ref.is_marked(e); },
      [&](std::int64_t c, const std::vector<bool>& markers,
          const std::vector<std::size_t>& longest) -> std::int64_t
      {
        PlazaRefinementND::get_simplices(simplex_set, markers, longest, tdim,
                                         tdim == 2 && uniform[c]);
        return simplex_set.size() / num_cell_vertices;
      });
  return std::make_pair(dolfin::MPI::sum(mesh.mpi_comm(), n),
                        dolfin::MPI::max(mesh.mpi_comm(), n));
}

// max_cells > 0 or max_rank_cells > 0: count the children first, and if there would be
// more than that in all or on some rank, return (npair, no parent map, the count, the
// most loaded rank's count, ...) with new_mesh left empty. Both counts are reduced, so
// every rank takes the same branch.
//
// `frozen` (local edge indices, may be empty): edges never to bisect. Their closure is
// taken (freeze_closure), and a marked cell holding a frozen edge is left unmarked --
// which is enough: every edge enforce_rules marks is then a face's long edge in a face
// holding a marked, unfrozen edge, and a face with a frozen long edge has only frozen
// edges. Checked anyway, and thrown on. The tuple's last two entries are the cells so
// unmarked and the cells holding a frozen edge, both summed over ranks.
py::tuple refine_periodic(Mesh& new_mesh, const Mesh& mesh,
                          const MeshFunction<bool>& marker,
                          const std::vector<std::size_t>& axes,
                          const std::vector<double>& lower,
                          const std::vector<double>& box,
                          double tol, bool redistribute, std::size_t max_cells,
                          std::size_t max_rank_cells,
                          const std::vector<std::size_t>& frozen)
{
  Box pbox;
  for (std::size_t d = 0; d < 3; ++d)
  {
    pbox.lower[d] = d < lower.size() ? lower[d] : 0.0;
    pbox.upper[d] = d < box.size() ? box[d] : 0.0;
  }
  pbox.axes.assign(axes.begin(), axes.end());
  pbox.tol = tol;
  std::vector<unsigned int> long_edge;
  std::vector<bool> uniform;
  const std::vector<double> vx = vertex_coordinates(mesh);
  face_long_edge(long_edge, uniform, mesh, vx, pbox);
  const Partners partners = periodic_partners(mesh, vx, pbox);
  const femtools_plaza::Exchange exchange(mesh.mpi_comm(), partners);
  const std::int64_t npair = femtools_plaza::allsum(mesh.mpi_comm(), partners.local.size());

  ParallelRefinement p_ref(mesh);
  // the freeze is collective: every rank takes this branch or none does
  const bool freezing = dolfin::MPI::max(mesh.mpi_comm(), frozen.size()) > 0;
  std::size_t n_unmarked = 0, n_frozen_cells = 0;
  if (!freezing)
    p_ref.mark(marker);
  else
  {
    ParallelRefinement f_ref(mesh);
    freeze_closure(f_ref, mesh, frozen, long_edge, exchange);
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      bool holds = false;
      for (EdgeIterator e(*c); !e.end(); ++e) holds |= f_ref.is_marked(e->index());
      n_frozen_cells += holds;
      if (!marker[*c]) continue;
      if (holds) { ++n_unmarked; continue; }
      p_ref.mark(*c);
    }
    enforce_rules_periodic(p_ref, mesh, long_edge, exchange);
    std::size_t bad = 0;
    for (EdgeIterator e(mesh); !e.end(); ++e)
      bad += f_ref.is_marked(e->index()) && p_ref.is_marked(e->index());
    if (dolfin::MPI::sum(mesh.mpi_comm(), bad) > 0)
      throw std::runtime_error("refine_periodic: the marking reached a frozen edge");
    n_unmarked = dolfin::MPI::sum(mesh.mpi_comm(), n_unmarked);
    n_frozen_cells = dolfin::MPI::sum(mesh.mpi_comm(), n_frozen_cells);
  }
  if (!freezing)
    enforce_rules_periodic(p_ref, mesh, long_edge, exchange);
  std::vector<std::size_t> parent_cell;
  std::size_t n_children = 0, n_rank = 0;
  if (max_cells > 0 || max_rank_cells > 0)
  {
    const auto counts = count_children(mesh, p_ref, long_edge, uniform);
    n_children = counts.first;
    n_rank = counts.second;
    if ((max_cells > 0 && n_children > max_cells)
        || (max_rank_cells > 0 && n_rank > max_rank_cells))
      return py::make_tuple(npair, parent_cell, n_children, n_rank,
                            n_unmarked, n_frozen_cells);
  }
  do_refine(new_mesh, mesh, p_ref, long_edge, uniform, redistribute, parent_cell);
  // as a NumPy array over the vector itself, moved to the heap and freed with it: a
  // list of Python ints was ~40 B a new cell at the refinement's peak
  auto* held = new std::vector<std::size_t>(std::move(parent_cell));
  py::capsule free_it(held, [](void* p) { delete static_cast<std::vector<std::size_t>*>(p); });
  py::array_t<std::size_t> parents(held->size(), held->data(), free_it);
  return py::make_tuple(npair, parents, n_children, n_rank,
                        n_unmarked, n_frozen_cells);
}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("refine_periodic", &refine_periodic);
}
"""

_MODULE = [None]


def _cpp():
    """The compiled extension, built once; raises where it will not build."""
    if _MODULE[0] is None:
        _MODULE[0] = df.compile_cpp_code(_CODE)
    return _MODULE[0]


def require_cells_on_every_rank(mesh, what):
    """Raise on every rank if some rank holds no cells of `mesh`: dolfin's distributed
    refinement (`df.refine` too) numbers edges collectively through
    DistributedMeshTools, which a rank with an empty mesh skips, and the job hangs.
    Collective."""
    comm = mesh.mpi_comm()
    empty = comm.allreduce(int(mesh.num_cells() == 0))
    if empty:
        raise ValueError(
            "{}: {} of {} ranks hold no cells, and dolfin's refinement (df.refine's too) "
            "hangs there; run on fewer ranks, or on a mesh with cells on every rank "
            "(DOLFINx's refinement, femtools.dolfinx.refine, runs either way)".format(
                what, empty, comm.size))


def refine_periodic(mesh, markers, box, axes, redistribute=True, tol=1e-10,
                    max_cells=None, max_rank_cells=None, frozen=None, report=None,
                    lower=None):
    """`df.refine(mesh, markers)`, with the identified faces bisected alike.

    `box` is the periodic box's upper corner and `lower` its lower one
    (`periodic.box_lower`: the origin when None, which a mesh elsewhere is refused),
    `axes` the periodic axes; `tol` how near a face a point must be, relative to the box
    (`femtools.periodic_core.box_scale`: as it is for a unit box at the origin). 2D and
    3D. Works at any rank count. `redistribute=True` matches `df.refine`'s default; a
    pass that grows the mesh a few percent is much faster without the repartition, and
    gives the same mesh.

    Returns `(new_mesh, npair, parent_cell)`. `parent_cell` maps each new local cell to
    the old local cell it was cut from -- the map dolfin's own `do_refine` stores as
    "parent_cell", valid under dolfin's own condition: serial, or parallel without
    redistribution. After a redistribution the new cells are scattered across ranks and
    the map is meaningless, so it comes back as `None` on every rank.

    `max_cells`: count the children once the marking has propagated (exactly: do_refine's
    own loop, emitting nothing) and raise `TooManyCells` instead of refining when there
    would be more. `max_rank_cells` likewise for the most loaded rank's children, which
    is what sets a job's peak.

    `frozen`: local edge indices never to bisect, closed under the longest-edge rule,
    shared edges and periodic partners; a marked cell holding a frozen edge is left
    unmarked, and the refinement then never reaches one (checked, and raised on).
    Collective: give it on every rank, empty where a rank holds none. `report`, a dict,
    receives `unmarked` (the marked cells so left) and `frozen_cells` (the cells holding
    a frozen edge), summed over ranks.

    Every rank must hold cells (dolfin's refinement hangs otherwise): a rank holding none
    raises ValueError on every rank.
    """
    import numpy as np
    require_cells_on_every_rank(mesh, "refine_periodic")
    new_mesh = df.Mesh()
    cap, rank_cap = int(max_cells or 0), int(max_rank_cells or 0)
    if cap < 0 or rank_cap < 0:
        raise ValueError("max_cells and max_rank_cells must be positive, got {} and {}"
                         .format(max_cells, max_rank_cells))
    from femtools.hdf5_rows import agreed
    from femtools.periodic import box_lower, xyz
    from femtools.periodic_core import box_scale
    lower = box_lower(mesh, axes, lower)
    frozen = np.zeros(0, np.int64) if frozen is None else \
        np.asarray(frozen).ravel().astype(np.int64)
    mesh.init(1)
    if agreed(mesh.mpi_comm(), bool(len(frozen)) and (
            frozen.min() < 0 or frozen.max() >= mesh.num_edges())):
        raise ValueError("refine_periodic: frozen edge indices outside the local edges")
    npair, parent, n_children, n_rank, n_unmarked, n_frozen = _cpp().refine_periodic(
        new_mesh, mesh, markers, [int(a) for a in axes], [float(v) for v in lower],
        [float(b) for b in xyz(box)], float(tol) * box_scale(lower, xyz(box)),
        redistribute, cap, rank_cap, [int(e) for e in frozen])
    if report is not None:
        report.update(unmarked=int(n_unmarked), frozen_cells=int(n_frozen))
    if (cap and n_children > cap) or (rank_cap and n_rank > rank_cap):
        raise TooManyCells(n_children, cap, n_rank, rank_cap)
    # Whether the map survives is decided by `redistribute` and the rank count, both
    # of which every rank agrees on -- never by whether *this* rank's map came back
    # empty. A rank that owns no cells has an empty map and a perfectly valid one,
    # and testing emptiness would hand it None while its peers got arrays: callers
    # branch on this to choose between two collective paths, so a per-rank answer
    # deadlocks rather than failing. This mirrors do_refine's own clearing condition.
    if df.MPI.size(mesh.mpi_comm()) > 1 and redistribute:
        return new_mesh, npair, None
    parent_cell = np.asarray(parent, dtype=np.int64)
    if len(parent_cell) != new_mesh.num_cells():
        raise RuntimeError(
            "the parent map has {} entries for {} new cells; do_refine's clearing "
            "condition and this one have drifted apart".format(
                len(parent_cell), new_mesh.num_cells()))
    return new_mesh, npair, parent_cell
