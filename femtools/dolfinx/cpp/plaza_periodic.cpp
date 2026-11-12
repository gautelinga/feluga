// femtools.dolfinx.refine's C++: DOLFINx's Plaza refinement with periodic faces bisected
// alike -- femtools.plaza_periodic (for dolfin) ported onto DOLFINx's own refinement.
//
// What DOLFINx does, kept: the marking is propagated by its enforce_rules and
// update_logical_edgefunction, the new vertices and cells are made by its
// compute_refinement, the mesh by create_mesh with its identity partitioner (children
// stay on their parent's rank) or, asked to redistribute, its default partitioner -- so
// this scales as dolfinx.mesh.refine does.
//
// What is changed, as in plaza_periodic:
//  - face_long_edge: edges within a relative 1e-10 of the longest are tied, and a tie
//    goes to the largest periodic key of the opposite vertex (its position with each
//    upper periodic plane mapped onto the lower one): the same on the two faces of a
//    periodic pair and on every rank, which DOLFINx's tie-break on global vertex
//    indices is not.
//  - the propagation: DOLFINx's enforce_rules alternated with an exchange over
//    periodic partner edges (every axis paired in one round, as plaza_periodic pairs
//    them) to a
//    joint fixed point, so a face is bisected exactly when its partner is.
//  - frozen edges (never bisected, closed under the longest-edge rule, shared edges
//    and partners) and a count of the children before refining, both from
//    plaza_periodic.
//
//  - in 2D, a triangle is split uniformly when its edges' ratio passes within a relative
//    1e-10 (DOLFINx's test is exact, so a right isosceles triangle hangs on rounding).
//
// The tie rule, the 2D ratio test, the partner pairing and the mark exchange are
// femtools/cpp/plaza_periodic_rules.h, shared with the dolfin module: one algorithm on
// both backends, the same cells from the same mesh.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <mpi.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/graphbuild.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/refinement/plaza.h>
#include <dolfinx/refinement/refine.h>
#include <dolfinx/refinement/utils.h>

#include <dolfinx/common/version.h>

#include "plaza_periodic_rules.h"

// Upgrading DOLFINx: this calls internals with no stability promise --
// refinement::plaza::impl::{enforce_rules, compute_refinement, get_simplices} and
// refinement::update_logical_edgefunction -- and its child count copies
// compute_refinement's loop over cells. On a new version, compare their signatures and
// that loop (dolfinx/refinement/plaza.h) with the calls here, run
// tests/test_dolfinx_refine.py and tests/test_plaza_periodic.py on 1 and 3 ranks, and
// only then admit the version below. femtools/dolfinx/refine.py names the installed
// version when the build fails.
#if !(DOLFINX_VERSION_MAJOR == 0 && DOLFINX_VERSION_MINOR == 11)
#error "femtools plaza_periodic.cpp is checked against DOLFINx 0.11 only: it calls refinement::plaza::impl internals (see the comment above this line)"
#endif

namespace nb = nanobind;
using namespace dolfinx;
using T = double;

namespace
{

using femtools_plaza::allmax;
using femtools_plaza::allsum;
using femtools_plaza::Box;
using femtools_plaza::Partners;

// Every local vertex's coordinates (owned and ghost), through the geometry
std::vector<double> vertex_coordinates(const mesh::Mesh<T>& mesh)
{
  auto topology = mesh.topology();
  const int tdim = topology->dim();
  auto map_v = topology->index_map(0);
  const std::int32_t nv = map_v->size_local() + map_v->num_ghosts();
  std::vector<std::int32_t> vertices(nv);
  std::iota(vertices.begin(), vertices.end(), 0);
  mesh.topology_mutable()->create_connectivity(0, tdim);
  auto [nodes, shape] = mesh::entities_to_geometry(mesh, 0, vertices, false);
  std::span<const T> x = mesh.geometry().x();
  std::vector<double> vx(3 * nv);
  for (std::int32_t v = 0; v < nv; ++v)
    for (int d = 0; d < 3; ++d)
      vx[3 * v + d] = x[3 * nodes[v] + d];
  return vx;
}

// DOLFINx's face_long_edge with the shared tie rule, and the 2D uniform-split test
std::pair<std::vector<std::int32_t>, std::vector<std::int8_t>>
face_long_edge(const mesh::Mesh<T>& mesh, const std::vector<double>& vx, const Box& box)
{
  auto topology = mesh.topology();
  const int tdim = topology->dim();
  auto f_to_v = topology->connectivity(2, 0);
  auto f_to_e = topology->connectivity(2, 1);
  auto e_to_v = topology->connectivity(1, 0);
  auto map_e = topology->index_map(1);
  const std::int32_t ne = map_e->size_local() + map_e->num_ghosts();
  std::vector<double> length(ne);
  for (std::int32_t e = 0; e < ne; ++e)
  {
    auto v = e_to_v->links(e);
    length[e] = femtools_plaza::distance(&vx[3 * v[0]], &vx[3 * v[1]]);
  }
  const std::int32_t nf = f_to_v->num_nodes();
  std::vector<std::int32_t> long_edge(nf);
  std::vector<std::int8_t> edge_ratio_ok;
  if (tdim == 2)
    edge_ratio_ok.resize(nf);
  for (std::int32_t f = 0; f < nf; ++f)
  {
    auto edges = f_to_e->links(f);
    auto verts = f_to_v->links(f);   // edge i is opposite vertex i
    const double len[3] = {length[edges[0]], length[edges[1]], length[edges[2]]};
    const double* x[3] = {&vx[3 * verts[0]], &vx[3 * verts[1]], &vx[3 * verts[2]]};
    long_edge[f] = edges[femtools_plaza::long_edge(len, x, box)];
    if (tdim == 2)
      edge_ratio_ok[f] = femtools_plaza::uniform_ok(len);
  }
  return {std::move(long_edge), std::move(edge_ratio_ok)};
}

// Partners of the owned edges on identified faces (the shared pairing)
Partners periodic_partners(const mesh::Mesh<T>& mesh, const std::vector<double>& vx,
                           const Box& box)
{
  auto topology = mesh.topology();
  auto e_to_v = topology->connectivity(1, 0);
  const std::int32_t n_owned = topology->index_map(1)->size_local();
  std::vector<std::int32_t> ids;
  std::vector<double> x;
  for (std::int32_t e = 0; e < n_owned; ++e)
  {
    auto v = e_to_v->links(e);
    double p[6];
    std::copy_n(&vx[3 * v[0]], 3, p);
    std::copy_n(&vx[3 * v[1]], 3, p + 3);
    if (!femtools_plaza::on_identified_plane(p, 2, box))
      continue;
    ids.push_back(e);
    x.insert(x.end(), p, p + 6);
  }
  return femtools_plaza::pair<2>(mesh.comm(), ids, x, box);
}

// Edge marks as femtools_plaza's loops take them: `in(e)`, and `put(e)`, which marks
// an edge and collects its sharing ranks' updates for update_logical_edgefunction
struct Marks
{
  std::vector<std::int8_t>& marks;
  const graph::AdjacencyList<int>& edge_ranks;
  std::vector<std::vector<std::int32_t>>& upd;
  bool in(std::int32_t e) const { return marks[e] != 0; }
  void put(std::int32_t e)
  {
    marks[e] = 1;
    for (int r : edge_ranks.links(e))
      upd[r].push_back(e);
  }
};

} // namespace
// The refinement (see the header). `cells`: local cells to refine (each marks its
// edges); `frozen`: local edges never to bisect. Returns (the new mesh, its parent
// cells, partner-pair count, children summed and on the most loaded rank -- 0 unless
// counted --, marked cells left unmarked by the freeze, cells holding a frozen edge,
// whether the mesh was made).
using Ints = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>;
using IntsOut = nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>;

IntsOut as_array(std::vector<std::int32_t>&& v)
{
  auto* held = new std::vector<std::int32_t>(std::move(v));
  nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<std::int32_t>*>(p); });
  return IntsOut(held->data(), {held->size()}, owner);
}

std::tuple<std::optional<mesh::Mesh<T>>, IntsOut, std::int64_t, std::int64_t,
           std::int64_t, std::int64_t, std::int64_t>
refine_periodic_nb(const mesh::Mesh<T>& mesh, Ints cells_in, std::array<double, 3> lower,
                   std::array<double, 3> upper, std::vector<int> axes, double tol,
                   bool redistribute, std::int64_t max_cells, std::int64_t max_rank_cells,
                   Ints frozen_in);

std::tuple<std::optional<mesh::Mesh<T>>, std::vector<std::int32_t>, std::int64_t,
           std::int64_t, std::int64_t, std::int64_t, std::int64_t>
refine_periodic(const mesh::Mesh<T>& mesh, std::span<const std::int32_t> cells,
                std::array<double, 3> lower, std::array<double, 3> upper,
                std::vector<int> axes, double tol, bool redistribute,
                std::int64_t max_cells, std::int64_t max_rank_cells,
                std::span<const std::int32_t> frozen)
{
  auto topology = mesh.topology_mutable();
  const int tdim = topology->dim();
  if (topology->cell_type() != mesh::CellType::triangle
      and topology->cell_type() != mesh::CellType::tetrahedron)
    throw std::runtime_error("refine_periodic: triangles and tetrahedra only");
  topology->create_entities(1);
  topology->create_entities(2);
  topology->create_connectivity(2, 1);
  topology->create_connectivity(2, 0);
  topology->create_connectivity(1, 0);
  topology->create_connectivity(1, tdim);
  topology->create_connectivity(tdim, 1);
  topology->create_connectivity(tdim, 2);
  MPI_Comm comm = mesh.comm();
  Box box{lower, upper, axes, tol};
  const std::vector<double> vx = vertex_coordinates(mesh);
  auto [long_edge, edge_ratio_ok] = face_long_edge(mesh, vx, box);
  Partners partners = periodic_partners(mesh, vx, box);
  const std::int64_t npair = allsum(comm, (std::int64_t)partners.local.size());

  // DOLFINx's sharing ranks and neighbourhood (as compute_refinement_data makes them)
  auto map_e = topology->index_map(1);
  graph::AdjacencyList<int> edge_ranks = map_e->index_to_dest_ranks();
  std::vector<int> ranks(edge_ranks.array().begin(), edge_ranks.array().end());
  std::ranges::sort(ranks);
  ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
  std::ranges::transform(edge_ranks.array(), edge_ranks.array().begin(),
                         [&ranks](auto r)
                         { return std::distance(ranks.begin(), std::ranges::lower_bound(ranks, r)); });
  MPI_Comm ncomm;
  MPI_Dist_graph_create_adjacent(comm, ranks.size(), ranks.data(), MPI_UNWEIGHTED,
                                 ranks.size(), ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
                                 false, &ncomm);
  const femtools_plaza::Exchange exchange(comm, partners);
  const std::int32_t ne = map_e->size_local() + map_e->num_ghosts();
  auto c_to_e = topology->connectivity(tdim, 1);
  auto f_to_e = topology->connectivity(2, 1);
  auto c_to_f = topology->connectivity(tdim, 2);

  auto sync = [&](std::vector<std::vector<std::int32_t>>& upd, std::vector<std::int8_t>& m)
  {
    refinement::update_logical_edgefunction(ncomm, upd, m, *map_e);
    for (auto& u : upd)
      u.clear();
  };

  // frozen edges, closed under the longest-edge rule, shared edges and partners
  std::vector<std::int8_t> fz(ne, 0);
  const bool freezing = allmax(comm, (std::int64_t)frozen.size()) > 0;
  std::int64_t n_unmarked = 0, n_frozen_cells = 0;
  if (freezing)
  {
    std::vector<std::vector<std::int32_t>> upd(ranks.size());
    Marks F{fz, edge_ranks, upd};
    auto in = [&F](std::int32_t e) { return F.in(e); };
    auto put = [&F](std::int32_t e) { F.put(e); };
    for (auto e : frozen)
      if (!fz[e])
        put(e);
    femtools_plaza::fixed_point(comm, [&]() -> std::int64_t
    {
      sync(upd, fz);
      std::int64_t n = femtools_plaza::close_under_long_edges(
          f_to_e->num_nodes(), [&](std::int64_t f, int k) { return f_to_e->links(f)[k]; },
          long_edge, in, put);
      n += femtools_plaza::partner_marks(exchange, in, put);
      sync(upd, fz);
      return n;
    });
  }

  // the marking: the edges of the marked cells (less those holding a frozen edge)
  std::vector<std::int8_t> marked(ne, 0);
  std::vector<std::vector<std::int32_t>> upd(ranks.size());
  Marks M{marked, edge_ranks, upd};
  const std::int32_t n_cells = topology->index_map(tdim)->size_local();
  std::vector<std::int8_t> holds(n_cells, 0);
  if (freezing)
    for (std::int32_t c = 0; c < n_cells; ++c)
    {
      for (auto e : c_to_e->links(c))
        holds[c] |= fz[e];
      n_frozen_cells += holds[c];
    }
  for (auto c : cells)
  {
    if (holds[c])
    {
      ++n_unmarked;
      continue;
    }
    for (auto e : c_to_e->links(c))
      if (!M.in(e))
        M.put(e);
  }
  refinement::update_logical_edgefunction(ncomm, upd, marked, *map_e);

  // DOLFINx's rules and the periodic partners, to a joint fixed point
  femtools_plaza::fixed_point(comm, [&]() -> std::int64_t
  {
    refinement::plaza::impl::enforce_rules(ncomm, edge_ranks, marked, *topology, long_edge);
    for (auto& u : upd)
      u.clear();
    const std::int64_t n = femtools_plaza::partner_marks(
        exchange, [&M](std::int32_t e) { return M.in(e); },
        [&M](std::int32_t e) { M.put(e); });
    refinement::update_logical_edgefunction(ncomm, upd, marked, *map_e);
    return n;
  });

  if (freezing)
  {
    std::int64_t bad = 0;
    for (std::int32_t e = 0; e < ne; ++e)
      bad += fz[e] and marked[e];
    if (allsum(comm, bad) > 0)
    {
      MPI_Comm_free(&ncomm);
      throw std::runtime_error("refine_periodic: the marking reached a frozen edge");
    }
    n_unmarked = allsum(comm, n_unmarked);
    n_frozen_cells = allsum(comm, n_frozen_cells);
  }

  // the children, counted first if asked (compute_refinement's own loop, emitting nothing)
  std::int64_t n_children = 0, n_rank = 0;
  if (max_cells > 0 or max_rank_cells > 0)
  {
    const int nce = 3 * tdim - 3, ncv = tdim + 1;
    std::vector<std::int64_t> indices(ncv + nce);
    std::iota(indices.begin(), indices.end(), 0);
    const std::int64_t n = femtools_plaza::count_children<std::int32_t>(
        n_cells, tdim, [&](std::int64_t c, int i) { return c_to_e->links(c)[i]; },
        [&](std::int64_t c, int k) { return c_to_f->links(c)[k]; }, long_edge,
        [&](std::int32_t e) { return marked[e] != 0; },
        [&](std::int64_t c, const std::vector<bool>& markers,
            const std::vector<std::int32_t>& le) -> std::int64_t
        {
          for (int ei = 0; ei < nce; ++ei)
            indices[ncv + ei] = markers[ei] ? ncv + ei : -1;
          const bool uniform = (tdim == 2) ? edge_ratio_ok[c] : false;
          auto [set, sz] = refinement::plaza::impl::get_simplices(indices, le, tdim, uniform);
          return sz / ncv;
        });
    n_children = allsum(comm, n);
    n_rank = allmax(comm, n);
    if ((max_cells > 0 and n_children > max_cells)
        or (max_rank_cells > 0 and n_rank > max_rank_cells))
    {
      MPI_Comm_free(&ncomm);
      return {std::nullopt, {}, npair, n_children, n_rank, n_unmarked, n_frozen_cells};
    }
  }

  // DOLFINx's refinement of the marked edges, with these long edges, and its mesh
  auto [cell_adj, coords, xshape, parent_cell, parent_facet]
      = refinement::plaza::impl::compute_refinement(
          ncomm, marked, edge_ranks, mesh, long_edge, edge_ratio_ok,
          refinement::Option::parent_cell);
  MPI_Comm_free(&ncomm);
  mesh::CellPartitionFunction partitioner
      = redistribute ? mesh::create_cell_partitioner(mesh::GhostMode::none, 2)
                     : refinement::create_identity_partitioner(mesh, parent_cell.value());
  mesh::Mesh<T> mesh1 = mesh::create_mesh(comm, comm, cell_adj.array(),
                                          mesh.geometry().cmaps().front(), comm, coords,
                                          xshape, partitioner, 2);
  std::vector<std::int32_t> parents;
  if (!redistribute or dolfinx::MPI::size(comm) == 1)
    parents = std::move(parent_cell.value());
  return {std::move(mesh1), std::move(parents), npair, n_children, n_rank, n_unmarked,
          n_frozen_cells};
}

std::tuple<std::optional<mesh::Mesh<T>>, IntsOut, std::int64_t, std::int64_t,
           std::int64_t, std::int64_t, std::int64_t>
refine_periodic_nb(const mesh::Mesh<T>& mesh, Ints cells_in, std::array<double, 3> lower,
                   std::array<double, 3> upper, std::vector<int> axes, double tol,
                   bool redistribute, std::int64_t max_cells, std::int64_t max_rank_cells,
                   Ints frozen_in)
{
  auto [m, parents, npair, nc, nr, nu, nf] = refine_periodic(
      mesh, std::span(cells_in.data(), cells_in.size()), lower, upper, axes, tol,
      redistribute, max_cells, max_rank_cells,
      std::span(frozen_in.data(), frozen_in.size()));
  return {std::move(m), as_array(std::move(parents)), npair, nc, nr, nu, nf};
}

NB_MODULE(plaza_periodic_x, m)
{
  m.def("refine_periodic", &refine_periodic_nb);
}
