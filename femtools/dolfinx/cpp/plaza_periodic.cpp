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
//    periodic partner edges (one axis at a time, as plaza_periodic pairs them) to a
//    joint fixed point, so a face is bisected exactly when its partner is.
//  - frozen edges (never bisected, closed under the longest-edge rule, shared edges
//    and partners) and a count of the children before refining, both from
//    plaza_periodic.
//
// Partner edges are found without gathering a face on any rank: each rank sends each
// owned edge on an identified face to a rank chosen by a hash of its periodic image
// (edges near a bin boundary to the neighbouring bins too), partners meet there and are
// told of each other. The ranks to send to are found by DOLFINx's NBX graph discovery;
// marks then travel on a neighbourhood communicator over the partner ranks.

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

namespace nb = nanobind;
using namespace dolfinx;
using T = double;

namespace
{

std::int64_t allsum(MPI_Comm comm, std::int64_t v)
{
  std::int64_t out;
  MPI_Allreduce(&v, &out, 1, MPI_INT64_T, MPI_SUM, comm);
  return out;
}

std::int64_t allmax(MPI_Comm comm, std::int64_t v)
{
  std::int64_t out;
  MPI_Allreduce(&v, &out, 1, MPI_INT64_T, MPI_MAX, comm);
  return out;
}

struct Box
{
  std::array<double, 3> lower, upper;
  std::vector<int> axes;
  double tol;
};

// A position with each upper periodic plane mapped onto the lower one
std::array<double, 3> periodic_key(const double* x, const Box& box)
{
  std::array<double, 3> k = {x[0], x[1], x[2]};
  for (int a : box.axes)
    if (std::abs(k[a] - box.upper[a]) < box.tol)
      k[a] = box.lower[a];
  return k;
}

bool key_less(const std::array<double, 3>& p, const std::array<double, 3>& q, double tol)
{
  for (int d = 0; d < 3; ++d)
    if (std::abs(p[d] - q[d]) > tol)
      return p[d] < q[d];
  return false;
}

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

// DOLFINx's face_long_edge with plaza_periodic's tie rule (see the header)
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
    double s = 0;
    for (int d = 0; d < 3; ++d)
    {
      const double dd = vx[3 * v[0] + d] - vx[3 * v[1] + d];
      s += dd * dd;
    }
    length[e] = std::sqrt(s);
  }
  const std::int32_t nf = f_to_v->num_nodes();
  std::vector<std::int32_t> long_edge(nf);
  std::vector<std::int8_t> edge_ratio_ok;
  if (tdim == 2)
    edge_ratio_ok.resize(nf);
  const double min_ratio = std::sqrt(2.0) / 2.0;
  for (std::int32_t f = 0; f < nf; ++f)
  {
    auto edges = f_to_e->links(f);
    auto verts = f_to_v->links(f);   // edge i is opposite vertex i
    double max_len = 0, min_len = std::numeric_limits<double>::max();
    for (int i = 0; i < 3; ++i)
    {
      max_len = std::max(max_len, length[edges[i]]);
      min_len = std::min(min_len, length[edges[i]]);
    }
    int imax = 3;
    std::array<double, 3> best = {0, 0, 0};
    for (int i = 0; i < 3; ++i)
    {
      if (length[edges[i]] < max_len * (1.0 - 1e-10))
        continue;
      const auto k = periodic_key(&vx[3 * verts[i]], box);
      if (imax == 3 or key_less(best, k, box.tol))
      {
        imax = i;
        best = k;
      }
    }
    long_edge[f] = edges[imax];
    if (tdim == 2)
      edge_ratio_ok[f] = (min_len / max_len >= min_ratio);
  }
  return {std::move(long_edge), std::move(edge_ratio_ok)};
}

// Partners of the owned edges on identified faces: (local edge, partner rank, partner's
// local edge), both ways, one axis at a time.
struct Partners
{
  std::vector<std::int32_t> local;
  std::vector<int> rank;
  std::vector<std::int32_t> remote;
};

Partners periodic_partners(const mesh::Mesh<T>& mesh, const std::vector<double>& vx,
                           const Box& box)
{
  MPI_Comm comm = mesh.comm();
  const int size = dolfinx::MPI::size(comm), rank = dolfinx::MPI::rank(comm);
  auto topology = mesh.topology();
  auto e_to_v = topology->connectivity(1, 0);
  const std::int32_t n_owned = topology->index_map(1)->size_local();
  double extent = 0;
  for (int d = 0; d < 3; ++d)
    extent = std::max(extent, box.upper[d] - box.lower[d]);
  // the bins edges meet in: much wider than tol, small beside the box
  const double h = std::max(1e3 * box.tol, 1e-6 * extent);
  Partners out;

  for (int a : box.axes)
  {
    // (dest) -> records: role (0 slave, 1 master), local edge, 6 key coordinates
    std::map<int, std::vector<double>> send;
    for (std::int32_t e = 0; e < n_owned; ++e)
    {
      auto v = e_to_v->links(e);
      const double* x0 = &vx[3 * v[0]];
      const double* x1 = &vx[3 * v[1]];
      const bool up = std::abs(x0[a] - box.upper[a]) < box.tol
                      and std::abs(x1[a] - box.upper[a]) < box.tol;
      const bool lo = std::abs(x0[a] - box.lower[a]) < box.tol
                      and std::abs(x1[a] - box.lower[a]) < box.tol;
      if (!up and !lo)
        continue;
      std::array<double, 3> p = {x0[0], x0[1], x0[2]}, q = {x1[0], x1[1], x1[2]};
      if (up)
        p[a] = q[a] = box.lower[a];   // the slave's image, on the lower face
      if (key_less(q, p, box.tol))
        std::swap(p, q);
      // bins of the midpoint, and the neighbouring ones when within tol of a boundary
      std::array<std::vector<long>, 3> bins;
      for (int d = 0; d < 3; ++d)
      {
        const double m = 0.5 * (p[d] + q[d]) / h;
        const long b = static_cast<long>(std::floor(m));
        bins[d].push_back(b);
        const double frac = m - b;
        if (frac < box.tol / h)
          bins[d].push_back(b - 1);
        if (frac > 1.0 - box.tol / h)
          bins[d].push_back(b + 1);
      }
      std::set<int> dests;
      for (long b0 : bins[0])
        for (long b1 : bins[1])
          for (long b2 : bins[2])
          {
            std::uint64_t key = static_cast<std::uint64_t>(b0) * 73856093ULL
                                ^ static_cast<std::uint64_t>(b1) * 19349663ULL
                                ^ static_cast<std::uint64_t>(b2) * 83492791ULL;
            dests.insert(static_cast<int>(key % static_cast<std::uint64_t>(size)));
          }
      for (int dest : dests)
      {
        auto& s = send[dest];
        s.insert(s.end(), {up ? 0.0 : 1.0, static_cast<double>(e), p[0], p[1], p[2], q[0],
                           q[1], q[2]});
      }
    }

    // to the bin owners (NBX discovery), matched there, answered back
    std::vector<int> dest;
    for (auto& kv : send)
      dest.push_back(kv.first);
    std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
    std::ranges::sort(src);
    MPI_Comm nbr;
    MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                   dest.size(), dest.data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, false, &nbr);
    std::vector<int> scount, sdisp = {0};
    std::vector<double> sbuf;
    for (int d : dest)
    {
      scount.push_back(send[d].size());
      sbuf.insert(sbuf.end(), send[d].begin(), send[d].end());
      sdisp.push_back(sdisp.back() + scount.back());
    }
    std::vector<int> rcount(src.size()), rdisp(src.size() + 1, 0);
    MPI_Neighbor_alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, nbr);
    std::partial_sum(rcount.begin(), rcount.end(), rdisp.begin() + 1);
    std::vector<double> rbuf(rdisp.back());
    MPI_Neighbor_alltoallv(sbuf.data(), scount.data(), sdisp.data(), MPI_DOUBLE,
                           rbuf.data(), rcount.data(), rdisp.data(), MPI_DOUBLE, nbr);
    MPI_Comm_free(&nbr);

    // match: masters sorted by their first key point, each slave looked up within tol
    struct Rec { int rank; std::int32_t edge; std::array<double, 6> k; };
    std::vector<Rec> slaves, masters;
    for (std::size_t s = 0; s < src.size(); ++s)
      for (int i = rdisp[s]; i < rdisp[s + 1]; i += 8)
      {
        Rec r{src[s], static_cast<std::int32_t>(rbuf[i + 1]),
              {rbuf[i + 2], rbuf[i + 3], rbuf[i + 4], rbuf[i + 5], rbuf[i + 6], rbuf[i + 7]}};
        (rbuf[i] == 0.0 ? slaves : masters).push_back(r);
      }
    std::ranges::sort(masters, [](auto& x, auto& y) { return x.k[0] < y.k[0]; });
    // replies, by the rank they go to: (my local edge, partner rank, partner edge)
    std::map<int, std::vector<std::int64_t>> reply;
    std::set<std::pair<int, std::int32_t>> done;   // a pair met in several bins: once
    for (auto& s : slaves)
    {
      auto it = std::lower_bound(masters.begin(), masters.end(), s.k[0] - box.tol,
                                 [](const Rec& m, double v) { return m.k[0] < v; });
      for (; it != masters.end() and it->k[0] <= s.k[0] + box.tol; ++it)
      {
        bool same = true;
        for (int d = 0; d < 6; ++d)
          same &= std::abs(it->k[d] - s.k[d]) <= box.tol;
        if (!same)
          continue;
        if (done.insert({s.rank, s.edge}).second)
        {
          auto& rs = reply[s.rank];
          rs.insert(rs.end(), {s.edge, it->rank, it->edge});
          auto& rm = reply[it->rank];
          rm.insert(rm.end(), {it->edge, s.rank, s.edge});
        }
        break;
      }
    }
    std::vector<int> rdest;
    for (auto& kv : reply)
      rdest.push_back(kv.first);
    std::vector<int> rsrc = dolfinx::MPI::compute_graph_edges_nbx(comm, rdest);
    std::ranges::sort(rsrc);
    MPI_Dist_graph_create_adjacent(comm, rsrc.size(), rsrc.data(), MPI_UNWEIGHTED,
                                   rdest.size(), rdest.data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, false, &nbr);
    std::vector<int> c2, d2 = {0};
    std::vector<std::int64_t> b2;
    for (int d : rdest)
    {
      c2.push_back(reply[d].size());
      b2.insert(b2.end(), reply[d].begin(), reply[d].end());
      d2.push_back(d2.back() + c2.back());
    }
    std::vector<int> rc2(rsrc.size()), rd2(rsrc.size() + 1, 0);
    MPI_Neighbor_alltoall(c2.data(), 1, MPI_INT, rc2.data(), 1, MPI_INT, nbr);
    std::partial_sum(rc2.begin(), rc2.end(), rd2.begin() + 1);
    std::vector<std::int64_t> rb2(rd2.back());
    MPI_Neighbor_alltoallv(b2.data(), c2.data(), d2.data(), MPI_INT64_T, rb2.data(),
                           rc2.data(), rd2.data(), MPI_INT64_T, nbr);
    MPI_Comm_free(&nbr);
    std::int64_t n_slaves_here = 0;
    for (std::int32_t e = 0; e < n_owned; ++e)
    {
      auto v = e_to_v->links(e);
      n_slaves_here += std::abs(vx[3 * v[0] + a] - box.upper[a]) < box.tol
                       and std::abs(vx[3 * v[1] + a] - box.upper[a]) < box.tol;
    }
    // a pair met in bins on two ranks is answered twice: once is enough
    std::vector<std::array<std::int64_t, 3>> got;
    for (std::size_t i = 0; i < rb2.size(); i += 3)
      got.push_back({rb2[i], rb2[i + 1], rb2[i + 2]});
    std::ranges::sort(got);
    got.erase(std::unique(got.begin(), got.end()), got.end());
    std::int64_t n_slave_partners = 0;
    for (auto& g : got)
    {
      const std::int32_t e = g[0];
      out.local.push_back(e);
      out.rank.push_back(static_cast<int>(g[1]));
      out.remote.push_back(static_cast<std::int32_t>(g[2]));
      auto v = e_to_v->links(e);
      n_slave_partners += std::abs(vx[3 * v[0] + a] - box.upper[a]) < box.tol
                          and std::abs(vx[3 * v[1] + a] - box.upper[a]) < box.tol;
    }
    const std::int64_t missing = allsum(comm, n_slaves_here - n_slave_partners);
    if (missing != 0)
      throw std::runtime_error(
          "refine_periodic: edges on an upper periodic face have no partner on the lower "
          "one: the identified faces are not congruent");
    (void)rank;
  }
  return out;
}

// A neighbourhood communicator over the partner ranks, both ways (the list is symmetric)
MPI_Comm partner_comm(MPI_Comm comm, const Partners& p, std::vector<int>& ranks)
{
  ranks.assign(p.rank.begin(), p.rank.end());
  std::ranges::sort(ranks);
  ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
  MPI_Comm nbr;
  MPI_Dist_graph_create_adjacent(comm, ranks.size(), ranks.data(), MPI_UNWEIGHTED,
                                 ranks.size(), ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
                                 false, &nbr);
  return nbr;
}

// The partners' marks: for each partner of a marked local edge, the partner is marked;
// returns how many local edges this marked (owned ones; their ghosts follow through
// update_logical_edgefunction, whose sends are collected in `for_update`).
std::int64_t exchange_partner_marks(MPI_Comm nbr, const std::vector<int>& ranks,
                                    const Partners& p, std::span<std::int8_t> marks,
                                    const graph::AdjacencyList<int>& edge_ranks,
                                    std::vector<std::vector<std::int32_t>>& for_update)
{
  std::vector<std::vector<std::int32_t>> send(ranks.size());
  for (std::size_t i = 0; i < p.local.size(); ++i)
    if (marks[p.local[i]])
    {
      auto it = std::ranges::lower_bound(ranks, p.rank[i]);
      send[std::distance(ranks.begin(), it)].push_back(p.remote[i]);
    }
  std::vector<int> sc, sd = {0};
  std::vector<std::int32_t> sb;
  for (auto& s : send)
  {
    sc.push_back(s.size());
    sb.insert(sb.end(), s.begin(), s.end());
    sd.push_back(sd.back() + sc.back());
  }
  std::vector<int> rc(ranks.size()), rd(ranks.size() + 1, 0);
  MPI_Neighbor_alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, nbr);
  std::partial_sum(rc.begin(), rc.end(), rd.begin() + 1);
  std::vector<std::int32_t> rb(rd.back());
  MPI_Neighbor_alltoallv(sb.data(), sc.data(), sd.data(), MPI_INT32_T, rb.data(), rc.data(),
                         rd.data(), MPI_INT32_T, nbr);
  std::int64_t changed = 0;
  for (std::int32_t e : rb)
    if (!marks[e])
    {
      marks[e] = true;
      ++changed;
      for (int r : edge_ranks.links(e))
        for_update[r].push_back(e);
    }
  return changed;
}

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
  std::vector<int> pranks;
  MPI_Comm pcomm = partner_comm(comm, partners, pranks);
  const std::int32_t ne = map_e->size_local() + map_e->num_ghosts();
  auto c_to_e = topology->connectivity(tdim, 1);
  auto f_to_e = topology->connectivity(2, 1);
  auto c_to_f = topology->connectivity(tdim, 2);

  // frozen edges, closed under the longest-edge rule, shared edges and partners
  std::vector<std::int8_t> fz(ne, 0);
  const bool freezing = allmax(comm, (std::int64_t)frozen.size()) > 0;
  std::int64_t n_unmarked = 0, n_frozen_cells = 0;
  if (freezing)
  {
    std::vector<std::vector<std::int32_t>> upd(ranks.size());
    for (auto e : frozen)
      if (!fz[e])
      {
        fz[e] = 1;
        for (int r : edge_ranks.links(e))
          upd[r].push_back(e);
      }
    std::int64_t changed = 1;
    while (changed != 0)
    {
      refinement::update_logical_edgefunction(ncomm, upd, fz, *map_e);
      for (auto& u : upd)
        u.clear();
      changed = 0;
      for (std::int32_t f = 0; f < f_to_e->num_nodes(); ++f)
        if (fz[long_edge[f]])
          for (auto e : f_to_e->links(f))
            if (!fz[e])
            {
              fz[e] = 1;
              ++changed;
              for (int r : edge_ranks.links(e))
                upd[r].push_back(e);
            }
      changed += exchange_partner_marks(pcomm, pranks, partners, fz, edge_ranks, upd);
      refinement::update_logical_edgefunction(ncomm, upd, fz, *map_e);
      for (auto& u : upd)
        u.clear();
      changed = allsum(comm, changed);
    }
  }

  // the marking: the edges of the marked cells (less those holding a frozen edge)
  std::vector<std::int8_t> marked(ne, 0);
  std::vector<std::vector<std::int32_t>> upd(ranks.size());
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
      if (!marked[e])
      {
        marked[e] = 1;
        for (int r : edge_ranks.links(e))
          upd[r].push_back(e);
      }
  }
  refinement::update_logical_edgefunction(ncomm, upd, marked, *map_e);

  // DOLFINx's rules and the periodic partners, to a joint fixed point
  std::int64_t changed = 1;
  while (changed != 0)
  {
    refinement::plaza::impl::enforce_rules(ncomm, edge_ranks, marked, *topology, long_edge);
    for (auto& u : upd)
      u.clear();
    changed = exchange_partner_marks(pcomm, pranks, partners, marked, edge_ranks, upd);
    refinement::update_logical_edgefunction(ncomm, upd, marked, *map_e);
    changed = allsum(comm, changed);
  }
  MPI_Comm_free(&pcomm);

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
    std::int64_t n = 0;
    for (std::int32_t c = 0; c < n_cells; ++c)
    {
      auto edges = c_to_e->links(c);
      bool any = false;
      for (std::size_t ei = 0; ei < edges.size(); ++ei)
      {
        indices[ncv + ei] = marked[edges[ei]] ? ncv + ei : -1;
        any |= marked[edges[ei]];
      }
      if (!any)
      {
        ++n;
        continue;
      }
      std::vector<std::int32_t> le;
      for (auto f : c_to_f->links(c))
      {
        std::int32_t p = long_edge[f];
        for (std::size_t ej = 0; ej < edges.size(); ++ej)
          if (p == edges[ej])
          {
            p = ej;
            break;
          }
        le.push_back(p);
      }
      const bool uniform = (tdim == 2) ? edge_ratio_ok[c] : false;
      auto [set, sz] = refinement::plaza::impl::get_simplices(indices, le, tdim, uniform);
      n += sz / ncv;
    }
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
