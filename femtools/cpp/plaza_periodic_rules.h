// The periodic rules of femtools' Plaza refinement, shared by its dolfin module
// (femtools/plaza_periodic.py) and its DOLFINx one (femtools/dolfinx/cpp): one algorithm
// on both backends, from plain arrays and MPI, nothing of either library.
//
//  - periodic_key: a position with each upper periodic plane mapped onto the lower one,
//    the same for a point and its periodic partner and on every rank.
//  - long_edge: the longest edge of a face, ties (within a relative 1e-10) going to the
//    largest periodic key of the opposite vertex -- the same on the two faces of a pair.
//  - uniform_ok: in 2D, whether a triangle may be split into four similar ones (its
//    shortest edge at least sqrt(2)/2 of its longest, within a relative 1e-10, so a right
//    isosceles triangle does not hang on rounding).
//  - pair: periodic partners of items (edges: two points; nodes: one point) on the
//    identified faces, one axis at a time, without gathering a face anywhere: each item
//    goes to the rank its periodic image hashes to, partners meet there and are told of
//    each other. The ranks to send to are found by NBX (non-blocking consensus).
//  - Exchange: marks sent to partners over a neighbourhood communicator.
//
// Used by: femtools/plaza_periodic.py (inlined into the code dolfin compiles) and
// femtools/dolfinx/cpp/plaza_periodic.cpp (by CMake's include path); both hash it, so an
// edit here rebuilds both.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

#include <mpi.h>

namespace femtools_plaza
{

struct Box
{
  std::array<double, 3> lower, upper;
  std::vector<int> axes;
  double tol;
};

inline std::array<double, 3> periodic_key(const double* x, const Box& box)
{
  std::array<double, 3> k = {x[0], x[1], x[2]};
  for (int a : box.axes)
    if (std::abs(k[a] - box.upper[a]) < box.tol)
      k[a] = box.lower[a];
  return k;
}

inline bool key_less(const std::array<double, 3>& p, const std::array<double, 3>& q,
                     double tol)
{
  for (int d = 0; d < 3; ++d)
    if (std::abs(p[d] - q[d]) > tol)
      return p[d] < q[d];
  return false;
}

inline double distance(const double* p, const double* q)
{
  double s = 0;
  for (int d = 0; d < 3; ++d)
    s += (p[d] - q[d]) * (p[d] - q[d]);
  return std::sqrt(s);
}

// The face's long edge (0, 1 or 2; edge i opposite vertex i) from its edges' lengths
// and its vertices' positions
inline int long_edge(const double len[3], const double* const x[3], const Box& box)
{
  double max_len = std::max({len[0], len[1], len[2]});
  int imax = 3;
  std::array<double, 3> best = {0, 0, 0};
  for (int i = 0; i < 3; ++i)
  {
    if (len[i] < max_len * (1.0 - 1e-10))
      continue;
    const auto k = periodic_key(x[i], box);
    if (imax == 3 or key_less(best, k, box.tol))
    {
      imax = i;
      best = k;
    }
  }
  return imax;
}

// Whether all K points lie on one identified plane (upper or lower, of one axis): the
// items worth handing to pair()
inline bool on_identified_plane(const double* x, int K, const Box& box)
{
  for (int a : box.axes)
    for (double v : {box.lower[a], box.upper[a]})
    {
      bool all = true;
      for (int k = 0; k < K and all; ++k)
        all = std::abs(x[3 * k + a] - v) < box.tol;
      if (all)
        return true;
    }
  return false;
}

inline bool uniform_ok(const double len[3])
{
  const double max_len = std::max({len[0], len[1], len[2]});
  const double min_len = std::min({len[0], len[1], len[2]});
  return min_len >= max_len * (std::sqrt(2.0) / 2.0) * (1.0 - 1e-10);
}

inline std::int64_t allsum(MPI_Comm comm, std::int64_t v)
{
  std::int64_t out;
  MPI_Allreduce(&v, &out, 1, MPI_INT64_T, MPI_SUM, comm);
  return out;
}

inline std::int64_t allmax(MPI_Comm comm, std::int64_t v)
{
  std::int64_t out;
  MPI_Allreduce(&v, &out, 1, MPI_INT64_T, MPI_MAX, comm);
  return out;
}

// The ranks that send to this one, given the ranks it sends to (NBX: synchronous sends,
// probing, a non-blocking barrier once this rank's sends have been received)
inline std::vector<int> nbx(MPI_Comm comm_in, const std::vector<int>& dests)
{
  const int tag = 7391;
  MPI_Comm comm;
  MPI_Comm_dup(comm_in, &comm);
  std::vector<MPI_Request> reqs(dests.size());
  int one = 1;
  for (std::size_t i = 0; i < dests.size(); ++i)
    MPI_Issend(&one, 1, MPI_INT, dests[i], tag, comm, &reqs[i]);
  std::vector<int> sources;
  MPI_Request barrier = MPI_REQUEST_NULL;
  bool barrier_started = false;
  while (true)
  {
    int flag = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &flag, &status);
    if (flag)
    {
      int buf;
      MPI_Recv(&buf, 1, MPI_INT, status.MPI_SOURCE, tag, comm, MPI_STATUS_IGNORE);
      sources.push_back(status.MPI_SOURCE);
    }
    if (barrier_started)
    {
      int done = 0;
      MPI_Test(&barrier, &done, MPI_STATUS_IGNORE);
      if (done)
        break;
    }
    else
    {
      int sent = 0;
      MPI_Testall(reqs.size(), reqs.data(), &sent, MPI_STATUSES_IGNORE);
      if (sent)
      {
        MPI_Ibarrier(comm, &barrier);
        barrier_started = true;
      }
    }
  }
  MPI_Comm_free(&comm);
  std::sort(sources.begin(), sources.end());
  return sources;
}

// One exchange of variable-length buffers over a neighbourhood: dests and sources known
template <typename V>
std::vector<std::vector<V>> neighbour_exchange(MPI_Comm comm, const std::vector<int>& dests,
                                               const std::vector<int>& sources,
                                               const std::vector<std::vector<V>>& send,
                                               MPI_Datatype type)
{
  MPI_Comm nbr;
  MPI_Dist_graph_create_adjacent(comm, sources.size(), sources.data(), MPI_UNWEIGHTED,
                                 dests.size(), dests.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
                                 false, &nbr);
  std::vector<int> sc, sd = {0};
  std::vector<V> sb;
  for (auto& s : send)
  {
    sc.push_back(s.size());
    sb.insert(sb.end(), s.begin(), s.end());
    sd.push_back(sd.back() + sc.back());
  }
  std::vector<int> rc(sources.size()), rd(sources.size() + 1, 0);
  MPI_Neighbor_alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, nbr);
  std::partial_sum(rc.begin(), rc.end(), rd.begin() + 1);
  std::vector<V> rb(rd.back());
  MPI_Neighbor_alltoallv(sb.data(), sc.data(), sd.data(), type, rb.data(), rc.data(),
                         rd.data(), type, nbr);
  MPI_Comm_free(&nbr);
  std::vector<std::vector<V>> out(sources.size());
  for (std::size_t s = 0; s < sources.size(); ++s)
    out[s].assign(rb.begin() + rd[s], rb.begin() + rd[s + 1]);
  return out;
}

struct Partners
{
  std::vector<std::int32_t> local;
  std::vector<int> rank;
  std::vector<std::int32_t> remote;
};

// Periodic partners of items with K points (x: 3*K doubles an item), given by their
// local ids, one axis at a time: an item with all points on the upper plane of the axis
// pairs with the item at its image on the lower plane. Each item is to be given by one
// rank only (its owner). Returns the partners, both ways, each pair once. An item with
// none is left so: whether the faces are congruent is the caller's guard to judge (a
// slab's cut leaves face nodes without partners, rightly), not the pairing's.
template <int K>
Partners pair(MPI_Comm comm, const std::vector<std::int32_t>& ids,
              const std::vector<double>& x, const Box& box)
{
  int size;
  MPI_Comm_size(comm, &size);
  double extent = 0;
  for (int d = 0; d < 3; ++d)
    extent = std::max(extent, box.upper[d] - box.lower[d]);
  // the bins items meet in: much wider than tol, small beside the box
  const double h = std::max(1e3 * box.tol, 1e-6 * extent);
  constexpr int W = 2 + 3 * K;   // role (0 slave, 1 master), id, the image's K points
  Partners out;
  for (int a : box.axes)
  {
    std::map<int, std::vector<double>> send;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
      const double* p = &x[3 * K * i];
      bool up = true, lo = true;
      for (int k = 0; k < K; ++k)
      {
        up &= std::abs(p[3 * k + a] - box.upper[a]) < box.tol;
        lo &= std::abs(p[3 * k + a] - box.lower[a]) < box.tol;
      }
      if (!up and !lo)
        continue;
      std::vector<std::array<double, 3>> pts(K);
      for (int k = 0; k < K; ++k)
      {
        pts[k] = {p[3 * k], p[3 * k + 1], p[3 * k + 2]};
        if (up)
          pts[k][a] = box.lower[a];   // the slave's image, on the lower plane
      }
      std::sort(pts.begin(), pts.end(),
                [&](const std::array<double, 3>& u, const std::array<double, 3>& v)
                { return key_less(u, v, box.tol); });
      // bins of the centroid, and the neighbouring ones when within tol of a boundary
      std::array<std::vector<long>, 3> bins;
      for (int d = 0; d < 3; ++d)
      {
        double m = 0;
        for (int k = 0; k < K; ++k)
          m += pts[k][d];
        m = m / K / h;
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
            const std::uint64_t key = static_cast<std::uint64_t>(b0) * 73856093ULL
                                      ^ static_cast<std::uint64_t>(b1) * 19349663ULL
                                      ^ static_cast<std::uint64_t>(b2) * 83492791ULL;
            dests.insert(static_cast<int>(key % static_cast<std::uint64_t>(size)));
          }
      for (int dest : dests)
      {
        auto& s = send[dest];
        s.push_back(up ? 0.0 : 1.0);
        s.push_back(static_cast<double>(ids[i]));
        for (int k = 0; k < K; ++k)
          s.insert(s.end(), pts[k].begin(), pts[k].end());
      }
    }
    std::vector<int> dests;
    std::vector<std::vector<double>> bufs;
    for (auto& kv : send)
    {
      dests.push_back(kv.first);
      bufs.push_back(std::move(kv.second));
    }
    const std::vector<int> sources = nbx(comm, dests);
    const auto got = neighbour_exchange<double>(comm, dests, sources, bufs, MPI_DOUBLE);

    // match: masters sorted by their first key coordinate, each slave looked up within tol
    struct Rec
    {
      int rank;
      std::int32_t id;
      std::array<double, 3 * K> k;
    };
    std::vector<Rec> slaves, masters;
    for (std::size_t s = 0; s < sources.size(); ++s)
      for (std::size_t i = 0; i + W <= got[s].size(); i += W)
      {
        Rec r{sources[s], static_cast<std::int32_t>(got[s][i + 1]), {}};
        for (int d = 0; d < 3 * K; ++d)
          r.k[d] = got[s][i + 2 + d];
        (got[s][i] == 0.0 ? slaves : masters).push_back(r);
      }
    std::sort(masters.begin(), masters.end(),
              [](const Rec& u, const Rec& v) { return u.k[0] < v.k[0]; });
    std::map<int, std::vector<std::int64_t>> reply;   // (my id, partner rank, partner id)
    std::set<std::pair<int, std::int32_t>> done;      // a slave met in several bins: once
    for (const auto& s : slaves)
    {
      auto it = std::lower_bound(masters.begin(), masters.end(), s.k[0] - box.tol,
                                 [](const Rec& m, double v) { return m.k[0] < v; });
      for (; it != masters.end() and it->k[0] <= s.k[0] + box.tol; ++it)
      {
        bool same = true;
        for (int d = 0; d < 3 * K; ++d)
          same &= std::abs(it->k[d] - s.k[d]) <= box.tol;
        if (!same)
          continue;
        if (done.insert({s.rank, s.id}).second)
        {
          auto& rs = reply[s.rank];
          rs.insert(rs.end(), {s.id, it->rank, it->id});
          auto& rm = reply[it->rank];
          rm.insert(rm.end(), {it->id, s.rank, s.id});
        }
        break;
      }
    }
    std::vector<int> rdests;
    std::vector<std::vector<std::int64_t>> rbufs;
    for (auto& kv : reply)
    {
      rdests.push_back(kv.first);
      rbufs.push_back(std::move(kv.second));
    }
    const std::vector<int> rsources = nbx(comm, rdests);
    const auto answers
        = neighbour_exchange<std::int64_t>(comm, rdests, rsources, rbufs, MPI_INT64_T);
    // a pair met in bins on two ranks is answered twice: once is enough
    std::vector<std::array<std::int64_t, 3>> pairs;
    for (const auto& a2 : answers)
      for (std::size_t i = 0; i + 3 <= a2.size(); i += 3)
        pairs.push_back({a2[i], a2[i + 1], a2[i + 2]});
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    for (const auto& p : pairs)
    {
      out.local.push_back(static_cast<std::int32_t>(p[0]));
      out.rank.push_back(static_cast<int>(p[1]));
      out.remote.push_back(static_cast<std::int32_t>(p[2]));
    }
  }
  return out;
}

// Marks sent to periodic partners: each partner of a marked local item is marked
class Exchange
{
public:
  Exchange(MPI_Comm comm, const Partners& p) : _p(p)
  {
    _ranks.assign(p.rank.begin(), p.rank.end());
    std::sort(_ranks.begin(), _ranks.end());
    _ranks.erase(std::unique(_ranks.begin(), _ranks.end()), _ranks.end());
    MPI_Dist_graph_create_adjacent(comm, _ranks.size(), _ranks.data(), MPI_UNWEIGHTED,
                                   _ranks.size(), _ranks.data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, false, &_comm);
  }
  ~Exchange() { MPI_Comm_free(&_comm); }
  Exchange(const Exchange&) = delete;              // it owns a communicator
  Exchange& operator=(const Exchange&) = delete;

  // the local items whose partners are marked, by `is_marked(local id)`
  template <typename IsMarked>
  std::vector<std::int32_t> incoming(IsMarked is_marked) const
  {
    std::vector<std::vector<std::int32_t>> send(_ranks.size());
    for (std::size_t i = 0; i < _p.local.size(); ++i)
      if (is_marked(_p.local[i]))
      {
        auto it = std::lower_bound(_ranks.begin(), _ranks.end(), _p.rank[i]);
        send[std::distance(_ranks.begin(), it)].push_back(_p.remote[i]);
      }
    std::vector<int> sc, sd = {0};
    std::vector<std::int32_t> sb;
    for (auto& s : send)
    {
      sc.push_back(s.size());
      sb.insert(sb.end(), s.begin(), s.end());
      sd.push_back(sd.back() + sc.back());
    }
    std::vector<int> rc(_ranks.size()), rd(_ranks.size() + 1, 0);
    MPI_Neighbor_alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, _comm);
    std::partial_sum(rc.begin(), rc.end(), rd.begin() + 1);
    std::vector<std::int32_t> rb(rd.back());
    MPI_Neighbor_alltoallv(sb.data(), sc.data(), sd.data(), MPI_INT32_T, rb.data(), rc.data(),
                           rd.data(), MPI_INT32_T, _comm);
    return rb;
  }

private:
  const Partners& _p;
  std::vector<int> _ranks;
  MPI_Comm _comm;
};

} // namespace femtools_plaza
