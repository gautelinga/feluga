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
//    identified faces, every axis in one round, without gathering a face anywhere: each
//    item goes to the rank its periodic image's bin hashes to (splitmix64), partners
//    meet there -- found through the bins, O(n log n) whatever the axis -- and are told
//    of each other. The ranks to send to are found by NBX (non-blocking consensus).
//
// `Box::tol` is a length: the callers make it relative to the box (femtools'
// `periodic_core.box_tol`), so a box far from the origin keeps its pairs.
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
#include <utility>
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

// splitmix64's finaliser folded over the integers of a key, then scaled onto `size`
// ranks: femtools.hdf5_rows.rank_of's hash, so structured faces (bins sharing factors
// of two) spread evenly over a power-of-two number of ranks
template <std::size_t N>
int rank_of(const std::array<std::int64_t, N>& key, int size)
{
  std::uint64_t h = 0;
  for (std::int64_t c : key)
  {
    h ^= static_cast<std::uint64_t>(c);
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
    h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
    h ^= h >> 31;
  }
  return static_cast<int>(((h >> 32) * static_cast<std::uint64_t>(size)) >> 32);
}

// The bin of a point's coordinate (width h) and, when within tol of the bin's edge (with
// a margin for the division's rounding), the neighbour across it: of two points within
// tol of each other, the first's list holds the second's own bin
inline std::vector<std::int64_t> bins_near(double v, double h, double tol)
{
  const double m = v / h;
  const auto b = static_cast<std::int64_t>(std::floor(m));
  std::vector<std::int64_t> out = {b};
  const double frac = m - static_cast<double>(b);
  const double near = 2 * tol / h;
  if (frac <= near)
    out.push_back(b - 1);
  if (frac >= 1.0 - near)
    out.push_back(b + 1);
  return out;
}

// Periodic partners of items with K points (x: 3*K doubles an item), given by their
// local ids: an item with all points on the upper plane of a periodic axis pairs with
// the item at its image on the lower plane, every axis at once (an item on two upper
// planes has a partner across each). Each item is to be given by one rank only (its
// owner). Returns the partners, both ways, each pair once. An item with none is left
// so: whether the faces are congruent is the caller's guard to judge (a slab's cut
// leaves face nodes without partners, rightly), not the pairing's.
//
// Items meet by the bins (width h, much wider than tol) of their image's centroid: a
// master goes to the rank of its own bin, a slave to the ranks of every bin a master
// within tol may have (its own and, near a bin's edge, the neighbour's). There the
// masters are sorted by (axis, bin, an in-plane coordinate) and each slave looks up its
// bins and, within one, the in-plane coordinate within tol: O(n log n) on any face.
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
  // role (0 slave, 1 master), axis, id, the image's K points
  constexpr int W = 3 + 3 * K;
  using Key = std::array<std::int64_t, 4>;   // axis, bin
  auto centroid = [](const double* pts, int d)
  {
    double m = 0;
    for (int k = 0; k < K; ++k)
      m += pts[3 * k + d];
    return m / K;
  };
  std::map<int, std::vector<double>> send;
  for (std::size_t i = 0; i < ids.size(); ++i)
  {
    const double* p = &x[3 * K * i];
    for (int a : box.axes)
    {
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
      std::array<std::vector<std::int64_t>, 3> bins;
      for (int d = 0; d < 3; ++d)
      {
        bins[d] = bins_near(centroid(pts[0].data(), d), h, box.tol);
        if (!up)
          bins[d].resize(1);         // a master: its own bin only
      }
      std::set<int> dests;
      for (auto b0 : bins[0])
        for (auto b1 : bins[1])
          for (auto b2 : bins[2])
            dests.insert(rank_of(Key{a, b0, b1, b2}, size));
      for (int dest : dests)
      {
        auto& s = send[dest];
        s.push_back(up ? 0.0 : 1.0);
        s.push_back(static_cast<double>(a));
        s.push_back(static_cast<double>(ids[i]));
        for (int k = 0; k < K; ++k)
          s.insert(s.end(), pts[k].begin(), pts[k].end());
      }
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

  struct Rec
  {
    int rank;
    std::int32_t id;
    int axis;
    std::array<double, 3 * K> k;
  };
  std::vector<Rec> slaves;
  std::vector<std::pair<Key, Rec>> masters;
  // the in-plane coordinate masters are sorted by within a bin: one that varies on the
  // face (the axis's own is the plane's; in 2D the third is 0)
  auto along = [](int a) { return a == 0 ? 1 : 0; };
  for (std::size_t s = 0; s < sources.size(); ++s)
    for (std::size_t i = 0; i + W <= got[s].size(); i += W)
    {
      Rec r{sources[s], static_cast<std::int32_t>(got[s][i + 2]),
            static_cast<int>(got[s][i + 1]), {}};
      for (int d = 0; d < 3 * K; ++d)
        r.k[d] = got[s][i + 3 + d];
      if (got[s][i] == 0.0)
        slaves.push_back(r);
      else
      {
        Key key{r.axis, 0, 0, 0};
        for (int d = 0; d < 3; ++d)
          key[1 + d] = bins_near(centroid(r.k.data(), d), h, box.tol)[0];
        masters.emplace_back(key, r);
      }
    }
  auto master_less = [&](const std::pair<Key, Rec>& u, const std::pair<Key, Rec>& v)
  {
    if (u.first != v.first)
      return u.first < v.first;
    return u.second.k[along(u.second.axis)] < v.second.k[along(v.second.axis)];
  };
  std::sort(masters.begin(), masters.end(), master_less);
  std::map<int, std::vector<std::int64_t>> reply;   // (my id, partner rank, partner id)
  for (const auto& s : slaves)
  {
    const int c = along(s.axis);
    std::array<std::vector<std::int64_t>, 3> bins;
    for (int d = 0; d < 3; ++d)
      bins[d] = bins_near(centroid(s.k.data(), d), h, box.tol);
    bool found = false;
    for (auto b0 : bins[0])
      for (auto b1 : bins[1])
        for (auto b2 : bins[2])
        {
          if (found)
            break;
          const Key key{s.axis, b0, b1, b2};
          auto it = std::lower_bound(
              masters.begin(), masters.end(), std::make_pair(key, s.k[c] - box.tol),
              [c](const std::pair<Key, Rec>& m, const std::pair<Key, double>& v)
              { return m.first != v.first ? m.first < v.first : m.second.k[c] < v.second; });
          for (; it != masters.end() and it->first == key
                 and it->second.k[c] <= s.k[c] + box.tol; ++it)
          {
            bool same = true;
            for (int d = 0; d < 3 * K; ++d)
              same &= std::abs(it->second.k[d] - s.k[d]) <= box.tol;
            if (!same)
              continue;
            const Rec& m = it->second;
            auto& rs = reply[s.rank];
            rs.insert(rs.end(), {s.id, m.rank, m.id});
            auto& rm = reply[m.rank];
            rm.insert(rm.end(), {m.id, s.rank, s.id});
            found = true;
            break;
          }
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
  // a slave sent to several ranks is answered once (its master is found only at the
  // rank of the master's own bin); duplicates are dropped anyway
  std::vector<std::array<std::int64_t, 3>> pairs;
  for (const auto& a2 : answers)
    for (std::size_t i = 0; i + 3 <= a2.size(); i += 3)
      pairs.push_back({a2[i], a2[i + 1], a2[i + 2]});
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  Partners out;
  for (const auto& p : pairs)
  {
    out.local.push_back(static_cast<std::int32_t>(p[0]));
    out.rank.push_back(static_cast<int>(p[1]));
    out.remote.push_back(static_cast<std::int32_t>(p[2]));
  }
  return out;
}

// Marks sent to periodic partners: each partner of a marked local item is marked
class Exchange
{
public:
  // `p` is kept (a copy, or moved in): the exchange outlives any temporary
  Exchange(MPI_Comm comm, Partners p) : _p(std::move(p))
  {
    _ranks.assign(_p.rank.begin(), _p.rank.end());
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
  const Partners _p;
  std::vector<int> _ranks;
  MPI_Comm _comm;
};

} // namespace femtools_plaza
