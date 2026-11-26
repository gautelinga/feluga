// feluga.dolfinx.assembly's C++: a bilinear form assembled straight into the arrays of
// a PETSc AIJ matrix over unique dofs, in place of MatSetValuesLocal.
//
// The matrix is one of feluga.dolfinx.assembly.create_matrix: its local-to-global maps
// take every local dof to its unique (compact) index, and it has been assembled once, so
// its diagonal and off-diagonal blocks hold their final patterns. Per call, each local
// dof's row becomes a row of those blocks (or -1, owned by another rank) and each local
// column an index into the diagonal block's columns or the off-diagonal block's
// compressed ones. DOLFINx's assembler then hands over one element matrix at a time,
// whose values are added where a binary search finds their columns in the row. Rows
// owned by another rank are kept and given to MatSetValues afterwards (PETSc stashes
// and sends them); the caller zeroes the matrix before and finalises it after.
//
// Both this and the vector assembly (assemble_vector, into a ghosted vector's local
// array) take the cells of a cell integral a block at a time, packing the coefficients
// of that block alone: DOLFINx's assembler packs every cell's at once, an array of
// cells x coefficient values that is often larger than the matrix's share of a cell.
// The cells and the order of the additions are DOLFINx's, so the result is its result.
// Facet integrals, over far fewer entities, are packed whole, as DOLFINx does.
//
// With couplings (feluga.couplings: each local dof's component, and a mask of the
// component pairs kept), an element matrix's entries between components the mask does
// not couple are skipped, whatever their value, and the nonzeros among them counted:
// the caller raises if there are any. The entries in the rows and columns of dofs that
// Dirichlet conditions fix (a flag a local dof) are dropped, and each element adds 1 on
// such a dof's diagonal (the same unique dof): the conditions applied symmetrically, as
// dolfin's assemble_system applies them (its diagonal is the number of cells holding the
// dof).

#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <map>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assemble_matrix_impl.h>
#include <dolfinx/fem/assemble_vector_impl.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/pack.h>
#include <dolfinx/fem/utils.h>

#include <petscmat.h>

namespace nb = nanobind;
using namespace dolfinx;

namespace
{
void check(PetscErrorCode ierr, const char* what)
{
  if (ierr != 0)
    throw std::runtime_error(std::string("PETSc error ") + std::to_string(ierr) + " in "
                             + what);
}

using T = double;
using Form = fem::Form<T, T>;
using IT = fem::IntegralType;
using xmap_t = md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 3>>;
using map2_t = md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>;
using ents2_t = md::mdspan<const std::int32_t, md::extents<std::size_t, md::dynamic_extent, 2>>;
using ents22_t
    = md::mdspan<const std::int32_t, md::extents<std::size_t, md::dynamic_extent, 2, 2>>;
using c2x_t = md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                               md::dynamic_extent>>;

// Cells of a cell integral taken at a time (set_block changes it, for tests)
std::size_t BLOCK = 8192;

using labels_t = nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using mask_t = nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Which entries of an element matrix are kept: all, or (with couplings) those whose row
// and column dofs' components the mask couples. Local dof indices.
struct Filter
{
  const std::int8_t* r = nullptr;
  const std::int8_t* c = nullptr;
  const std::uint8_t* m = nullptr;
  const std::int8_t* f = nullptr;
  const PetscInt* gr = nullptr;   // the unique (global) index of each local dof, rows
  const PetscInt* gc = nullptr;   // ... and columns: set once the maps are open
  std::size_t n1 = 0;

  Filter(const labels_t& rows, const labels_t& cols, const mask_t& mask, const labels_t& fixed)
  {
    if (mask.size() == 0)
      return;
    r = rows.data();
    c = cols.data();
    m = mask.data();
    n1 = mask.shape(1);
    f = fixed.size() ? fixed.data() : nullptr;
  }
  bool on() const { return m != nullptr; }

  // What becomes of the element's entry v at local (i, j), the one rule every inserter
  // follows: ADD (*x the value added), DROP (a fixed dof's row or column off its
  // diagonal: the condition applied, whatever the value) or SKIP (components the mask
  // does not couple: the caller counts it when v is nonzero). A fixed dof's diagonal is
  // kept whatever the mask, and adds 1 (dolfin's assemble_system: the number of cells
  // holding the dof, the form being one cell integral).
  enum Fate { ADD, DROP, SKIP };
  Fate fate(std::int32_t i, std::int32_t j, double v, double* x) const
  {
    *x = v;
    if (!on())
      return ADD;
    if (f && (f[i] || f[j]))
    {
      if (gr[i] != gc[j])
        return DROP;
      *x = 1.0;
      return ADD;
    }
    return m[r[i] * n1 + c[j]] != 0 ? ADD : SKIP;
  }
};

// The coefficients of cell integral i for its cells [start, start + m), into c
void pack_cells(std::vector<T>& c, const Form& a, int i, std::size_t start, std::size_t m,
                int cstride)
{
  const auto& coefficients = a.coefficients();
  const std::vector<int> offsets = a.coefficient_offsets();
  c.assign(m * cstride, 0);
  for (int k : a.active_coeffs(IT::cell, i))
  {
    std::span<const std::int32_t> cells = a.domain_coeff(IT::cell, i, k);
    md::mdspan<const std::int32_t, md::dextents<std::size_t, 1>> block(cells.data() + start,
                                                                       m);
    fem::impl::pack_coefficient_entity(std::span<T>(c), cstride, *coefficients[k],
                                       fem::impl::get_cell_orientation_info(*coefficients[k]),
                                       block, offsets[k]);
  }
}

// The coefficients of the form's facet (and vertex, ridge) integrals, packed whole
std::map<std::pair<IT, int>, std::pair<std::vector<T>, int>> pack_facets(const Form& a)
{
  std::map<std::pair<IT, int>, std::pair<std::vector<T>, int>> coeffs;
  for (IT type : {IT::interior_facet, IT::exterior_facet, IT::vertex, IT::ridge})
    for (int i = 0; i < a.num_integrals(type, 0); ++i)
      coeffs.emplace(std::pair{type, i}, fem::allocate_coefficient_storage(a, type, i));
  fem::pack_coefficients(a, coeffs);
  return coeffs;
}

md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> facet_permutations(const Form& a)
{
  md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms;
  if (a.needs_facet_permutations())
  {
    auto mesh = a.mesh();
    int n = mesh::cell_num_entities(mesh->topology()->cell_types()[0],
                                    mesh->topology()->dim() - 1);
    mesh->topology_mutable()->create_entity_permutations();
    const std::vector<std::uint8_t>& p = mesh->topology()->get_facet_permutations();
    perms = md::mdspan(p.data(), p.size() / n, n);
  }
  return perms;
}

std::span<const std::uint32_t> cell_info(const Form& a, const fem::FiniteElement<T>& e,
                                         const mesh::Mesh<T>& m)
{
  if (e.needs_dof_transformations() or a.needs_facet_permutations())
  {
    m.topology_mutable()->create_entity_permutations();
    return std::span(m.topology()->get_cell_permutation_info());
  }
  return {};
}

// Matrix form a's element matrices given to `add`, as fem::assemble_matrix(add, a, {})
// gives them, the cells a block at a time (one cell type; else DOLFINx's own)
template <typename Add>
void assemble_matrix_blocks(Add add, const Form& a)
{
  auto mesh = a.mesh();
  if (mesh->topology()->cell_types().size() != 1)
  {
    fem::assemble_matrix(
        add, a, std::vector<std::reference_wrapper<const fem::DirichletBC<T, T>>>{});
    return;
  }
  const std::vector<T> constants = fem::pack_constants(a);
  auto xg = mesh->geometry().x();
  xmap_t x(xg.data(), xg.size() / 3, 3);
  map2_t x_dofmap = mesh->geometry().dofmap();
  auto V0 = a.function_spaces().at(0), V1 = a.function_spaces().at(1);
  auto dofmap0 = V0->dofmap(), dofmap1 = V1->dofmap();
  auto dofs0 = dofmap0->map(), dofs1 = dofmap1->map();
  const int bs0 = dofmap0->bs(), bs1 = dofmap1->bs();
  auto element0 = V0->element(), element1 = V1->element();
  auto P0 = element0->template dof_transformation_fn<T>(fem::doftransform::standard);
  auto P1T
      = element1->template dof_transformation_right_fn<T>(fem::doftransform::transpose);
  std::span<const std::uint32_t> info0 = cell_info(a, *element0, *V0->mesh());
  std::span<const std::uint32_t> info1 = cell_info(a, *element1, *V1->mesh());
  std::span<const std::int8_t> bc0, bc1;
  const std::vector<int> offsets = a.coefficient_offsets();
  const int cstride = offsets.empty() ? 0 : offsets.back();

  std::vector<T> c;
  for (int i = 0; i < a.num_integrals(IT::cell, 0); ++i)
  {
    auto fn = a.kernel(IT::cell, i, 0);
    std::span<const std::int32_t> cells = a.domain(IT::cell, i, 0);
    std::span<const std::int32_t> cells0 = a.domain_arg(IT::cell, 0, i, 0);
    std::span<const std::int32_t> cells1 = a.domain_arg(IT::cell, 1, i, 0);
    for (std::size_t start = 0; start < cells.size(); start += BLOCK)
    {
      const std::size_t m = std::min(BLOCK, cells.size() - start);
      pack_cells(c, a, i, start, m, cstride);
      fem::impl::assemble_cells_matrix<T, false>(
          add, x_dofmap, x, cells.subspan(start, m), {dofs0, bs0, cells0.subspan(start, m)},
          P0, {dofs1, bs1, cells1.subspan(start, m)}, P1T, bc0, bc1, fn,
          md::mdspan<const T, md::dextents<std::size_t, 2>>(c.data(), m, cstride),
          std::span<const T>(constants), info0, info1);
    }
  }

  auto coeffs = pack_facets(a);
  auto perms = facet_permutations(a);
  for (int i = 0; i < a.num_integrals(IT::interior_facet, 0); ++i)
  {
    auto fn = a.kernel(IT::interior_facet, i, 0);
    auto& [cf, cs] = coeffs.at({IT::interior_facet, i});
    std::span f = a.domain(IT::interior_facet, i, 0);
    std::span f0 = a.domain_arg(IT::interior_facet, 0, i, 0);
    std::span f1 = a.domain_arg(IT::interior_facet, 1, i, 0);
    fem::impl::assemble_interior_facets<T, false>(
        add, x_dofmap, x, ents22_t(f.data(), f.size() / 4, 2, 2),
        {*dofmap0, bs0, ents22_t(f0.data(), f0.size() / 4, 2, 2)}, P0,
        {*dofmap1, bs1, ents22_t(f1.data(), f1.size() / 4, 2, 2)}, P1T, bc0, bc1, fn,
        c2x_t(cf.data(), f.size() / 4, 2, cs), std::span<const T>(constants), info0, info1,
        perms);
  }
  for (IT type : {IT::exterior_facet, IT::vertex, IT::ridge})
  {
    auto p = type == IT::exterior_facet
                 ? perms
                 : md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>>{};
    for (int i = 0; i < a.num_integrals(type, 0); ++i)
    {
      auto fn = a.kernel(type, i, 0);
      auto& [cf, cs] = coeffs.at({type, i});
      std::span e = a.domain(type, i, 0);
      std::span e0 = a.domain_arg(type, 0, i, 0);
      std::span e1 = a.domain_arg(type, 1, i, 0);
      ents2_t ents(e.data(), e.size() / 2, 2);
      fem::impl::assemble_entities<T, false>(
          add, x_dofmap, x, ents, {dofs0, bs0, ents2_t(e0.data(), e0.size() / 2, 2)}, P0,
          {dofs1, bs1, ents2_t(e1.data(), e1.size() / 2, 2)}, P1T, bc0, bc1, fn,
          md::mdspan<const T, md::dextents<std::size_t, 2>>(cf.data(), ents.extent(0), cs),
          std::span<const T>(constants), info0, info1, p);
    }
  }
}

// Vector form L assembled into b (a ghosted vector's local array, owned then ghost
// entries), added to what b holds, the cells a block at a time
template <int BS>
void assemble_vector_blocks(std::span<T> b, const Form& L)
{
  auto mesh = L.mesh();
  const std::vector<T> constants = fem::pack_constants(L);
  auto xg = mesh->geometry().x();
  xmap_t x(xg.data(), xg.size() / 3, 3);
  map2_t x_dofmap = mesh->geometry().dofmap();
  auto V0 = L.function_spaces().at(0);
  auto dofmap = V0->dofmap();
  auto dofs = dofmap->map();
  const int bs = dofmap->bs();
  auto element = V0->element();
  auto P0 = element->template dof_transformation_fn<T>(fem::doftransform::standard);
  std::span<const std::uint32_t> info0 = cell_info(L, *element, *V0->mesh());
  const std::vector<int> offsets = L.coefficient_offsets();
  const int cstride = offsets.empty() ? 0 : offsets.back();

  std::vector<T> c;
  for (int i = 0; i < L.num_integrals(IT::cell, 0); ++i)
  {
    auto fn = L.kernel(IT::cell, i, 0);
    std::span<const std::int32_t> cells = L.domain(IT::cell, i, 0);
    std::span<const std::int32_t> cells0 = L.domain_arg(IT::cell, 0, i, 0);
    for (std::size_t start = 0; start < cells.size(); start += BLOCK)
    {
      const std::size_t m = std::min(BLOCK, cells.size() - start);
      pack_cells(c, L, i, start, m, cstride);
      fem::impl::assemble_cells<BS>(
          P0, b, x_dofmap, x, cells.subspan(start, m),
          {dofs, bs, cells0.subspan(start, m)}, fn, std::span<const T>(constants),
          md::mdspan<const T, md::dextents<std::size_t, 2>>(c.data(), m, cstride), info0);
    }
  }

  auto coeffs = pack_facets(L);
  auto perms = facet_permutations(L);
  for (int i = 0; i < L.num_integrals(IT::interior_facet, 0); ++i)
  {
    auto fn = L.kernel(IT::interior_facet, i, 0);
    auto& [cf, cs] = coeffs.at({IT::interior_facet, i});
    std::span f = L.domain(IT::interior_facet, i, 0);
    std::span f1 = L.domain_arg(IT::interior_facet, 0, i, 0);
    fem::impl::assemble_interior_facets<BS>(
        P0, b, x_dofmap, x, ents22_t(f.data(), f.size() / 4, 2, 2),
        {*dofmap, bs, ents22_t(f1.data(), f1.size() / 4, 2, 2)}, fn,
        std::span<const T>(constants), c2x_t(cf.data(), f.size() / 4, 2, cs), info0, perms);
  }
  for (IT type : {IT::exterior_facet, IT::vertex, IT::ridge})
  {
    auto p = type == IT::exterior_facet
                 ? perms
                 : md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>>{};
    for (int i = 0; i < L.num_integrals(type, 0); ++i)
    {
      auto fn = L.kernel(type, i, 0);
      auto& [cf, cs] = coeffs.at({type, i});
      std::span e = L.domain(type, i, 0);
      std::span e1 = L.domain_arg(type, 0, i, 0);
      ents2_t ents(e.data(), e.size() / 2, 2);
      fem::impl::assemble_entities<BS>(
          P0, b, x_dofmap, x, ents, {dofs, bs, ents2_t(e1.data(), e1.size() / 2, 2)}, fn,
          std::span<const T>(constants),
          md::mdspan<const T, md::dextents<std::size_t, 2>>(cf.data(), ents.extent(0), cs),
          info0, p);
    }
  }
}

// The local-to-global indices of a map of block size 1, held for the call.
struct Indices
{
  ISLocalToGlobalMapping map = nullptr;
  const PetscInt* g = nullptr;
  PetscInt n = 0;

  bool open(ISLocalToGlobalMapping m)
  {
    PetscInt bs;
    if (!m)
      return false;
    check(ISLocalToGlobalMappingGetBlockSize(m, &bs), "ISLocalToGlobalMappingGetBlockSize");
    if (bs != 1)
      return false;
    map = m;
    check(ISLocalToGlobalMappingGetSize(map, &n), "ISLocalToGlobalMappingGetSize");
    check(ISLocalToGlobalMappingGetIndices(map, &g), "ISLocalToGlobalMappingGetIndices");
    return true;
  }

  void close()
  {
    if (g)
      ISLocalToGlobalMappingRestoreIndices(map, &g);
    g = nullptr;
  }
};

/// Add the matrix of form `a` into the PETSc matrix at `handle` through
/// MatSetValuesLocal (its local-to-global maps), the cells a block at a time: for a
/// matrix whose pattern is not yet final (its first assembly). The caller zeroes and
/// finalises it. Returns -1, adding nothing, unless the matrix has local-to-global maps
/// of block size 1 over the local dofs of the form's spaces; else the number of values
/// the couplings skipped that were nonzero (the caller raises on any).
std::int64_t assemble_local(const Form& a, std::uintptr_t handle, const labels_t& rl,
                            const labels_t& cl, const mask_t& mask, const labels_t& fixed)
{
  Filter filt(rl, cl, mask, fixed);
  Mat A = reinterpret_cast<Mat>(handle);
  ISLocalToGlobalMapping maps[2];
  check(MatGetLocalToGlobalMapping(A, &maps[0], &maps[1]), "MatGetLocalToGlobalMapping");
  for (int k = 0; k < 2; ++k)
  {
    PetscInt bs, n;
    if (!maps[k])
      return -1;
    check(ISLocalToGlobalMappingGetBlockSize(maps[k], &bs), "ISLocalToGlobalMappingGetBlockSize");
    check(ISLocalToGlobalMappingGetSize(maps[k], &n), "ISLocalToGlobalMappingGetSize");
    auto dofmap = a.function_spaces().at(k)->dofmap();
    if (bs != 1
        or n != PetscInt(dofmap->index_map_bs()
                         * (dofmap->index_map->size_local() + dofmap->index_map->num_ghosts())))
    {
      return -1;
    }
  }
  std::vector<PetscInt> r, c;   // the local indices as PetscInt, 32- or 64-bit
  std::vector<double> kept;
  Indices rgi, cgi;             // the unique indices, for the diagonal of fixed dofs
  if (filt.f)
  {
    rgi.open(maps[0]);
    cgi.open(maps[1]);
    filt.gr = rgi.g;
    filt.gc = cgi.g;
  }
  std::int64_t skipped = 0;     // nonzeros between components the mask does not couple
  auto add = [A, &r, &c, &kept, &filt, &skipped](std::span<const std::int32_t> rows,
                                                 std::span<const std::int32_t> cols,
                                                 std::span<const double> vals) -> int
  {
    if (!filt.on())
    {
      r.assign(rows.begin(), rows.end());
      c.assign(cols.begin(), cols.end());
      check(MatSetValuesLocal(A, r.size(), r.data(), c.size(), c.data(), vals.data(),
                              ADD_VALUES),
            "MatSetValuesLocal");
      return 0;
    }
    const std::size_t nc = cols.size();
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
      c.clear();
      kept.clear();
      for (std::size_t j = 0; j < nc; ++j)
      {
        double x;
        const double v = vals[i * nc + j];
        switch (filt.fate(rows[i], cols[j], v, &x))
        {
        case Filter::ADD:
          c.push_back(cols[j]);
          kept.push_back(x);
          break;
        case Filter::SKIP:
          skipped += v != 0.0;
          break;
        case Filter::DROP:
          break;
        }
      }
      const PetscInt ri = rows[i];
      if (!c.empty())
        check(MatSetValuesLocal(A, 1, &ri, c.size(), c.data(), kept.data(), ADD_VALUES),
              "MatSetValuesLocal");
    }
    return 0;
  };
  assemble_matrix_blocks(add, a);
  rgi.close();
  cgi.close();
  return skipped;
}

/// Add the vector of form `L` into `b`, the local array (owned then ghost entries) of a
/// vector over L's space, the cells a block at a time. Returns false, adding nothing,
/// for a mesh of several cell types.
bool assemble_vector(const Form& L,
                     nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> b)
{
  if (L.mesh()->topology()->cell_types().size() != 1)
    return false;
  std::span<T> out(b.data(), b.size());
  switch (L.function_spaces().at(0)->dofmap()->bs())
  {
  case 1:
    assemble_vector_blocks<1>(out, L);
    break;
  case 3:
    assemble_vector_blocks<3>(out, L);
    break;
  default:
    assemble_vector_blocks<-1>(out, L);
  }
  return true;
}

// One AIJ block opened for writing: its rows' column indices and its values.
struct Block
{
  Mat m = nullptr;
  const PetscInt* ia = nullptr;
  const PetscInt* ja = nullptr;
  PetscScalar* a = nullptr;

  void open(Mat block)
  {
    PetscInt n;
    PetscBool done;
    m = block;
    check(MatGetRowIJ(m, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done), "MatGetRowIJ");
    if (!done)
      throw std::runtime_error("MatGetRowIJ gave no rows");
    check(MatSeqAIJGetArray(m, &a), "MatSeqAIJGetArray");
  }

  void close()
  {
    PetscInt n;
    PetscBool done;
    if (a)
      MatSeqAIJRestoreArray(m, &a);
    if (ia)
      MatRestoreRowIJ(m, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done);
    a = nullptr;
    ia = ja = nullptr;
  }

  // Add v at column c of row r; false if the pattern has no such entry.
  bool add(PetscInt r, PetscInt c, PetscScalar v)
  {
    const PetscInt* lo = ja + ia[r];
    const PetscInt* hi = ja + ia[r + 1];
    const PetscInt* p = std::lower_bound(lo, hi, c);
    if (p == hi || *p != c)
      return false;
    a[p - ja] += v;
    return true;
  }
};


/// Add the matrix of form `a` into the PETSc matrix at `handle` (a petsc4py
/// Mat.handle), skipping the entries the couplings do not keep (rl, cl, mask; an
/// empty mask keeps all). Returns (the number of kept values that fell outside the
/// matrix's pattern, not added; the number of skipped values that were nonzero); the
/// first is -1 if the matrix was never assembled (its pattern not final: see
/// assemble_local), -2 if it is not one this can fill (not AIJ, or without
/// local-to-global maps of block size 1 over the local dofs of the form's spaces),
/// when nothing is added.
std::pair<std::int64_t, std::int64_t> assemble(const fem::Form<double, double>& a,
                                               std::uintptr_t handle, const labels_t& rl,
                                               const labels_t& cl, const mask_t& mask,
                                               const labels_t& fixed)
{
  Filter filt(rl, cl, mask, fixed);
  std::int64_t skipped = 0;
  Mat A = reinterpret_cast<Mat>(handle);
  PetscBool mpi, seq, assembled;
  check(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &mpi), "PetscObjectTypeCompare");
  check(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &seq), "PetscObjectTypeCompare");
  check(MatAssembled(A, &assembled), "MatAssembled");
  if (!(mpi || seq))
    return {-2, 0};
  if (!assembled)
    return {-1, 0};

  Mat Ad = A, Ao = nullptr;
  const PetscInt* garray = nullptr;
  PetscInt n_far_cols = 0;
  if (mpi)
  {
    check(MatMPIAIJGetSeqAIJ(A, &Ad, &Ao, &garray), "MatMPIAIJGetSeqAIJ");
    check(MatGetSize(Ao, nullptr, &n_far_cols), "MatGetSize");
  }
  PetscInt rstart, rend, cstart, cend;
  check(MatGetOwnershipRange(A, &rstart, &rend), "MatGetOwnershipRange");
  check(MatGetOwnershipRangeColumn(A, &cstart, &cend), "MatGetOwnershipRangeColumn");

  ISLocalToGlobalMapping rmap, cmap;
  check(MatGetLocalToGlobalMapping(A, &rmap, &cmap), "MatGetLocalToGlobalMapping");
  Indices rg, cg;
  Block D, O;
  std::int64_t missed = 0;
  // rows owned by another rank, one element row each: global row, its columns' start
  // in far_cols and far_vals (global columns, values)
  std::vector<PetscInt> far_rows, far_start{0}, far_cols;
  std::vector<PetscScalar> far_vals;
  try
  {
    // the maps must cover the local dofs of the form's two spaces
    auto n_local = [&](int k)
    {
      auto dofmap = a.function_spaces().at(k)->dofmap();
      return PetscInt(dofmap->index_map_bs()
                      * (dofmap->index_map->size_local() + dofmap->index_map->num_ghosts()));
    };
    if (!rg.open(rmap) || !cg.open(cmap) || rg.n != n_local(0) || cg.n != n_local(1))
    {
      rg.close();
      cg.close();
      return {-2, 0};
    }
    // row[i]: row of local dof i in the blocks, -1 if another rank owns it
    std::vector<PetscInt> row(rg.n);
    for (PetscInt i = 0; i < rg.n; ++i)
      row[i] = (rg.g[i] >= rstart && rg.g[i] < rend) ? rg.g[i] - rstart : -1;
    // col[j]: >= 0 a column of the diagonal block; -2 - k column k of the off-diagonal
    // block; -1 in neither (so outside the pattern of every row owned here)
    std::vector<PetscInt> col(cg.n);
    for (PetscInt j = 0; j < cg.n; ++j)
    {
      const PetscInt g = cg.g[j];
      if (g >= cstart && g < cend)
        col[j] = g - cstart;
      else
      {
        const PetscInt* p = std::lower_bound(garray, garray + n_far_cols, g);
        col[j] = (p != garray + n_far_cols && *p == g) ? -2 - (p - garray) : -1;
      }
    }

    filt.gr = rg.g;
    filt.gc = cg.g;
    D.open(Ad);
    if (Ao)
      O.open(Ao);
    std::vector<PetscInt> mapped; // the element's columns, mapped as col
    auto add = [&](std::span<const std::int32_t> rows, std::span<const std::int32_t> cols,
                   std::span<const double> vals) -> int
    {
      const std::size_t nc = cols.size();
      mapped.resize(nc);
      for (std::size_t j = 0; j < nc; ++j)
        mapped[j] = col[cols[j]];
      for (std::size_t i = 0; i < rows.size(); ++i)
      {
        const double* v = vals.data() + i * nc;
        const PetscInt r = row[rows[i]];
        if (r < 0)
        {
          far_rows.push_back(rg.g[rows[i]]);
          for (std::size_t j = 0; j < nc; ++j)
          {
            double x;
            const Filter::Fate what = filt.fate(rows[i], cols[j], v[j], &x);
            skipped += what == Filter::SKIP && v[j] != 0.0;
            if (what != Filter::ADD)
              continue;
            far_cols.push_back(cg.g[cols[j]]);
            far_vals.push_back(x);
          }
          far_start.push_back(far_cols.size());
          continue;
        }
        for (std::size_t j = 0; j < nc; ++j)
        {
          double x;
          const Filter::Fate what = filt.fate(rows[i], cols[j], v[j], &x);
          skipped += what == Filter::SKIP && v[j] != 0.0;
          if (what != Filter::ADD)
            continue;
          const PetscInt c = mapped[j];
          if (!(c >= 0 ? D.add(r, c, x) : c < -1 && O.add(r, -2 - c, x)))
            ++missed;
        }
      }
      return 0;
    };
    assemble_matrix_blocks(add, a);
  }
  catch (...)
  {
    D.close();
    O.close();
    rg.close();
    cg.close();
    throw;
  }
  D.close();
  O.close();
  rg.close();
  cg.close();
  for (std::size_t k = 0; k < far_rows.size(); ++k)
  {
    const PetscInt n = far_start[k + 1] - far_start[k];
    check(MatSetValues(A, 1, &far_rows[k], n, far_cols.data() + far_start[k],
                       far_vals.data() + far_start[k], ADD_VALUES),
          "MatSetValues");
  }
  return {missed, skipped};
}
} // namespace

NB_MODULE(folded_assembly_x, m)
{
  m.def("assemble", &assemble, nb::arg("form"), nb::arg("handle"), nb::arg("rows"),
        nb::arg("cols"), nb::arg("mask"), nb::arg("fixed"));
  m.def("assemble_vector", &assemble_vector, nb::arg("form"), nb::arg("b"));
  m.def("assemble_local", &assemble_local, nb::arg("form"), nb::arg("handle"),
        nb::arg("rows"), nb::arg("cols"), nb::arg("mask"), nb::arg("fixed"));
  m.def("set_block", [](std::size_t n) { BLOCK = std::max<std::size_t>(n, 1); },
        nb::arg("cells"));
}
