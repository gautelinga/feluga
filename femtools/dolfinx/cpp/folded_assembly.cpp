// femtools.dolfinx.assembly's C++: a bilinear form assembled straight into the arrays of
// a PETSc AIJ matrix over unique dofs, in place of MatSetValuesLocal.
//
// The matrix is one of femtools.dolfinx.assembly.create_matrix: its local-to-global maps
// take every local dof to its unique (compact) index, and it has been assembled once, so
// its diagonal and off-diagonal blocks hold their final patterns. Per call, each local
// dof's row becomes a row of those blocks (or -1, owned by another rank) and each local
// column an index into the diagonal block's columns or the off-diagonal block's
// compressed ones. DOLFINx's assembler then hands over one element matrix at a time,
// whose values are added where a binary search finds their columns in the row. Rows
// owned by another rank are kept and given to MatSetValues afterwards (PETSc stashes
// and sends them); the caller zeroes the matrix before and finalises it after.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>

#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>

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

/// Add the matrix of form `a` into the PETSc matrix at `handle` (a petsc4py
/// Mat.handle). Returns the number of values that fell outside the matrix's pattern
/// (not added), or -1 if the matrix is not one this can fill (not AIJ, never assembled,
/// without local-to-global maps of block size 1 over the local dofs of the form's
/// spaces), when nothing is added.
std::int64_t assemble(const fem::Form<double, double>& a, std::uintptr_t handle)
{
  Mat A = reinterpret_cast<Mat>(handle);
  PetscBool mpi, seq, assembled;
  check(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &mpi), "PetscObjectTypeCompare");
  check(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &seq), "PetscObjectTypeCompare");
  check(MatAssembled(A, &assembled), "MatAssembled");
  if (!(mpi || seq) || !assembled)
    return -1;

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
      return -1;
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
            far_cols.push_back(cg.g[cols[j]]);
          far_vals.insert(far_vals.end(), v, v + nc);
          far_start.push_back(far_cols.size());
          continue;
        }
        for (std::size_t j = 0; j < nc; ++j)
        {
          const PetscInt c = mapped[j];
          if (!(c >= 0 ? D.add(r, c, v[j]) : c < -1 && O.add(r, -2 - c, v[j])))
            ++missed;
        }
      }
      return 0;
    };
    fem::assemble_matrix(
        add, a, std::vector<std::reference_wrapper<const fem::DirichletBC<double, double>>>{});
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
  return missed;
}
} // namespace

NB_MODULE(folded_assembly_x, m)
{
  m.def("assemble", &assemble, nb::arg("form"), nb::arg("handle"));
}
