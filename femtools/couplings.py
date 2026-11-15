"""Which components of a mixed space a bilinear form couples, and matrices that store only
those blocks.

    mask = form_couplings(a)            # bool (test components x trial components)
    assert (mask == expected).all()     # the caller says what it expects
    A = backend.assemble(a, couplings="form")       # or couplings=mask

A mixed space's components are its scalar leaves in value order: on [P2^3, P1] the
three velocity components and the pressure, 0..3. Both backends' sparsity patterns couple
every dof of a cell with every other, so a form that never couples two components (the
Stokes operator: no velocity component with another, nor the pressure with itself) is
stored with those blocks as zeros -- about 70 % of the Stokes matrix. With `couplings`
a matrix is preallocated for the coupled blocks alone (and, for a square form, the
diagonal), and assembly skips the others **by component, never by value**: an entry of a
coupled block that happens to be zero is stored, so the pattern is the same whatever the
values, and a nonzero in a skipped block is an error, never dropped.

`form_couplings` decides from the form itself: for each pair (i, j) the test function is
replaced by its component i alone (the others zero) and the trial function by its
component j, and the pair is coupled unless UFL simplifies the form to nothing. That is
conservative (a form that is zero only numerically counts as coupled), and the assembly
checks the other way round. UFL's and legacy dolfin's `ufl_legacy` alike.
"""
import numpy as np


def _ufl(form):
    """The UFL module a form belongs to: `ufl` (DOLFINx) or `ufl_legacy` (dolfin)."""
    import importlib
    return importlib.import_module(type(form).__module__.split(".")[0])


def _width(arg):
    """Scalar components of an Argument's value (1 for a scalar)."""
    return int(np.prod(arg.ufl_shape)) if arg.ufl_shape else 1


def _component(ufl, arg, k):
    """`arg` with every value component but k set to zero (a scalar: itself)."""
    if not arg.ufl_shape:
        return arg
    if len(arg.ufl_shape) != 1:
        raise NotImplementedError("couplings of an argument of value shape {}"
                                  .format(arg.ufl_shape))
    return ufl.as_vector([arg[m] if m == k else 0 for m in range(arg.ufl_shape[0])])


def _vanishes(ufl, form):
    """Whether UFL reduces `form` to nothing: algebra lowered, derivatives applied and
    indices expanded, every integrand is a Zero."""
    from importlib import import_module
    alg = import_module(ufl.__name__ + ".algorithms")
    lower = import_module(ufl.__name__ + ".algorithms.apply_algebra_lowering")
    derivs = import_module(ufl.__name__ + ".algorithms.apply_derivatives")
    form = derivs.apply_derivatives(lower.apply_algebra_lowering(form))
    if hasattr(alg, "expand_indices"):
        form = alg.expand_indices(form)
    zero = import_module(ufl.__name__ + ".constantvalue").Zero
    return form.empty() or all(isinstance(i.integrand(), zero) for i in form.integrals())


def form_couplings(a):
    """`mask[i, j]`: whether bilinear form `a` couples component i of its test function
    with component j of its trial function (scalar components, in value order). Coupled
    unless the form with the other components set to zero simplifies to nothing."""
    ufl = _ufl(a)
    args = sorted(a.arguments(), key=lambda x: x.number())
    if len(args) != 2:
        raise ValueError("form_couplings takes a bilinear form")
    test, trial = args
    mask = np.zeros((_width(test), _width(trial)), dtype=bool)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            replaced = ufl.replace(a, {test: _component(ufl, test, i),
                                       trial: _component(ufl, trial, j)})
            mask[i, j] = not _vanishes(ufl, replaced)
    return mask


def resolve(couplings, form):
    """The mask `couplings` stands for on `form`: None (every block), "form"
    (`form_couplings`), or a boolean array of shape (test components, trial components)."""
    if couplings is None:
        return None
    if isinstance(couplings, str):
        if couplings != "form":
            raise ValueError('couplings is None, "form" or a mask, not {!r}'.format(couplings))
        return form_couplings(form)
    mask = np.asarray(couplings, dtype=bool)
    args = sorted(form.arguments(), key=lambda x: x.number())
    want = tuple(_width(x) for x in args)
    if mask.shape != want:
        raise ValueError("a couplings mask of shape {} for a form of {} x {} components"
                         .format(mask.shape, *want))
    return mask


def ones(form):
    """The mask of every component pair of bilinear `form`."""
    args = sorted(form.arguments(), key=lambda x: x.number())
    return np.ones(tuple(_width(x) for x in args), dtype=bool)


def leaves(V, sub=None, count=None):
    """The paths (tuples of sub-space indices) of the scalar components of space `V`, in
    value order: () for a scalar space, ((0, 0), (0, 1), (0, 2), (1,)) for [P2^3, P1].
    `sub(V, i)` gives a sub-space and `count(V)` the number of them (by default `V.sub(i)`
    and `V.num_sub_spaces`, called or read)."""
    sub = sub or (lambda W, i: W.sub(i))
    count = count or (lambda W: W.num_sub_spaces() if callable(W.num_sub_spaces)
                      else W.num_sub_spaces)
    out = []

    def walk(W, path):
        n = count(W)
        if n == 0:
            out.append(path)
        for i in range(n):
            walk(sub(W, i), path + (i,))
    walk(V, ())
    return out


# --- legacy dolfin: the pattern of the coupled blocks, as a TensorLayout ---------------

_CODE = r"""
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dolfin/common/ArrayView.h>
#include <dolfin/la/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/mesh/Mesh.h>
#include <petscmat.h>

namespace py = pybind11;
using namespace dolfin;

using Paths = std::vector<std::vector<std::size_t>>;

// Each component's sub-dofmap, in W's own local numbering (a periodic constraint's
// masters included); a scalar space's own dofmap for the empty path
static std::vector<std::shared_ptr<GenericDofMap>> parts(const FunctionSpace& V,
                                                         const Paths& paths)
{
  const GenericDofMap& dm = *V.dofmap();
  std::vector<std::shared_ptr<GenericDofMap>> out;
  for (const auto& p : paths)
    out.push_back(p.empty() ? V.dofmap()->copy() : dm.extract_sub_dofmap(p, *V.mesh()));
  return out;
}

// The local dofs (owned and ghost) the conditions constrain, found as assemble_system
// finds them: each condition's boundary values, gathered from the ranks that see the
// facets when there are several
static std::vector<char> fixed_dofs(const FunctionSpace& V,
                                    const std::vector<std::shared_ptr<const DirichletBC>>& bcs)
{
  auto im = V.dofmap()->index_map();
  std::vector<char> fixed(im->size(IndexMap::MapSize::ALL) * im->block_size(), 0);
  for (const auto& bc : bcs)
  {
    DirichletBC::Map values;
    bc->get_boundary_values(values);
    if (dolfin::MPI::size(V.mesh()->mpi_comm()) > 1 && bc->method() != "pointwise")
      bc->gather(values);
    for (const auto& kv : values)
      fixed[kv.first] = 1;
  }
  return fixed;
}

// The dofs of `dofs` not fixed
static void unfixed(const Eigen::Ref<const Eigen::Array<dolfin::la_index, Eigen::Dynamic, 1>> dofs,
                    const std::vector<char>& fixed, std::vector<dolfin::la_index>& out)
{
  out.clear();
  for (Eigen::Index k = 0; k < dofs.size(); ++k)
    if (fixed.empty() || !fixed[dofs[k]])
      out.push_back(dofs[k]);
}

// Calls f(rows, cols) for every coupled block of every cell, the fixed dofs' rows and
// columns left out, and f(d, d) for each row dof alone when `diagonal` (a square form: the
// diagonal is kept whatever the mask, fixed dofs' included)
template <typename F>
static void blocks(const Mesh& mesh, const std::vector<std::shared_ptr<GenericDofMap>>& p0,
                   const std::vector<std::shared_ptr<GenericDofMap>>& p1,
                   const std::vector<int>& mask, bool diagonal, const std::vector<char>& fixed,
                   F f)
{
  const std::size_t n0 = p0.size(), n1 = p1.size();
  std::vector<dolfin::la_index> r, s;
  for (std::size_t c = 0; c < mesh.num_cells(); ++c)
  {
    for (std::size_t i = 0; i < n0; ++i)
    {
      auto rd = p0[i]->cell_dofs(c);
      if (diagonal)
        for (Eigen::Index k = 0; k < rd.size(); ++k)
        {
          ArrayView<const dolfin::la_index> d(1, rd.data() + k);
          f(d, d);
        }
      unfixed(rd, fixed, r);
      if (r.empty())
        continue;
      for (std::size_t j = 0; j < n1; ++j)
        if (mask[i * n1 + j])
        {
          unfixed(p1[j]->cell_dofs(c), fixed, s);
          if (!s.empty())
            f(ArrayView<const dolfin::la_index>(r.size(), r.data()),
              ArrayView<const dolfin::la_index>(s.size(), s.data()));
        }
    }
  }
}

// A's layout from the coupled blocks of V0 x V1 (and the diagonal when square; the rows
// and columns of the dofs `bcs` constrain hold the diagonal alone), its
// pattern then stored in full as explicit zeros, zero entries ignored from then on and a
// new nonzero location an error: an assembly afterwards adds the coupled blocks' values,
// drops the skipped blocks' zeros, and fails on a nonzero outside the pattern.
void init_matrix(PETScMatrix& A, const FunctionSpace& V0, const FunctionSpace& V1,
                 const Paths& paths0, const Paths& paths1, const std::vector<int>& mask,
                 bool square, const std::vector<std::shared_ptr<const DirichletBC>>& bcs)
{
  const std::vector<char> fixed = bcs.empty() ? std::vector<char>() : fixed_dofs(V0, bcs);
  const Mesh& mesh = *V0.mesh();
  auto p0 = parts(V0, paths0), p1 = parts(V1, paths1);
  auto im0 = V0.dofmap()->index_map(), im1 = V1.dofmap()->index_map();
  auto layout = std::make_shared<TensorLayout>(
      mesh.mpi_comm(), std::vector<std::shared_ptr<const IndexMap>>{im0, im1}, 0,
      TensorLayout::Sparsity::SPARSE, TensorLayout::Ghosts::UNGHOSTED);
  SparsityPattern& sp = *layout->sparsity_pattern();
  sp.init({im0, im1});
  blocks(mesh, p0, p1, mask, square, fixed,
         [&sp](ArrayView<const dolfin::la_index> r, ArrayView<const dolfin::la_index> c)
         { sp.insert_local({r, c}); });
  sp.apply();
  A.init(*layout);
  std::vector<double> zeros;
  blocks(mesh, p0, p1, mask, square, fixed,
         [&A, &zeros](ArrayView<const dolfin::la_index> r, ArrayView<const dolfin::la_index> c)
         {
           zeros.assign(r.size() * c.size(), 0.0);
           A.set_local(zeros.data(), r.size(), r.data(), c.size(), c.data());
         });
  A.apply("insert");
  MatSetOption(A.mat(), MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
  MatSetOption(A.mat(), MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);
}

// The owned local dofs condition `bc` constrains, in the order of its boundary values,
// gathered from the ranks that see the facets as assemble_system and apply gather them
std::vector<std::int64_t> fixed_owned(const FunctionSpace& V, std::shared_ptr<const DirichletBC> bc)
{
  DirichletBC::Map values;
  bc->get_boundary_values(values);
  if (dolfin::MPI::size(V.mesh()->mpi_comm()) > 1 && bc->method() != "pointwise")
    bc->gather(values);
  auto im = V.dofmap()->index_map();
  const std::size_t owned = im->size(IndexMap::MapSize::OWNED) * im->block_size();
  std::vector<std::int64_t> out;
  for (const auto& kv : values)
    if (kv.first < owned)
      out.push_back(kv.first);
  return out;
}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("init_matrix", &init_matrix);
  m.def("fixed_owned", &fixed_owned);
}
"""

_MODULE = []


def _module(comm):
    if not _MODULE:
        from femtools.jit import dolfin_module
        _MODULE.append(dolfin_module(_CODE, comm, "femtools.couplings"))
    return _MODULE[0]


def dolfin_matrix(form, mask, bcs=()):
    """An empty dolfin PETScMatrix for bilinear `form` (legacy dolfin) storing the blocks
    `mask` couples, and the diagonal when the test and trial spaces are one: the pattern
    held as explicit zeros, zero entries ignored as they are added, and a nonzero outside
    the pattern PETSc's error. With Dirichlet conditions `bcs` (a square form), the rows
    and columns of the dofs they constrain hold the diagonal alone: fill it with
    `assemble_system(form, L, bcs, A_tensor=A)`, which applies them symmetrically. Assemble
    into it as often as needed. Collective."""
    import dolfin as df
    args = sorted(form.arguments(), key=lambda x: x.number())
    V0, V1 = (x.function_space() for x in args)
    p0, p1 = leaves(V0), leaves(V1)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(p0), len(p1)):
        raise ValueError("a couplings mask of shape {} for spaces of {} x {} components"
                         .format(mask.shape, len(p0), len(p1)))
    comm = V0.mesh().mpi_comm()
    A = df.PETScMatrix(comm)
    square = V0 == V1
    if bcs and not square:
        raise ValueError("Dirichlet conditions on a matrix whose test and trial spaces differ")
    _module(comm).init_matrix(A, V0._cpp_object, V1._cpp_object,
                              [list(p) for p in p0], [list(p) for p in p1],
                              [int(x) for x in mask.ravel()], bool(square), list(bcs))
    return A


def dolfin_fixed_owned(V, bc):
    """The owned local dofs of legacy dolfin space `V` that condition `bc` constrains, in
    the order of its boundary values, gathered across the ranks as dolfin's
    assemble_system and `DirichletBC.apply` gather them (a rank whose facets do not hold an
    owned dof learns it from one whose do). Collective."""
    return np.asarray(_module(V.mesh().mpi_comm()).fixed_owned(V._cpp_object, bc),
                      dtype=np.int64)


# --- DOLFINx: each local dof's component -----------------------------------------------

def dolfinx_labels(V):
    """The component (`leaves` order) of each local dof of DOLFINx space `V` (a Python or
    C++ FunctionSpace; owned dofs, then ghosts), as int8. Raises for a blocked space, or a
    dof no leaf names."""
    V = getattr(V, "_cpp_object", V)
    dm = V.dofmap
    if dm.index_map_bs != 1:
        raise NotImplementedError("couplings over a blocked space")
    n = dm.index_map.size_local + dm.index_map.num_ghosts
    paths = leaves(V, sub=lambda W, i: W.sub([i]),
                   count=lambda W: W.element.num_sub_elements)
    if len(paths) > 127:
        raise NotImplementedError("more than 127 components")
    labels = np.full(n, -1, dtype=np.int8)
    for k, path in enumerate(paths):
        W = V
        for i in path:
            W = W.sub([i])
        labels[np.asarray(W.dofmap.map()).ravel()] = k
    if (labels < 0).any():
        raise RuntimeError("{} local dofs of a space belong to none of its {} components"
                           .format(int((labels < 0).sum()), len(paths)))
    return labels
