"""`|(1/V) integral u|` per cell, compiled, and correct for a P2 velocity as well as P1.

This exists because addictif's `AbsVecCell` -- which is what supplies `|u|` to the SUPG
parameter tau -- averages the element's nodal coefficients with **uniform** weight
`1/ncoeff`. For P1 that is exactly right and is simultaneously the centroid value and
the cell mean, measured identical to an assembled mean to 5.5e-16 on the real packing.
For P2 it is neither, because the P2 basis functions do not have equal integrals over
the element. On a tetrahedron, with the barycentric integral
`int prod(lambda_i^a_i) dV = V * prod(a_i!) * 3! / (sum(a_i) + 3)!`:

    vertex function   N_i  = lambda_i (2 lambda_i - 1)   ->  (1/V) int N_i  = -1/20
    midside function  N_ij = 4 lambda_i lambda_j         ->  (1/V) int N_ij = +1/5

and `4*(-1/20) + 6*(+1/5) = 1`, as a set of mean weights must sum to. `AbsVecCell` uses
`+1/10` for all ten.

**The error is one-sided in this geometry, which is why it matters.** 27% of vertices in
a bead pack lie on the no-slip solid where u = 0 (DESIGN-NOTES.md, "Steady
advection-diffusion"), and uniform weighting gives those zeros `+0.1` where the true mean
gives them `-0.05`. Measured on the real packing's stored P2 velocity against the
assembled cell mean of the same field: max deviation 10.6%, median 1.7%, and it
**under-states `|u|` in 96.3% of cells**. tau carries a `max(0, 1 - 1/Pe_el)` factor and
`Pe_el = |u| h / 2D`, so under-stating `|u|` pushes cells below `Pe_el = 1` and switches
their stabilisation off: at D=5e-5, 20985 cells (1.62%) change stabilised state and
20955 of them -- 99.9% -- lose stabilisation they should have.

None of that is a bug today, because `u_proj_` is P1 on every current path. It is a
prerequisite for advecting at P2.

**Why compiled rather than assembled.** The exact mean is also had from three DG0
assemblies, `assemble(u[i]*v*dx) / assemble(v*dx)`, which is how the claims above were
measured and is what `tests/test_cell_mean.py` checks against. But that builds three
global forms and three vectors per solve where this walks the cells once, and the P1
path has to stay bit-identical to addictif's helper -- which it is, since for `ncoeff=4`
this computes the same uniform average with the same restrict call in the same order.

Everything else is addictif's `AbsVecCell` unchanged: the same `Function::restrict` into
a component-blocked coefficient array, the same `sqrt(ux^2 + uy^2 + uz^2)` of the
averaged vector (the norm of the mean, not the mean of the norm), the same DG0 target.
An element that is neither P1 nor P2 **raises** rather than guessing a weight set --
there is no meaningful default, and a wrong one is invisible.

On DOLFINx (femtools.backend.dolfinx) the same means and norm are taken in numpy from the
space's cell dofs, in basix's local order (the vertices first, as FFC's). From
stokes-beadpack (`ade/cell_mean_cpp.py`), where SUPG's tau takes `|u|` from it.
"""

import numpy as np

from femtools.phases import mpi_print

_CODE = r"""
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/common/constants.h>
#include <stdexcept>
#include <vector>
#include <cmath>

// The cell-mean weights of the Lagrange basis on a tetrahedron: (1/V) int N_a dV.
// P1: 1/4 each. P2: -1/20 on the four vertex functions, +1/5 on the six midside ones,
// in FFC's local dof order, which puts the vertices first. Each set sums to one.
//
// Returned as a pointer into static storage rather than filled into a vector: this is
// called once per cell, and a heap allocation per cell is 21 million of them on the
// meshes this runs on. The tables are read-only and identical on every rank.
static const double P1_MEAN_W[4]  = {0.25, 0.25, 0.25, 0.25};
static const double P2_MEAN_W[10] = {-0.05, -0.05, -0.05, -0.05,
                                     0.2, 0.2, 0.2, 0.2, 0.2, 0.2};

static const double* mean_weights(std::size_t ncoeff)
{
  if (ncoeff == 4)  return P1_MEAN_W;
  if (ncoeff == 10) return P2_MEAN_W;
  throw std::runtime_error(
    "CellMeanNorm: the cell-mean weights are known for P1 (4 dofs a component) and "
    "P2 (10) on a tetrahedron, and this element has a different number. Add its "
    "weights rather than letting it average uniformly, which is right for neither.");
}

class CellMeanNorm : public dolfin::Expression
{
public:

  CellMeanNorm() : dolfin::Expression() {}

  void eval(Eigen::Ref<Eigen::VectorXd> values,
            Eigen::Ref<const Eigen::VectorXd> x,
            const ufc::cell& ufc_cell) const override
  {
    const std::size_t cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*u->function_space()->mesh(), cell_index);
    const dolfin::FiniteElement element = *u->function_space()->element();

    std::vector<double> coordinate_dofs;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs);

    const std::size_t ncoeff = element.space_dimension() / 3;
    const double* w = mean_weights(ncoeff);   // throws before anything is allocated

    // stack, not heap: 30 doubles covers P1 and P2, and this runs once per cell
    double coefficients_[30];
    u->restrict(coefficients_, element, dolfin_cell,
                coordinate_dofs.data(), ufc_cell);

    double ub[3] = {0.0, 0.0, 0.0};
    for (std::size_t c = 0; c < 3; ++c)
      for (std::size_t a = 0; a < ncoeff; ++a)
        ub[c] += w[a] * coefficients_[c * ncoeff + a];

    values[0] = std::sqrt(ub[0] * ub[0] + ub[1] * ub[1] + ub[2] * ub[2]);
  }

  std::shared_ptr<dolfin::Function> u;
};

// The same, for a velocity held as three scalar Functions on one scalar space (the
// solver's memory-lean P1 velocity, `solve.py::p1_velocity`). Each component is
// restricted on its own -- its coefficients are the vector's block for that component,
// in the same local order -- and averaged and normed in exactly the loop above, so
// the result is the vector form's to the bit.
class CellMeanNormComponents : public dolfin::Expression
{
public:

  CellMeanNormComponents() : dolfin::Expression() {}

  void eval(Eigen::Ref<Eigen::VectorXd> values,
            Eigen::Ref<const Eigen::VectorXd> x,
            const ufc::cell& ufc_cell) const override
  {
    const std::size_t cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*ux->function_space()->mesh(), cell_index);
    const dolfin::FiniteElement element = *ux->function_space()->element();

    std::vector<double> coordinate_dofs;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs);

    const std::size_t ncoeff = element.space_dimension();
    const double* w = mean_weights(ncoeff);

    const dolfin::Function* comps[3] = {ux.get(), uy.get(), uz.get()};
    double coefficients_[30];
    for (std::size_t c = 0; c < 3; ++c)
      comps[c]->restrict(coefficients_ + c * ncoeff, element, dolfin_cell,
                         coordinate_dofs.data(), ufc_cell);

    double ub[3] = {0.0, 0.0, 0.0};
    for (std::size_t c = 0; c < 3; ++c)
      for (std::size_t a = 0; a < ncoeff; ++a)
        ub[c] += w[a] * coefficients_[c * ncoeff + a];

    values[0] = std::sqrt(ub[0] * ub[0] + ub[1] * ub[1] + ub[2] * ub[2]);
  }

  std::shared_ptr<dolfin::Function> ux, uy, uz;
};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<CellMeanNorm, std::shared_ptr<CellMeanNorm>, dolfin::Expression>
    (m, "CellMeanNorm")
    .def(py::init<>())
    .def_readwrite("u", &CellMeanNorm::u);
  py::class_<CellMeanNormComponents, std::shared_ptr<CellMeanNormComponents>,
             dolfin::Expression>
    (m, "CellMeanNormComponents")
    .def(py::init<>())
    .def_readwrite("ux", &CellMeanNormComponents::ux)
    .def_readwrite("uy", &CellMeanNormComponents::uy)
    .def_readwrite("uz", &CellMeanNormComponents::uz);
}
"""

_MODULE = [None]                 # the compiled module, built once


def cell_mean_norm(u, DG0):
    """`|(1/V) integral u|` on every cell of `u`'s mesh, as a DG0 Function.

    Exact for a P1 or P2 vector `u` on tetrahedra; raises for anything else. For P1 it
    is addictif's `AbsVecCell` to the bit, which is what keeps every recorded number on
    the P1 path where it was. `u` may also be `as_vector` of three scalar Functions on
    one space (`scalar_components`), which gives the vector form's numbers bit for bit.

    A working `g++` and `libdolfin-dev` are assumed, not hoped for -- the same
    assumption `femtools.periodic` and `femtools.plaza_periodic` make, and the same
    one addictif's own `CompiledExpression` helpers make one import earlier.
    """
    from femtools.tables import is_dolfinx
    if is_dolfinx(DG0):
        return _cell_mean_norm_dolfinx(u, DG0)
    import dolfin as df
    _check_legacy(u)                         # before compiling, the same on every rank
    if _MODULE[0] is None:
        from femtools.jit import dolfin_module
        _MODULE[0] = dolfin_module(_CODE, DG0.mesh().mpi_comm(), "femtools.cell_mean")
        mpi_print("cell-mean |u|: compiled")
    # `u=` through CompiledExpression, not by assigning the attribute: dolfin's
    # wrapper unwraps the Python Function to the cpp one on the way in, and the raw
    # setter takes only the latter. This is addictif's own call shape for AbsVecCell.
    comps = scalar_components(u)
    if comps is not None:
        # a velocity held as three scalar Functions (`solve.py::p1_velocity`): the
        # same average and norm, component by component, bit for bit
        ux, uy, uz = comps
        return df.interpolate(df.CompiledExpression(
            _MODULE[0].CellMeanNormComponents(), ux=ux, uy=uy, uz=uz, degree=0), DG0)
    return df.interpolate(
        df.CompiledExpression(_MODULE[0].CellMeanNorm(), u=u, degree=0), DG0)


def _check_legacy(u):
    """Refuse, on every rank alike, anything but a P1 or P2 Lagrange velocity of three
    components on tetrahedra: the compiled weights know no other element, and a 2D P2
    vector (12 dofs) would read as three P1 components (and the C++ throws only on the
    ranks holding cells)."""
    comps = scalar_components(u)
    spaces = [_space(c) for c in comps] if comps is not None else [_space(u)]
    for V in spaces:
        el = V.ufl_element()
        mesh = V.mesh()
        shape = el.value_shape()
        base = el.sub_elements()[0] if shape else el
        ok = (mesh.ufl_cell().cellname() == "tetrahedron"
              and shape == (() if comps is not None else (3,))
              and base.family() in ("Lagrange", "Q") and base.degree() in (1, 2))
        if not ok:
            raise ValueError("cell_mean_norm: the cell-mean weights are known for a P1 or P2 "
                             "Lagrange velocity of three components on tetrahedra, not {} "
                             "on {}".format(el, mesh.ufl_cell().cellname()))


def scalar_components(u):
    """The three scalar Functions of a velocity given as `as_vector` of them (the
    solver's memory-lean P1 velocity), or None for anything else -- a vector Function
    in particular."""
    import importlib
    ufl = importlib.import_module(type(u).__module__.split(".")[0])
    if not hasattr(ufl, "classes") or not isinstance(u, ufl.classes.ListTensor):
        return None
    comps = tuple(u.ufl_operands)
    if len(comps) != 3 or not all(hasattr(c, "vector") and callable(c.vector)
                                  for c in comps):
        return None
    # the compiled class restricts all three with the first one's element and cells
    space = [_space(c) for c in comps]
    if any(s != space[0] for s in space[1:]):
        raise ValueError("the velocity's three components are on different spaces")
    return comps


def _space(f):
    return f.function_space() if callable(f.function_space) else f.function_space


# the cell-mean weights of the Lagrange basis on a tetrahedron, in the local order both
# FFC and basix give it (the vertices first): as the C++ tables above
_MEAN_WEIGHTS = {4: np.full(4, 0.25), 10: np.r_[np.full(4, -0.05), np.full(6, 0.2)]}


def _cell_mean_norm_dolfinx(u, DG0):
    """cell_mean_norm on femtools.backend.dolfinx: each local cell's nodal values from the
    dofmap, weighted, the norm of the mean into DG0's cell dofs."""
    from femtools.backend.dolfinx import Function
    comps = scalar_components(u)
    funcs = comps if comps is not None else (u,)
    if any(hasattr(f, "_parent") or not hasattr(f, "sync") for f in funcs):
        raise ValueError("cell_mean_norm: a Function of its own space (collapse a mixed "
                         "Function's component first)")
    for f in funcs:
        f.sync()
    V = funcs[0].function_space
    dofs = np.asarray(V.dofmap.list)                       # (local cells, nodes)
    w = _MEAN_WEIGHTS.get(dofs.shape[1])
    if w is None:
        raise ValueError("cell_mean_norm: the cell-mean weights are known for P1 (4 dofs "
                         "a component) and P2 (10) on a tetrahedron, not {}"
                         .format(dofs.shape[1]))
    if comps is not None:
        mean = np.stack([f.x.array[dofs] @ w for f in funcs], axis=1)
    else:
        bs = V.dofmap.bs
        if bs != 3:
            raise ValueError("cell_mean_norm: a vector of three components, not {}".format(bs))
        vals = u.x.array.reshape(-1, bs)[dofs]                 # (cells, nodes, 3)
        mean = np.einsum("cnk,n->ck", vals, w)
    out = Function(DG0)
    cells = np.asarray(DG0.dofmap.list)[:, 0]        # each local cell's DG0 dof
    if len(cells) != len(mean) or DG0.mesh is not V.mesh:
        raise ValueError("cell_mean_norm: DG0 on another mesh than the velocity")
    out.x.array[cells] = np.sqrt((mean ** 2).sum(axis=1))
    return out.restrict()
