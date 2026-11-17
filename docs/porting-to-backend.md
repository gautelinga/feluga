# Porting dolfin code to femtools.backend (and so to DOLFINx)

This guide is for code written against legacy dolfin (`import dolfin as df`) that should run
on both dolfin and DOLFINx without being written twice. The route is to replace every
dolfin name with the matching name from `femtools.backend`. That module has dolfin's
object API (`FunctionSpace(mesh, "CG", 1)`, `u.vector()`, `DirichletBC(V, g, markers, 1)`,
`assemble`) on both backends, so most of a port is mechanical. The rest, a dozen dolfin
features DOLFINx lacks, each have a femtools replacement (section 5).

Port to the backend on dolfin first, then switch to DOLFINx. On dolfin, the ported code
should give the same numbers as the original to round-off. Once it does, every difference
that appears on DOLFINx comes from the backend, not from the port.

[addictif](https://github.com/gautelinga/addictif) (Stokes flow, steady advection-diffusion
with SUPG, adaptive refinement, reaction postprocessing, probing on a grid) was ported
this way, and is the worked example below.

## 1. One import module

Give the project one module that every script imports its finite-element names from,
and let `femtools.backend.load` pick the backend:

```python
# myproject/backend.py
import os

import femtools.backend

name = os.environ.get("MYPROJECT_BACKEND", "dolfin")
B = femtools.backend.load(name)                      # femtools.backend.dolfin or .dolfinx
globals().update({n: getattr(B, n) for n in B.NAMES})
__all__ = list(B.NAMES) + ["name", "B"]
```

Scripts then use the bare names (`from myproject.backend import *`) or `B.FunctionSpace`.
Code that receives a mesh or Function from elsewhere can ask `femtools.backend.of(obj)` for
its backend instead.

A command-line switch (`--backend dolfinx`) can set the variable before the first import
of this module. addictif's `run_addictif.py` takes `--backend` out of `argv` and sets
`ADDICTIF_BACKEND`.

Import this module **before mpi4py**. dolfin then initializes MPI itself and finalizes it
after its own objects are freed. If mpi4py comes first, mpi4py finalizes MPI at exit, the
remaining dolfin objects make MPI calls afterwards, and the run aborts after finishing its
work ("MPI_Comm_get_attr() was called after MPI_FINALIZE").

## 2. The mechanical part: `df.X` becomes `X`

These names are on both backends with dolfin's meaning (see `femtools.backend.NAMES`):

- **UFL:** `inner`, `grad`, `div`, `dot`, `dx`, `Measure`, `FacetNormal`,
  `SpatialCoordinate`, `TrialFunction(s)`, `TestFunction(s)`, `sqrt`, `conditional`, and
  `ufl` itself. `FiniteElement`, `VectorElement` and `MixedElement` are also here.
- **Spaces and functions:** `FunctionSpace` (`FunctionSpace(mesh, "CG", 2)`, or
  `FunctionSpace(mesh, element)`), `Function(V, name=...)`, `Constant`, `Expression`
  (C++ code in `x[0]`, compiled), `DirichletBC`, `interpolate`.
  - `VectorFunctionSpace(mesh, "CG", 2)` becomes
    `FunctionSpace(mesh, VectorElement("Lagrange", mesh.ufl_cell(), 2))`.
- **Linear algebra:** `assemble`, `apply_bc`, `PETScKrylovSolver` (with dolfin's
  `parameters`), `as_backend_type`, `solve(a == L, u, bcs)`, `norm`. `u.vector()` supports
  `get_local`, `set_local`, `apply("insert")` and slicing (`u.vector()[:] = ...`).
- **Meshes:** `BoxMesh`, `RectangleMesh`, `Point`, `SubDomain` with `inside(x, on_boundary)`,
  `DOLFIN_EPS(_LARGE)`.
- **Runtime:** `Timer`, `list_timings`, `TimingClear`, `TimingType`, `parameters`, `MPI`.

## 3. Methods that are properties on DOLFINx

dolfin's `V.mesh()` is DOLFINx's `V.mesh`, and so on. Code meant for both calls the
neutral helpers instead:

| dolfin | femtools.backend |
|---|---|
| `u.function_space()` | `function_space(u)` |
| `V.mesh()` | `space_mesh(V)` |
| `mesh.mpi_comm()` | `mesh_comm(mesh)` |
| `mesh.coordinates()` | `mesh_coordinates(mesh)` (a writable view, as in dolfin) |
| `mesh.geometry().dim()` | `geometric_dimension(mesh)` |
| `MPI.sum(comm, mesh.num_cells())` | `global_num_cells(mesh)`, `global_num_vertices(mesh)` |
| `V.dim()` | `global_dim(V)` |

The quickest way to find them is to run the port on DOLFINx: each one fails with
`'_Space' object is not callable` or the like.

## 4. Files

| dolfin | femtools.backend |
|---|---|
| `HDF5File(...).read(mesh, "mesh", False)` | `mesh = read_mesh_file(path)` |
| `HDF5File(...).write(mesh, "mesh")` | `write_mesh_h5(path, mesh, "mesh")` |
| `HDF5File(...).read(u, "u")` | `read_function_h5(path, u, "u")` |
| `HDF5File(...).write(u, "u")` | `write_function_h5(path, u, "u", mode="w")`, then `mode="a"` |
| read a vector field, then `assign(ux, u.sub(0))` | `read_vector_function_h5(path, "u", [ux, uy, uz])` |
| `XDMFFile(comm, path)` | `XDMFFile(mesh_comm(mesh), path)` |

The HDF5 files are in dolfin's layout on both backends. Either backend reads the other's
files on any number of ranks. Continuous Lagrange of degree 1 and 2 (scalar, vector,
mixed) and DG0 are supported.

`XDMFFile` takes dolfin's three `parameters`. With `functions_share_mesh`, successive writes
at one time land on one grid, as in dolfin. Lagrange fields of degree 1 and 2 go on the
nodes, with the midside nodes of degree 2 kept. DG0 fields are written as cell data. DG1
and higher are refused. On DOLFINx every write is a timed grid (t = 0 when no time is
given), so a reader that opens the HDF5 datasets by path, rather than through the XML,
sees `Mesh/0/mesh` where dolfin has `Mesh/mesh`.

A mesh refined in parallel, or on DOLFINx, is written with a `cell_indices` dataset that is
not the identity. That is dolfin's own layout: a function file's `cells` are global cell
indices, and `cell_indices` maps them to the mesh file's rows. A script comparing two runs
by reading the files directly has to go through it, or match by position.

## 5. What DOLFINx does not have

| dolfin | femtools.backend |
|---|---|
| `MeshFunction("size_t", mesh, tdim - 1)` (facet markers) | `BoundaryMarkers(mesh)`, then `SubDomain.mark(markers, 1)`, `set_all`, `rename`, `DirichletBC(V, g, markers, 1)`, `Measure("ds", subdomain_data=markers)` |
| `MeshFunction("bool", mesh, tdim)` + `refine(mesh, markers)` | `refine_cells(mesh, cell_values(indicator) > tol)`: the same cells on both backends and on any number of ranks |
| reading a DG0 Function cell by cell | `cell_values(f)`: the owned cells' values in local cell order, (cells, components) for a vector |
| `LagrangeInterpolator.interpolate(v, u)` (another mesh) | `interpolate(u, V, out=v)`: a Function on another mesh is evaluated at V's points |
| `assemble_system(a, L, bcs)` | `A = assemble(a, couplings="form", bcs=bcs)` (conditions applied symmetrically) and `b = assemble(L)`, then `bc.apply(b)` for each condition (homogeneous conditions) |
| `LinearVariationalSolver` | `A = assemble(a); b = assemble(L); apply_bc(bcs, A, b)`, then `PETScKrylovSolver(method, pc).solve(A, u.vector(), b)` |
| `project(expr, V)` | a mass-matrix solve (below) |
| `KrylovSolver("minres", "hypre_amg")` on Stokes | `femtools.saddle`: MINRES with the block-diagonal preconditioner (`block_preconditioner`, `pressure_mass_diagonal`, `solve_checked`) |
| `fenicstools` / `femtools.probes` `Probes`, `GradProbes` | `Probes`, `GradProbes` from the backend: the same calls (`probes(u)`, `grad(u)`, `array`, `array_grad`, `dump`) |
| `vertex_to_dof_map` for one value per vertex | `femtools.vertex_field` (`read_function`, `is_vertex_layout`) |

**Compiled C++ expressions** (`compile_cpp_code`, `CompiledExpression`) that compute
per-cell quantities have no counterpart. Replace them with exact cell means, assembled:

```python
def cell_mean(expr, DG0):
    v = TestFunction(DG0)
    f = Function(DG0)
    f.vector().set_local(assemble(expr * v * dx).get_local()
                         / assemble(v * dx).get_local())
    f.vector().apply("insert")
    return f
```

For a P1 field `a`, `cell_mean(a, DG0)` is the cell's mean value and
`cell_mean(sqrt(inner(grad(a), grad(a))), DG0)` its |grad a|. Both are exact, and they
equal what addictif's C++ helpers computed to round-off.
`cell_mean(ufl.MaxCellEdgeLength(mesh.ufl_domain()), DG0)` is dolfin's `Cell.h()`. For
|cell mean of a vector|, use `femtools.cell_mean.cell_mean_norm`. A gradient component as
DG0 is `cell_mean(a.dx(i), DG0)`.

`project(expr, V)` is the L2 projection, a solve with V's mass matrix:

```python
def project(expr, V):
    u, v = TrialFunction(V), TestFunction(V)
    f = Function(V)
    PETScKrylovSolver("cg", "jacobi").solve(assemble(inner(u, v) * dx), f.vector(),
                                             assemble(inner(expr, v) * dx))
    return f
```

## 6. Pitfalls

- **Constants and Expressions belong to a mesh on DOLFINx.** dolfin's belong to none.
  Without a mesh, the backend puts them on the first mesh read or made. When a script
  reads a second mesh and assembles on it (a refined mesh, read after the flow mesh), call
  `set_default_mesh(mesh)` first. Otherwise UFL stops with "Multiple domains found". On
  dolfin the call does nothing. `constant_on(value, mesh)` makes one on a given mesh.
- **Interpolation is round-off, not bitwise.** DOLFINx interpolates at points it pushes
  forward from the reference cell, so a field interpolated on DOLFINx can differ in the
  last bit from the same values computed at the mesh file's coordinates. A test that
  asserts equality to the bit on dolfin needs a round-off tolerance on DOLFINx.
- **Point evaluation on a face is ambiguous.** A probe on a face shared by two cells is
  evaluated in one of them. On DOLFINx that is the cell with the smallest original (input)
  cell index, whatever the number of ranks. On dolfin it is the first cell the search
  finds, which can change with the number of ranks. The two backends need not pick the
  same side. For continuous fields this does not matter. For gradients of Lagrange fields
  and for DG fields it does: the value is one side's or the other's. A probe grid that
  lines up with the mesh puts most of its points on faces.
- **A rank's `mesh_coordinates` include vertices other ranks hold too**, on both
  backends. Reductions such as min and max are unaffected; sums and counts over them are
  not (`global_num_vertices` counts each vertex once).

## 7. Checking a port

1. Before touching anything, run the original on dolfin and keep its outputs as the
   reference.
2. Port, run on dolfin serially and compare every output with the reference. Match
   fields by position (vertices for P1, cell centroids for DG0) or through the global
   cell indices (section 4). Expect round-off, or the tolerance of an iterative solver.
3. The same on two ranks.
4. Switch to DOLFINx, serially and on two ranks, and compare again.

Compare fields through position or global numbering, never by the order of the dofs: each
run, each backend and each rank count numbers dofs its own way. Where the output is a
probe of a discontinuous quantity, check that the differing points are exactly those on
faces (section 6) before suspecting the port.
