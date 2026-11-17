# femtools

Tools shared by the FEniCS solvers stokes-beadpack, twoasis and addictif: what more than
one of them needs, written once.

femtools began as a toolbox for **legacy dolfin** (FEniCS 2019.2). It now also runs on
**DOLFINx** (0.11): `femtools.backend` gives both libraries one API, so a solver can be
written once and run on either.

What it is for:

- **One API on two backends.** dolfin-style code (`FunctionSpace`, `DirichletBC`,
  `assemble`, ...) that runs on legacy dolfin or on DOLFINx.
- **Periodic boxes that hold up in parallel.** Periodic spaces on either backend, a mesh
  partition that dolfin's parallel periodic dofmap cannot get wrong, and periodic
  refinement that keeps opposite faces matching.
- **Files that keep what was computed.** XDMF that keeps P2 fields quadratic, dolfin's HDF5
  layout written and read by either backend on any number of ranks, and every dataset
  compressed.
- **Repeated work done once.** Forms, boundary conditions, interpolations and transfers
  set up once and reused, and matrices that store only the blocks a form couples.
- **Very large meshes.** Nothing gathered on one rank, readers that stream, and errors that
  stop every rank instead of hanging the job.

## Quick start

### A solver written once, run on either backend

```python
import femtools.backend

B = femtools.backend.load("dolfinx")                     # or "dolfin"

mesh = B.BoxMesh(B.Point(0, 0, 0), B.Point(1, 1, 1), 8, 8, 8)
box = B.PeriodicBox({0: (0.0, 1.0), 1: (0.0, 1.0)})      # periodic in x and y
V = B.FunctionSpace(mesh, "CG", 2, constrained_domain=box)

walls = B.BoundaryMarkers(mesh, 0)
B.Faces(2, 0.0, 1.0).mark(walls, 1)                      # the planes z = 0 and z = 1
bc = B.DirichletBC(V, B.Constant(0.0), walls, 1)

u, v = B.TrialFunction(V), B.TestFunction(V)
w = B.Function(V)
B.solve(B.inner(B.grad(u), B.grad(v)) * B.dx == B.Constant(1.0) * v * B.dx, w, bcs=[bc])
print(B.assemble(w * B.dx))                              # 1/12, on either backend
```

Both backends export the same names (`femtools.backend.NAMES`). Code that is handed a mesh
or Function finds its backend with `femtools.backend.of(obj)`.
`tests/backend_stokes_worker.py` is a complete periodic Stokes solve written this way.

### Quadratic fields in XDMF (dolfin)

`dolfin.XDMFFile` silently interpolates a P2 field down to its vertices. `XDMFFile2` is a
drop-in replacement that keeps the midside nodes, and can put several fields on one grid:

```python
import dolfin as df
from femtools.xdmf import XDMFFile2

mesh = df.UnitCubeMesh(8, 8, 8)
u = df.Function(df.VectorFunctionSpace(mesh, "CG", 2), name="u")
p = df.Function(df.FunctionSpace(mesh, "CG", 1), name="p")

with XDMFFile2(mesh.mpi_comm(), "up.xdmf") as f:
    f.write(u, 0.0)
    f.write((u, p), 1.0)                                 # two fields on one grid
```

In ParaView, raise "Nonlinear Subdivision Level" to see the quadratic field. On DOLFINx,
`femtools.dolfinx.xdmf.XDMFSeries` writes the same XML for a time series (but for the HDF5
file's name).

### A periodic box and probes (dolfin)

```python
from femtools.periodic import periodic_box
from femtools.probes import Probes

V = df.FunctionSpace(mesh, "CG", 2, constrained_domain=periodic_box(
    upper=(Lx, Ly), periodic=(True, False), lower=(0, 0)))

probes = Probes(points.flatten(), V)                     # located once
probes(u)                                                # evaluated each time step
values = probes.array()
```

### Boundary markers that are written and read back (dolfin)

```python
from femtools.boundaries import BoundaryMarkers

markers = BoundaryMarkers(mesh)                          # where a facet MeshFunction was
markers.mark(df.CompiledSubDomain("on_boundary && near(x[2], 0)"), 5)
bc = df.DirichletBC(V, g, markers, 5)
markers.write("boundaries.xdmf")                         # the marked exterior facets
markers = BoundaryMarkers.read("boundaries.xdmf", mesh)
```

### Command-line tools

```bash
femtools-cut-facets boundaries.xdmf walls.xdmf --markers 3,5,7:106   # keep some markers
femtools-build-dof-reorder --out libdof_reorder_star.so    # "Large meshes and long jobs"
LD_PRELOAD=$PWD/libdof_reorder_star.so python3 solve.py
```

## What is in it

Each module's docstring has the details, the trade-offs and the measurements.

### Backends

| module | what it is for |
|---|---|
| `backend` | `load(name)`, `of(obj)`, `is_dolfinx(obj)` and `NAMES`: the one API |
| `backend.dolfin` | that API on legacy dolfin (mostly dolfin itself, plus femtools' tools) |
| `backend.dolfinx` | that API on DOLFINx, built on the `dolfinx.*` modules below |
| `tables`, `ufc` | a mesh's cells and a space's dofs as arrays, in dolfin's order, on either backend |

### Files

| module | what it is for |
|---|---|
| `xdmf` | `XDMFFile2` (P2 kept, several fields on one grid); `grids` and `parse_xdmf` to find a series' arrays; `merge_series` to join a run and its restarts |
| `xdmfread` | read compressed datasets fast, inflating chunks on a thread pool |
| `compress` | compressed datasets, the default everywhere |
| `hdf5_rows` | rows of datasets read, written and exchanged between ranks, no rank holding the whole |
| `dolfin_layout` | a Mesh or Function in the layout `dolfin.HDF5File` uses, compressed |
| `layout_reader` | that layout read with numpy and h5py alone, streamed |
| `vertex_field` | a P1 field stored as one value per mesh vertex: much smaller, either backend |
| `digest` | a SHA-256 of what a stored Mesh or Function holds, the same on any number of ranks |
| `boundaries`, `boundary_markers` | boundary facet files (the layout partrac reads) and `BoundaryMarkers`, a facet MeshFunction that writes and reads them |
| `facets` | `femtools-cut-facets`: a facet file cut down to some markers |

### Periodic meshes and refinement

| module | what it is for |
|---|---|
| `periodic` | `periodic_box`, the compiled `constrained_domain` of a periodic box (any lower corner); `faces_congruent` |
| `periodic_core` | the backend-free part: the pairing rule (`classify`), tolerances, congruence checks |
| `box` | `Faces` (planes of a box) and `mark_walls` (every boundary facet off the box's faces) |
| `periodic_partition` | a parallel mesh read and partition (ParMETIS) that dolfin's periodic dofmap cannot get wrong, and the check that it did not |
| `plaza_periodic`, `refine_periodic` | periodic Plaza refinement: opposite faces bisected alike, with the parent-cell map |
| `hierarchical_transfer` | P1 and P2 fields moved to a refined mesh through ancestor cells, no point search |
| `nested_transfer` | transfers between a mesh and its uniform refinement, as matrices built once, either backend |

### Repeated work, faster

| module | what it is for |
|---|---|
| `fast_assembly` | a form assembled every time step compiled once; constant matrices assembled once and shared |
| `fast_bc` | `DirichletBC.apply` with the constrained dofs found once |
| `fast_interpolation` | interpolation between Lagrange spaces as one sparse product built once; `p1_to_p2`, either backend |
| `couplings` | matrices that store only the blocks a form couples (a Stokes operator keeps under half of the mixed pattern) |

### Solvers

| module | what it is for |
|---|---|
| `ksp` | PETSc Krylov solvers from options under their own prefix, checked; `scoped` options; tuned BoomerAMG settings |
| `saddle` | the block preconditioner of a Taylor-Hood Stokes system and its checked solve, either backend and layout |
| `shared_amg` | one BoomerAMG for all velocity components of that preconditioner |
| `nullspace` | the constants on a space, or on one component of a mixed space, as a null space |
| `mixed` | a mixed Function's components through their collapsed spaces; vector Functions from scalars |

### Point values and gradients

| module | what it is for |
|---|---|
| `probes` | a Function's values and gradients at fixed points, located once (from fenicstools) |
| `gradient` | the weighted gradient matrix: a P1 field's nodal derivative as one product (from fenicstools) |
| `recovery` | a continuous gradient recovered by averaging in a periodic P1 space, either backend |
| `cell_mean` | the size of each cell's mean of a P1 or P2 velocity, exactly, either backend |

### Large meshes and long jobs

| module | what it is for |
|---|---|
| `abort` | `run(main)`: an exception on one MPI rank ends the job instead of hanging it |
| `jit`, `lru` | compiled extensions built on one rank and loaded on all; bounded and weak caches |
| `dof_reorder` | `femtools-build-dof-reorder`: dolfin's dof reordering without its memory peak at large sizes |
| `swap` | a folder written aside and swapped in, never left half-written by a kill |
| `memory`, `phases` | freed heap handed back, and a line per phase of a run with its time and memory |
| `garbage` | PETSc objects Python dropped, destroyed collectively |

### DOLFINx internals

What `backend.dolfinx` is built on; useful directly when writing plain DOLFINx code.

| module | what it is for |
|---|---|
| `dolfinx.periodic` | the slave and master dofs of a periodic box, paired by position in parallel (no dolfinx_mpc) |
| `dolfinx.unique` | `UniqueDofs`: periodic slaves folded onto their masters, numbered compactly as dolfin's `constrained_domain` numbers them |
| `dolfinx.assembly` | matrices and vectors over unique dofs, folded during assembly |
| `dolfinx.mixed` | a mixed Function's components through their collapsed spaces |
| `dolfinx.checkpoint`, `dolfinx.mesh` | Functions and meshes in dolfin's HDF5 layout, each backend reading the other's |
| `dolfinx.xdmf`, `dolfinx.xdmf_read` | XDMF series in `XDMFFile2`'s layout, written and read |
| `dolfinx.refine` | periodic Plaza refinement on DOLFINx |
| `dolfinx.expression` | dolfin's C++ `Expression` for DOLFINx |
| `dolfinx.boundary_markers`, `dolfinx.cells` | facet markers as on dolfin; a space's cells and dofs in dolfin's order |
| `dolfinx.compiled` | the C++ extensions above, built on first use (`python3 -m femtools.dolfinx.compiled` builds them ahead) |

## How the two backends differ

The API is the same; underneath, a few things differ, visibly:

- **Periodicity.** dolfin builds periodic spaces into its dofmap (`constrained_domain`). Its
  parallel version can lose the pairing on some partitions, so a parallel periodic dolfin
  mesh should be read through `read_mesh_file(..., periodic=...)`, which partitions it
  safely (`periodic_partition`). DOLFINx spaces are made periodic by femtools
  (`dolfinx.unique`), and pair correctly on any partition.
- **Dof order.** The two number dofs differently, so iteration counts of AMG or ILU solves
  differ a little between backends. Answers agree to the solver tolerance.
- **Stored vectors.** In dolfin's HDF5 layout, a component split off a periodic mixed space
  is stored on all its nodes by dolfin and on its unique (periodic) dofs by DOLFINx. Both
  files name each cell's dofs, and every reader here goes through those.
- **XDMF files.** On dolfin `XDMFFile` is femtools' `XDMFFile2` (P2 kept, several fields on
  one grid), handing what it does not lay out (DG, P3, a whole mixed Function) to dolfin's
  own writer; it reads and checkpoints too. On DOLFINx it writes facet markers and P1/P2
  Functions alone, every write a timed grid (t = 0 when none is given).
- **Methods and properties.** dolfin's `V.mesh()` is DOLFINx's `V.mesh`. Code meant for both
  uses the neutral helpers (`space_mesh`, `function_space`, `mesh_comm`, `owned_dofs`,
  `global_dim`, ...) and `as_backend_type(x).vec()` for a PETSc vector.

## Requirements

- **Always:** numpy and h5py. `import femtools` needs nothing else; dolfin, DOLFINx,
  mpi4py and scipy are imported only where used.
- **For the dolfin side:** legacy dolfin 2019.2, with a compiler and `libdolfin-dev`, since
  several modules compile C++ on first use (dolfin's JIT, or cppimport for `probes` and
  `gradient`).
- **For the DOLFINx side:** DOLFINx 0.11 with its headers, CMake and nanobind: the
  extensions in `dolfinx.compiled` build on first use and are cached in `FEMTOOLS_JIT_DIR`
  (default `~/.cache/femtools`). That cache must be on a file system every rank sees.
- **Parallel HDF5:** h5py built with MPI (Debian/Ubuntu: `python3-h5py-mpi`). Without it
  every file goes through rank 0, which is correct but slow on large meshes; femtools says
  so once.
- **Very large meshes:** past 2^31 cells, ParMETIS with a 64-bit `idx_t`; past 2^31 dofs,
  PETSc and hypre with 64-bit indices.

## Install

Into the interpreter that has dolfin or DOLFINx, without dependencies (a pip-built mpi4py
would be built against another MPI than theirs):

```bash
python3 -m pip install --user --break-system-packages --no-deps --no-build-isolation -e .
```

## Tests

```bash
python3 -m pytest              # the quick tests, a minute or two
python3 -m pytest -m slow      # the mpirun and compile tests, minutes
python3 -m pytest -m ""        # everything
python3 -m pytest -m "" -n 4   # everything, in parallel (pytest-xdist)
```

Tests for a backend that is not installed skip themselves.

## Credits and licence

- **Probes and the weighted gradient matrix:** Mikael Mortensen's
  [fenicstools](https://github.com/mikaem/fenicstools), Copyright (C) 2011–2013 Mikael
  Mortensen. The probes come through addictif's extended copy (adding `GradProbes`).
- **Periodic Plaza refinement and the dof reordering:** adapted from
  [DOLFIN](https://bitbucket.org/fenics-project/dolfin) 2019.2, Copyright (C) 2008–2015
  Anders Logg, Ola Skavhaug, Garth N. Wells and others.

Both are LGPL-3.0-or-later, and their files keep their headers; NOTICE lists what came from
where and what changed. femtools is LGPL-3.0-or-later (COPYING.LESSER, COPYING).
