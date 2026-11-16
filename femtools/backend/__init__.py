"""One finite-element API on two backends: legacy dolfin and DOLFINx.

    from femtools.backend.dolfin import *      # or: from femtools.backend.dolfinx import *
    V = FunctionSpace(mesh, "CG", 2, constrained_domain=PeriodicBox({0: (0, 1)}))
    u = Function(V)
    A = assemble(inner(grad(TrialFunction(V)), grad(TestFunction(V))) * dx)

Both modules export the names in NAMES, with dolfin's object API: forms in UFL, spaces
with a periodic `constrained_domain`, Functions with `vector()`, PETSc vectors and
matrices over the unique (periodic-folded) dofs, `DirichletBC`, `assemble`,
`PETScKrylovSolver`. `femtools.backend.dolfin` re-exports dolfin and femtools' dolfin
tools; `femtools.backend.dolfinx` implements the same API on DOLFINx (see its docstring).

Where dolfin's methods are DOLFINx properties (`V.mesh()` against `V.mesh`), code
written for both calls the neutral helpers instead: `space_mesh`, `function_space`,
`mesh_comm`, `geometric_dimension`, `element_degree`, `owned_dofs`, `fixed_dofs`; and for
counts over all ranks `global_dim`, `global_num_cells`, `global_num_vertices`,
`marker_counts`.

Neither module is imported by `import femtools.backend`.
"""

NAMES = (
    # backend
    "name",
    # UFL
    "ufl", "Coefficient", "as_vector", "split", "dot", "inner", "grad", "nabla_grad", "div",
    "lhs", "rhs", "dx", "Measure", "FacetNormal", "SpatialCoordinate", "TrialFunction",
    "TrialFunctions", "TestFunction", "TestFunctions", "FiniteElement", "MixedElement",
    "conditional", "cos", "exp", "ge", "gt", "le", "lt", "sin", "sqrt",
    # meshes and boxes
    "Mesh", "RectangleMesh", "UnitSquareMesh", "BoxMesh", "Point", "read_mesh_file",
    "write_mesh_h5", "mesh_coordinates", "hmax", "hmin", "extent", "Faces", "mark_walls",
    "global_num_cells", "global_num_vertices", "marker_counts",
    "SubDomain", "PeriodicBox", "repair_partition", "check_periodic_pairing",
    # spaces, functions, conditions
    "FunctionSpace", "Function", "Constant", "Expression", "DirichletBC", "BoundaryMarkers",
    "AssignedVectorFunction", "set_subfunction", "subfunction", "subspace", "global_dim",
    # linear algebra and solvers
    "Vector", "Matrix", "as_backend_type", "assemble", "assemble_matrix", "apply_bc",
    "form_couplings",
    "interpolate", "nodal_values", "constant_nullspace", "norm", "normalize",
    "PETScKrylovSolver", "ksp", "solve", "weighted_gradient_matrix",
    # runtime
    "MPI", "parameters", "Timer", "list_timings", "TimingClear", "TimingType",
    "DOLFIN_EPS", "DOLFIN_EPS_LARGE",
    # neutral helpers
    "set_default_mesh", "mesh_id", "space_id", "function_space", "space_mesh", "mesh_comm",
    "geometric_dimension", "element_degree", "is_lagrange", "owned_dofs", "fixed_dofs",
    "sync", "collect_garbage", "constant_on",
    # files: dolfin's HDF5 layout and XDMF time series, the same on both backends
    "XDMFFile", "TimeSeriesFile", "write_dof_tables", "write_function_h5",
    "write_functions_h5", "read_function_h5", "read_vector_function_h5", "read_series_step",
    "read_vector_series_step",
)
