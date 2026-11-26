"""feluga.dolfinx: the DOLFINx side of feluga.

    feluga.dolfinx.periodic  periodic_box: slave and master dofs of a periodic box,
                               paired by coordinates (no dolfinx_mpc)
    feluga.dolfinx.unique    UniqueDofs: a space's unique dofs, numbered compactly,
                               with restrict / prolong / fold between Functions and
                               compact PETSc vectors
    feluga.dolfinx.assembly  matrices and right-hand sides over unique dofs, folded
                               during assembly; the constants' null space
    feluga.dolfinx.mixed     subfunction / set_subfunction / subspace: a mixed
                               Function's components, as feluga.mixed
    feluga.dolfinx.cells     a space's cells and dofs in dolfin's (UFC's) order, and
                               dolfin's signature of its element (feluga.tables)
    feluga.dolfinx.checkpoint  Functions in dolfin's HDF5 layout (feluga.dolfin_layout),
                               written and read: checkpoints across the backends and the
                               numbers of ranks
    feluga.dolfinx.mesh      read_mesh / write_mesh: meshes in dolfin's HDF5 layout,
                               numbered as dolfin numbers them; write_boundary_facets, as
                               feluga.boundary_markers writes a facet MeshFunction
    feluga.dolfinx.boundary_markers  BoundaryMarkers: facet markers as dolfin's facet
                               MeshFunction, written and read as feluga.boundary_markers
                               does; compiled_facets: the facets C++ code marks
    feluga.dolfinx.refine    refine_periodic, refine_congruent, check_congruent: Plaza
                               refinement with periodic faces bisected alike, the same cells
                               as feluga.plaza_periodic on dolfin (C++ shared with it,
                               built on first use)
    feluga.dolfinx.expression  CompiledExpression: dolfin's Expression, C++ compiled and
                               cached, evaluated where DOLFINx interpolates
    feluga.dolfinx.xdmf      XDMFSeries: time series in feluga.xdmf.XDMFFile2's layout
                               (quadratic grids kept), the same files as dolfin's
    feluga.dolfinx.xdmf_read read_nodal: a step of such a series (either backend's) into
                               a DOLFINx Function, nodes matched by position

A solver algorithm built on these sees PETSc vectors and matrices over unique dofs only,
as dolfin's constrained_domain gives it, whatever the periodicity. `import
feluga.dolfinx` imports none of the submodules; they need dolfinx, petsc4py, mpi4py,
and scipy (periodic).
"""
