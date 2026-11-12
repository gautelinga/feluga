"""femtools: tools for legacy-dolfin solvers.

    femtools.xdmf        XDMFFile2, a drop-in for dolfin.XDMFFile that keeps P2
    femtools.compress    every HDF5 dataset shuffled and gzipped
    femtools.hdf5_rows   rows of HDF5 datasets read, written and exchanged between ranks
    femtools.boundaries  a boundary facet file: facets with integer markers, written and read
    femtools.boundary_markers  BoundaryMarkers, a facet MeshFunction written and read as one
    femtools.facets      a facet file cut to selected markers (femtools-cut-facets)
    femtools.probes      Probes and GradProbes, from fenicstools (compiled on import)
    femtools.gradient    the weighted gradient matrix, from fenicstools (compiled on import)
    femtools.periodic    the periodic box a constrained_domain is built with, and its checks
    femtools.dolfin_layout  a Mesh or Function written as dolfin's HDF5File writes it, compressed
    femtools.periodic_partition  a parallel mesh partition the periodic dofmap cannot break
    femtools.plaza_periodic  Plaza refinement that bisects identified faces alike
    femtools.refine_periodic  that refinement with the congruence guard; Dorfler threshold
    femtools.hierarchical_transfer  fields carried onto a refined mesh through its parent map
    femtools.fast_interpolation  dolfin.interpolate between Lagrange spaces as a sparse product
    femtools.fast_assembly  dolfin.assemble for a form assembled again and again, compiled once
    femtools.fast_bc     DirichletBC.apply on a vector, the constrained dofs found once
    femtools.ksp         PETSc Krylov solvers set up from options under a prefix, and checked
    femtools.nullspace   the constants on a space, or on a mixed space's component, as a null space
    femtools.mixed       components of mixed Functions on their own spaces; vectors from scalars
    femtools.box         the faces of a box, and the walls of the obstacles inside it
    femtools.swap        a folder written whole, then swapped in by renaming
    femtools.dof_reorder  femtools-build-dof-reorder: dolfin's dof ordering, a library to preload
    femtools.abort       a command's entry point: one rank's failure ends the MPI job
    femtools.memory      freed heap handed back to the operating system
    femtools.dolfinx     the DOLFINx side: periodic_box, unique dofs, assembly over them
    femtools.backend     one finite-element API, dolfin's, on dolfin and on DOLFINx

`import femtools` imports none of them; xdmf, compress, hdf5_rows, boundaries, facets,
dolfin_layout and hierarchical_transfer need numpy and h5py alone, abort, memory, swap and
dof_reorder the standard library, ksp petsc4py.
"""
__version__ = "2026.2.2"
