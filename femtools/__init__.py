"""femtools: tools for legacy-dolfin and DOLFINx solvers.

    femtools.xdmf        XDMFFile2, a drop-in for dolfin.XDMFFile that keeps P2
    femtools.compress    every HDF5 dataset shuffled and gzipped
    femtools.hdf5_rows   rows of HDF5 datasets read, written and exchanged between ranks
    femtools.boundaries  a boundary facet file: facets with integer markers, written and read
    femtools.boundary_markers  BoundaryMarkers, a facet MeshFunction written and read as one
    femtools.facets      a facet file cut to selected markers (femtools-cut-facets)
    femtools.probes      Probes and GradProbes, from fenicstools (compiled at first use)
    femtools.gradient    the weighted gradient matrix, from fenicstools (compiled on import)
    femtools.periodic    the periodic box a constrained_domain is built with, and its checks
    femtools.dolfin_layout  a Mesh or Function written as dolfin's HDF5File writes it, compressed
    femtools.layout_reader  a Mesh and its Functions in that layout, read with numpy and h5py
    femtools.vertex_field  a P1 field stored by the mesh file's vertex rows
    femtools.digest      SHA-256 digests of a stored Mesh or Function
    femtools.tables      a mesh's cells and a space's dofs as arrays, alike on both backends
    femtools.periodic_partition  a parallel mesh partition the periodic dofmap cannot break
    femtools.plaza_periodic  Plaza refinement that bisects identified faces alike
    femtools.refine_periodic  that refinement with the congruence guard; Dorfler threshold
    femtools.hierarchical_transfer  fields carried onto a refined mesh through its parent map
    femtools.nested_transfer  transfers between a mesh and its uniform refinement
    femtools.fast_interpolation  dolfin.interpolate between Lagrange spaces as a sparse product;
                         p1_to_p2 from the dofmaps
    femtools.fast_assembly  dolfin.assemble for a form assembled again and again, compiled once
    femtools.fast_bc     DirichletBC.apply on a vector, the constrained dofs found once
    femtools.ksp         PETSc Krylov solvers set up from options under a prefix, and checked
    femtools.couplings   matrices storing only the blocks a form couples (and Dirichlet dofs)
    femtools.shared_amg  one BoomerAMG for a Stokes operator's velocity components
    femtools.saddle      a Stokes system's block preconditioner, either backend, and its solve
    femtools.recovery    the ZZ gradient recovered in a periodic P1 space
    femtools.cell_mean   |cell mean of u| for a P1 or P2 velocity
    femtools.nullspace   the constants on a space, or on a mixed space's component, as a null space
    femtools.mixed       components of mixed Functions on their own spaces; vectors from scalars
    femtools.box         the faces of a box, and the walls of the obstacles inside it
    femtools.swap        a folder written whole, then swapped in by renaming
    femtools.dof_reorder  femtools-build-dof-reorder: dolfin's dof ordering, a library to preload
    femtools.abort       a command's entry point: one rank's failure ends the MPI job
    femtools.memory      freed heap handed back to the operating system
    femtools.phases      a line a phase of a run (time, memory), and mpi_print
    femtools.garbage     the PETSc objects Python dropped, destroyed collectively
    femtools.jit         compiled extensions built on one rank, loaded on all
    femtools.lru         bounded and weak caches
    femtools.dolfinx     the DOLFINx side: periodic_box, unique dofs, assembly over them
    femtools.backend     one finite-element API, dolfin's, on dolfin and on DOLFINx

`import femtools` imports none of them; xdmf, compress, hdf5_rows, boundaries, facets,
dolfin_layout, layout_reader, digest, vertex_field (its readers), phases, couplings,
recovery, cell_mean and hierarchical_transfer import with numpy and h5py alone (dolfin,
DOLFINx and mpi4py where used); abort, memory, swap and dof_reorder need the standard
library, ksp petsc4py.
"""
__version__ = "2026.2.4"
