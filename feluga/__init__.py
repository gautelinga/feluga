"""feluga, finite element utilities with a generic API: tools for legacy-dolfin and DOLFINx
solvers.

    feluga.xdmf        XDMFFile2, a drop-in for dolfin.XDMFFile that keeps P2
    feluga.compress    every HDF5 dataset shuffled and gzipped
    feluga.hdf5_rows   rows of HDF5 datasets read, written and exchanged between ranks
    feluga.boundaries  a boundary facet file: facets with integer markers, written and read
    feluga.boundary_markers  BoundaryMarkers, a facet MeshFunction written and read as one
    feluga.facets      a facet file cut to selected markers (feluga-cut-facets)
    feluga.probes      Probes and GradProbes, from fenicstools (compiled at first use)
    feluga.gradient    the weighted gradient matrix, from fenicstools (compiled on import)
    feluga.periodic    the periodic box a constrained_domain is built with, and its checks
    feluga.dolfin_layout  a Mesh or Function written as dolfin's HDF5File writes it, compressed
    feluga.layout_reader  a Mesh and its Functions in that layout, read with numpy and h5py
    feluga.vertex_field  a P1 field stored by the mesh file's vertex rows
    feluga.digest      SHA-256 digests of a stored Mesh or Function
    feluga.tables      a mesh's cells and a space's dofs as arrays, alike on both backends
    feluga.periodic_partition  a parallel mesh partition the periodic dofmap cannot break
    feluga.plaza_periodic  Plaza refinement that bisects identified faces alike
    feluga.refine_periodic  that refinement with the congruence guard; Dorfler threshold
    feluga.hierarchical_transfer  fields carried onto a refined mesh through its parent map
    feluga.nested_transfer  transfers between a mesh and its uniform refinement
    feluga.fast_interpolation  dolfin.interpolate between Lagrange spaces as a sparse product;
                         p1_to_p2 from the dofmaps
    feluga.fast_assembly  dolfin.assemble for a form assembled again and again, compiled once
    feluga.fast_bc     DirichletBC.apply on a vector, the constrained dofs found once
    feluga.ksp         PETSc Krylov solvers set up from options under a prefix, and checked
    feluga.couplings   matrices storing only the blocks a form couples (and Dirichlet dofs)
    feluga.shared_amg  one BoomerAMG for a Stokes operator's velocity components
    feluga.saddle      a Stokes system's block preconditioner, either backend, and its solve
    feluga.recovery    the ZZ gradient recovered in a periodic P1 space
    feluga.cell_mean   |cell mean of u| for a P1 or P2 velocity
    feluga.nullspace   the constants on a space, or on a mixed space's component, as a null space
    feluga.mixed       components of mixed Functions on their own spaces; vectors from scalars
    feluga.box         the faces of a box, and the walls of the obstacles inside it
    feluga.swap        a folder written whole, then swapped in by renaming
    feluga.dof_reorder  feluga-build-dof-reorder: dolfin's dof ordering, a library to preload
    feluga.abort       a command's entry point: one rank's failure ends the MPI job
    feluga.memory      freed heap handed back to the operating system
    feluga.phases      a line a phase of a run (time, memory), and mpi_print
    feluga.garbage     the PETSc objects Python dropped, destroyed collectively
    feluga.jit         compiled extensions built on one rank, loaded on all
    feluga.lru         bounded and weak caches
    feluga.dolfinx     the DOLFINx side: periodic_box, unique dofs, assembly over them
    feluga.backend     one finite-element API, dolfin's, on dolfin and on DOLFINx

`import feluga` imports none of them; xdmf, compress, hdf5_rows, boundaries, facets,
dolfin_layout, layout_reader, digest, vertex_field (its readers), phases, couplings,
recovery, cell_mean and hierarchical_transfer import with numpy and h5py alone (dolfin,
DOLFINx and mpi4py where used); abort, memory, swap and dof_reorder need the standard
library, ksp petsc4py.
"""
__version__ = "2026.2.4"
