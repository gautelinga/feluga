"""femtools.dolfinx: the DOLFINx side of femtools.

    femtools.dolfinx.periodic  periodic_box: slave and master dofs of a periodic box,
                               paired by coordinates (no dolfinx_mpc)
    femtools.dolfinx.unique    UniqueDofs: a space's unique dofs, numbered compactly,
                               with restrict / prolong / fold between Functions and
                               compact PETSc vectors
    femtools.dolfinx.assembly  matrices and right-hand sides over unique dofs, folded
                               during assembly; Dirichlet values; the constants' null space
    femtools.dolfinx.mixed     subfunction / set_subfunction / subspace /
                               AssignedVectorFunction, as femtools.mixed

A solver algorithm built on these sees PETSc vectors and matrices over unique dofs only,
as dolfin's constrained_domain gives it, whatever the periodicity. `import
femtools.dolfinx` imports none of the submodules; they need dolfinx, petsc4py, mpi4py,
and scipy (periodic).
"""
