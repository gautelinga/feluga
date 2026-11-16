"""Block-diagonal preconditioning of a Stokes saddle-point system, on either backend.

    diag_M = pressure_mass_diagonal(W, mu)                # on the pressure's owned dofs
    pre = block_preconditioner(ksp, A, W, diag_M, amg_options, kind="shared")
    iterations = solve_checked(ksp, b, x)                 # raises unless it converged
    pre.destroy()

The preconditioner is diag(velocity Laplacian, pressure mass matrix), which Silvester and
Wathen show optimal for MINRES, each block approximated as cheaply as it can be stored: the
velocity by BoomerAMG on the operator's own component blocks (no second matrix assembled),
the pressure by Jacobi on diag(M) alone, one vector assembly (for P1 on simplices diag(M) is
exactly 2/(d + 2) of the lumped mass: vol/10 against vol/4 in a tetrahedron's row).

`kind="shared"` applies one AMG to every velocity component (`femtools.shared_amg`: the
scalar Laplacian held once, one hierarchy); `kind="fieldsplit"` gives each component its
own through PETSc's additive fieldsplit -- the same preconditioner, the AMG settings set
for its setup alone (`femtools.ksp.scoped`). `W` is the Taylor-Hood space of either layout,
[P2^3, P1] or [P2, P2, P2, P1]: the pressure is its last component, the velocity every
scalar component before it (`shared_amg.velocity_paths`). A is the operator stored by
component (`femtools.couplings`), as `assemble(a, couplings="form", bcs=bcs)` gives it on
either backend. From stokes-beadpack's solve-stokes (2026-11-16), where it was written out
for legacy dolfin.
"""
import numpy as np


def _backend(W):
    from femtools.backend import of
    return of(W)


def _pressure_index(W):
    """The pressure's index among W's sub-spaces: its last."""
    n = W.num_sub_spaces() if callable(W.num_sub_spaces) else W.num_sub_spaces
    if n < 2:
        raise ValueError("a Stokes space has a velocity and a pressure; this one has {} "
                         "sub-space(s)".format(n))
    return n - 1


def pressure_dofs(W):
    """The pressure's owned dofs, as global indices of A's numbering."""
    return _backend(W).owned_dofs(W, _pressure_index(W))


def pressure_mass_diagonal(W, mu=1.0):
    """diag(M)/mu on the pressure's owned dofs (in `pressure_dofs` order): the lumped P1
    mass times 2/(d + 2), which is diag(M) exactly on simplices. Collective."""
    L = _backend(W)
    i = _pressure_index(W)
    mesh = L.space_mesh(W)
    if (L.element_degree(W.sub(i)) != 1
            or str(mesh.ufl_cell()) not in ("triangle", "tetrahedron")):
        raise NotImplementedError("pressure_mass_diagonal: P1 on triangles or tetrahedra")
    d = L.geometric_dimension(mesh)
    q = L.TestFunctions(W)[-1]
    lumped = L.assemble((1.0 / mu) * q * L.dx)
    start = lumped.local_range()[0]
    return 2.0 / (d + 2) * lumped.get_local()[pressure_dofs(W) - start]


def _leaf_dofs(L, W, path):
    """The owned dofs of W's scalar component at `path`, global indices of A's numbering."""
    from femtools.backend import is_dolfinx
    from petsc4py import PETSc
    if is_dolfinx(W):
        if len(path) != 1:
            raise NotImplementedError("a component of a component on DOLFINx")
        return L.owned_dofs(W, path[0])
    sub = W
    for i in path:
        sub = sub.sub(i)
    return np.asarray(sub.dofmap().dofs(), dtype=PETSc.IntType)


class BlockPreconditioner:
    """What `block_preconditioner` set up and must free: `destroy()` once the solves are
    done (before the next phases' buffers; the KSP is the caller's)."""

    def __init__(self, kind, shell=None, D=None, sets=()):
        self.kind, self.shell, self.D, self.sets = kind, shell, D, list(sets)

    def destroy(self):
        for o in (self.shell, self.D, *self.sets):
            if o is not None:
                o.destroy()
        self.shell, self.D, self.sets = None, None, []


def block_preconditioner(ksp, A, W, diag_M, amg_options, kind="shared", velocity=None):
    """Set up `ksp`'s preconditioner on A (the operator on W) as diag(AMG on each velocity
    component, Jacobi on `diag_M`): `kind` "shared" (one AMG, `shared_amg.SharedVelocityPC`)
    or "fieldsplit" (one each). `velocity`, the components' paths (default every scalar
    component but the last); `amg_options` without a prefix (PETSC_OPTIONS winning). Sets
    the KSP up. Returns the BlockPreconditioner to destroy. Collective."""
    from petsc4py import PETSc

    from femtools.ksp import defaults, diagonal_matrix, scoped
    from femtools.shared_amg import (SharedVelocityPC, check_blocks, component_index_sets,
                                     velocity_paths)
    L = _backend(W)
    comm = L.mesh_comm(L.space_mesh(W))
    velocity = velocity_paths(W) if velocity is None else tuple(velocity)
    p_dofs = pressure_dofs(W)
    pc = ksp.getPC()
    if kind == "shared":
        sets = component_index_sets(W, comm, velocity)
        check_blocks(A, sets, comm)
        shell = SharedVelocityPC(A, sets, p_dofs, diag_M, comm, amg_options)
        pc.setType("python")
        pc.setPythonContext(shell)
        ksp.setFromOptions()
        ksp.setUp()
        return BlockPreconditioner(kind, shell=shell)
    if kind != "fieldsplit":
        raise ValueError("block_preconditioner: kind 'shared' or 'fieldsplit', not "
                         "{!r}".format(kind))
    pc.setType("fieldsplit")
    splits = [("u{}".format(c), PETSc.IS().createGeneral(_leaf_dofs(L, W, path), comm=comm))
              for c, path in enumerate(velocity)]
    splits.append(("p", PETSc.IS().createGeneral(p_dofs, comm=comm)))
    pc.setFieldSplitIS(*splits)
    # the pressure split's operator is replaced below; until then it is the operator's
    # empty pressure block, on which nothing should be set up
    options = {"pc_fieldsplit_type": "additive", "fieldsplit_p_ksp_type": "preonly",
               "fieldsplit_p_pc_type": "none"}
    for c in range(len(velocity)):
        prefix = "fieldsplit_u{}_".format(c)
        options.update({prefix + "ksp_type": "preonly", prefix + "pc_type": "hypre"})
        options.update(defaults(prefix, amg_options))
    # in the database for the setup, which reads them, and taken out after it: left
    # there, a later build in the process kept this one's AMG settings whatever it asked
    with scoped("", options):
        ksp.setFromOptions()
        ksp.setUp()
    D = diagonal_matrix(diag_M, comm)
    pressure = pc.getFieldSplitSubKSP()[len(velocity)]
    pressure.setOperators(D, D)
    pressure.getPC().setType("jacobi")
    pressure.setUp()
    return BlockPreconditioner(kind, D=D, sets=[s for _, s in splits])


def solve_checked(ksp, b, x):
    """ksp.solve(b, x) (PETSc vectors), raising unless it converged; returns the
    iterations. The ghosts of a Function whose vector x is are the caller's to bring up to
    date (the backend's `sync`)."""
    ksp.solve(b, x)
    iterations = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()
    if reason < 0:
        raise RuntimeError("the Krylov solver did not converge: PETSc reason {}, after {} "
                           "iterations".format(reason, iterations))
    return iterations
