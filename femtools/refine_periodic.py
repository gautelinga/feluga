"""Refinement of a mesh with periodic faces, checked before it is trusted.

    refine_congruent(mesh, markers, box, axes)   periodic Plaza, then the congruence guard
    congruent(mesh, box, axes)                    whether identified faces carry the same nodes
    check_congruent(mesh, box, axes)              the same, or exit with the refusal
    (each takes `lower=`, the box's lower corner when it is not the origin)
    compute_box(mesh)                             the box's upper corner, over the mesh's ranks
    dorfler_threshold(indicator, theta)           the bulk-chasing threshold of a DG0 indicator

`constrained_domain` pairs periodic dofs by coordinate. If a face is refined and the face
it is identified with is not, the slave dofs with no master are simply left
unconstrained: dolfin builds the space, the solve runs, and periodicity is quietly gone
along that face. `refine_congruent` refines through `femtools.plaza_periodic`, which
bisects identified faces alike, and checks the faces are congruent after it. There is no
fallback: marking every cell on the identified faces also keeps them congruent, but
refines that layer uniformly on every pass, and the cost compounds.

Runs under `mpirun`: `congruent` gathers the face nodes across ranks and deduplicates them
by position before comparing (a rank-boundary vertex is ghosted onto more than one rank).
"""

import dolfin as df

from femtools.periodic_core import TOL, dorfler_threshold_values


def congruent(mesh, box, axes, exempt=None, lower=None):
    """Whether the identified faces still carry the same nodes, edges included
    (`femtools.periodic.faces_congruent`). `exempt` is a bool per vertex then per edge
    of nodes allowed no partner; `lower` the box's lower corner (the origin when None,
    which a mesh elsewhere is refused)."""
    from femtools.periodic import faces_congruent
    ok, _, _, _ = faces_congruent(mesh, box, axes, TOL, edges=True, exempt=exempt,
                                  lower=lower)
    return ok


def check_congruent(mesh, box, axes, exempt=None, lower=None):
    """`congruent`, or exit with the refinement's refusal: the guard on a refined mesh
    (`refine_congruent`, or its caller after `deferred=True`)."""
    if not congruent(mesh, box, axes, exempt, lower):
        raise SystemExit(
            "the identified faces came out of the refinement not matching. Do "
            "not solve on this mesh: a periodic dof with no partner is left "
            "unconstrained and dolfin will not say so.")


def compute_box(mesh):
    """The periodic box's upper corner off the mesh itself, over the mesh's ranks
    (`periodic.extent`)."""
    from femtools.periodic import extent
    return extent(mesh)[1]


def dorfler_threshold(indicator_, theta, comm=None, iterations=60):
    """The bulk-chasing threshold of a DG0 indicator Function
    (`periodic_core.dorfler_threshold_values` of its local values; `comm` default the
    world)."""
    return dorfler_threshold_values(indicator_.vector().get_local(), theta,
                                    comm or df.MPI.comm_world, iterations)


def refine_congruent(mesh, cell_markers, box, axes, redistribute=False, max_cells=None,
                     exempt=None, max_rank_cells=None, frozen=None, report=None,
                     deferred=False, lower=None):
    """The periodic-safe refinement, with its output checked before it is trusted.

    Returns `(new_mesh, npair, parent_cell)`; `parent_cell` is None after a
    redistribution (`plaza_periodic.refine_periodic`) and for the no-periodic-axes
    `df.refine` path. The congruence guard lives here so no caller can skip it, with one
    exception asked for by name: `deferred=True` leaves it to the caller, which must run
    `check_congruent` on the mesh it goes on with before solving on it or refining it
    again (on a rebalanced mesh, say: the positions it compares are the same on any
    partition).

    `max_cells`: raise `plaza_periodic.TooManyCells` rather than refine past it,
    counted exactly before anything is built; `max_rank_cells` the same for the most
    loaded rank. Periodic axes only: `df.refine` cannot count first.

    `exempt` is a callable `(new_mesh, parent_cell) ->` a bool per vertex then per edge
    of the new mesh: nodes allowed no periodic partner (a slab's inlet and outlet, say).
    The guard forgives those and nothing else.

    `frozen` and `report`: edges never to bisect, and what freezing them left unmarked
    (`plaza_periodic.refine_periodic`). Periodic axes only.

    `lower`: the box's lower corner, when it is not the origin.
    """
    from femtools import periodic_core
    from femtools.plaza_periodic import refine_periodic, require_cells_on_every_rank

    def refine_plain(mesh, cell_markers):
        require_cells_on_every_rank(mesh, "refine_congruent")
        return df.refine(mesh, cell_markers)

    return periodic_core.refine_congruent(
        mesh, cell_markers, box, axes, comm=mesh.mpi_comm(), refine_periodic=refine_periodic,
        check_congruent=check_congruent, refine_plain=refine_plain, plain_name="df.refine",
        redistribute=redistribute, max_cells=max_cells, exempt=exempt,
        max_rank_cells=max_rank_cells, frozen=frozen, report=report, deferred=deferred,
        lower=lower)
