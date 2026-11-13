"""PETSc objects that Python has dropped, destroyed.

petsc4py does not destroy an object on a parallel communicator when Python frees it:
destruction is collective, and the ranks need not free their objects in the same order.
It keeps the object instead until `PETSc.garbage_cleanup` is called on that
communicator, on every rank of it. Until then a dropped matrix keeps all its memory.
(PETSc objects made in C++, such as dolfin's, are destroyed directly.)

`register(comm)` records a communicator objects are made on; `cleanup()` destroys the
dropped objects on every recorded one and on COMM_WORLD, `cleanup(comm)` on one.
Both are collective over the communicators concerned, which the ranks must record in
the same order (as collective constructions, such as spaces, do).
"""

_COMMS = []


def register(comm):
    """Record `comm` (an mpi4py communicator) for `cleanup`, once."""
    if not any(comm is c or comm == c for c in _COMMS):
        _COMMS.append(comm)


def cleanup(comm=None):
    """Destroy the PETSc objects Python has dropped, on `comm`, or on every recorded
    communicator and COMM_WORLD. Collective."""
    from petsc4py import PETSc

    if not hasattr(PETSc, "garbage_cleanup"):     # petsc4py before 3.18 destroys at once
        return
    if comm is not None:
        PETSc.garbage_cleanup(comm)
        return
    for c in _COMMS:
        PETSc.garbage_cleanup(c)
    PETSc.garbage_cleanup()
