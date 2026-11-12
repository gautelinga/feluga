"""Building a compiled extension once under MPI, without the ways that hang a job.

    path = on_rank0(comm, build, "what")    build() on rank 0, its result on every rank
    lib = on_every_rank(comm, load, "what") load() on every rank, all raising if one did
    path = install_dir(tmp, final)          a finished build moved into the cache

A rank-local `raise` before a collective is the shape that hangs: rank 0 fails to build
(no compiler on PATH, a cache folder it may not write) and leaves, while the others wait
in the broadcast for ever. Here whatever rank 0 raises is caught, broadcast as text and
raised on every rank; a load that fails on some rank (a file system that shows the new
library late) is agreed on, so every rank raises with it.

The cache is shared by concurrent jobs: a build goes into a fresh temporary folder next
to its final place and is renamed there when complete (atomic on one file system), so no
job loads a half-built library; a job that loses the race to the same folder uses the
winner's. Nothing is ever deleted.

Used by femtools.dolfinx.refine (CMake) and femtools.dolfinx.expression (the compiler).
"""
import os
import time


def on_rank0(comm, build, what):
    """`build()` on rank 0 of `comm` (every caller when `comm` is None or of one rank),
    its (picklable) result returned on every rank. If it raises, every rank raises the
    same kind of error (a builtin one; else RuntimeError), naming `what`. Collective."""
    result, error = None, None
    if comm is None or comm.rank == 0:
        try:
            result = build()
        except Exception as e:         # every rank must hear of it, not rank 0 alone
            kind = type(e) if type(e).__module__ == "builtins" else RuntimeError
            error = (kind, "{}: {}: {}".format(what, type(e).__name__, e))
    if comm is not None and comm.size > 1:
        result, error = comm.bcast((result, error), root=0)
    if error is not None:
        raise error[0](error[1])
    return result


def on_every_rank(comm, load, what, wait_for=None, timeout=30.0):
    """`load()` on every rank of `comm`; if any rank fails, every rank raises (the
    failing ranks with their error). `wait_for`: a path to wait up to `timeout` seconds
    for first, which a shared file system may show a moment after rank 0 wrote it.
    Collective."""
    from femtools.hdf5_rows import agreed
    if wait_for is not None:
        deadline = time.monotonic() + timeout
        while not os.path.exists(wait_for) and time.monotonic() < deadline:
            time.sleep(0.2)
    result, error = None, None
    try:
        result = load()
    except Exception as e:
        error = "{}: {}".format(type(e).__name__, e)
    if agreed(comm, error is not None):
        raise RuntimeError("{}: {}".format(
            what, error or "it failed on another rank (see that rank's error)"))
    return result


def install_dir(tmp, final):
    """Rename the finished build folder `tmp` to `final`, atomically; returns where the
    build now is: `final`, or -- when another job got there first, or a folder left by
    an older layout is in the way -- `final` if it holds a finished build (its `done`
    file) and else `tmp`, used where it is."""
    try:
        os.rename(tmp, final)
        return final
    except OSError:
        if os.path.exists(os.path.join(final, "done")):
            return final
        return tmp
