"""Wall time and memory at the end of each phase of a run, printed on rank zero.

    phase = Phases(comm)
    ...; phase("assembly")
    ...; phase("solve")
    mpi_print("done")                      # rank zero of the world only

Either backend: an mpi4py communicator, the world when None (one rank without mpi4py);
dolfin is never imported.
"""
import time

from femtools.hdf5_rows import allreduce, allsum, world


def mpi_print(*args, comm=None):
    """Prints on rank zero of `comm` (the world when None) only."""
    comm = world() if comm is None else comm
    if comm is None or comm.Get_rank() == 0:
        print(*args, flush=True)


class Phases:
    """Wall time and memory at the end of each phase, on rank zero.

    One line per phase, `phase <name>: <s> s (total <s> s), peak RSS <GB> max a rank,
    <GB> summed; now <GB> max a rank, <GB> summed`. Each phase ends by handing freed
    heap back to the operating system (`femtools.memory.release`): with petsc4py
    imported glibc keeps it otherwise, so the next phase could not reuse it and the
    resident size would count it. The peak is `ru_maxrss`, so the line after assembly
    carries the assembly peak; *now* is what the process holds after the release,
    which is what a phase leaves behind. Wall time rather than cProfile: a C extension
    clears the profiling hook partway through anything that imports dolfin.
    """

    def __init__(self, comm=None, clock=time.perf_counter):
        self.comm, self.clock = world() if comm is None else comm, clock
        self.start = self.last = self.clock()

    def __call__(self, name):
        """The line for phase `name`. Collective (reductions over the ranks)."""
        import resource
        from femtools.memory import release, resident_gb
        release()
        now = self.clock()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20      # KB -> GB
        cur = resident_gb()
        peak, total = allreduce(self.comm, rss, "max"), float(allsum(self.comm, rss))
        cpeak, ctotal = allreduce(self.comm, cur, "max"), float(allsum(self.comm, cur))
        mpi_print('phase {}: {:.1f} s (total {:.1f} s), peak RSS {:.2f} GB max a rank, '
                  '{:.2f} GB summed; now {:.2f} GB max a rank, {:.2f} GB summed'
                  .format(name, now - self.last, now - self.start, peak, total,
                          cpeak, ctotal), comm=self.comm)
        self.last = now
