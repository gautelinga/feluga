__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
A Function's values, and gradients, at many points, evaluated many times.

    probes = Probes(points.flatten(), V)      # every rank given the same points
    probes(u)                                 # one snapshot
    probes.array()                            # (points, components, snapshots) on root
    probes.array(N=0)                         # one snapshot: (points, components)

    probes.dump("out/u")                      # out/u_rank<r>.probe, a file a rank
    read_dump("out/u")                        # {id: (coordinates, values)}

A point is kept by the lowest rank that finds it in its cells, so a point on a partition
boundary is evaluated once; a point outside the mesh is found by no rank and reads 0 in
`array`, which gathers over the mesh's communicator. `GradProbes` takes gradients too
(`grad`, `array_grad`). The C++ classes (femtools/cpp/probe11.cpp) are built with
cppimport at their first use, collectively; read_dump needs none.
"""
import glob
import os
import re

import numpy as np


# The compiled classes, built on first use of Probe, Probes, GradProbe or GradProbes
# (on rank 0, loaded on every rank: that first use is collective under MPI); importing
# the module, or read_dump, builds nothing.
_COMPILED = {}
_NAMES = ("probe11", "Probe", "Probes", "GradProbe", "GradProbes")


def __getattr__(name):
    if name in _NAMES:
        if not _COMPILED:
            _build()
        return _COMPILED[name]
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def _on_root(comm, ids, local, total, root):
    """Every rank's rows `local` (one a local probe, `ids` their places) as one array on
    root of comm, `total` rows, zeros where no rank holds the probe; None elsewhere."""
    local = np.ascontiguousarray(local, dtype=float)
    tail = local.shape[1:]
    width = int(np.prod(tail)) if tail else 1
    counts = comm.gather(len(ids), root=root)
    all_ids = comm.gather(np.asarray(ids, dtype=np.int64), root=root)
    is_root = comm.Get_rank() == root
    buf = np.empty(sum(counts) * width) if is_root else None
    comm.Gatherv(local.reshape(-1), [buf, [c * width for c in counts]] if is_root else None,
                 root=root)
    if not is_root:
        return None
    z = np.zeros((total,) + tail)
    z[np.concatenate(all_ids)] = buf.reshape((sum(counts),) + tail)
    return z


def read_dump(filename):
    """What `dump(filename)` or `dump(i, filename)` wrote, on any number of ranks: each
    probe's id to `(coordinates, values)`, its three coordinates and its values a row a
    snapshot (one column for a dumped component, a column a component otherwise). Reads
    the files a rank (`filename_rank<r>.probe`) and the files a probe of fenicstools'
    layout (`filename_<id>.probe`); the two hold the same text a probe. With the list
    of a dump's files (`filename.probes`), those files alone."""
    out = {}
    listed = filename + ".probes"
    if os.path.exists(listed):
        with open(listed) as handle:
            here = os.path.dirname(filename)
            names = [os.path.join(here, n.strip()) for n in handle if n.strip()]
    else:
        names = sorted(glob.glob(glob.escape(filename) + "_*.probe"))
        pattern = re.compile(re.escape(os.path.basename(filename)) + r"_(rank)?\d+\.probe$")
        names = [n for n in names if pattern.match(os.path.basename(n))]
    for name in names:
        with open(name) as handle:
            out.update(_parse_dump(handle.read().splitlines()))
    return out


def _parse_dump(lines):
    """`read_dump`'s probes of one file's lines."""
    out, k = {}, 0
    while k < len(lines):
        if not lines[k].startswith("Probe id = "):
            k += 1
            continue
        pid = int(lines[k].split("=")[1])
        n = int(lines[k + 1].split("=")[1])
        x = np.array(lines[k + 3].split(), dtype=float)
        rows = lines[k + 5:k + 5 + n]
        values = np.array([r.split() for r in rows], dtype=float).reshape(n, -1)
        out[pid] = (x, values)
        k += 5 + n
    return out


class _Collection(object):
    """What Probes and GradProbes share in Python: call to evaluate, iterate over the
    local probes as (id, probe), and gather to root of the mesh's communicator."""

    def __init__(self, x, V):
        super().__init__(x, V)
        self._comm = V.mesh().mpi_comm()

    def __call__(self, *args):
        return self.eval(*args)

    def dump(self, *args):
        """`dump(filename)` or `dump(i, filename)`: each rank's probes to
        filename_rank<r>.probe, and on rank 0 the list of this dump's files to
        filename.probes, which read_dump reads (so files left by an earlier dump on
        more ranks are not). Collective."""
        filename = args[-1]
        super().dump(*args)
        mine = ("{}_rank{}.probe".format(os.path.basename(filename), self._comm.Get_rank())
                if self.local_size() else None)
        names = self._comm.gather(mine, root=0)
        if self._comm.Get_rank() == 0:
            with open(filename + ".probes", "w") as handle:
                handle.write("".join(n + "\n" for n in names if n))

    def __len__(self):
        return self.local_size()

    def __iter__(self):
        for i in range(self.local_size()):
            yield self[i]

    def __getitem__(self, i):
        return self.get_probe_id(i), self.get_probe(i)

    def _array(self, values, N, filename, component, root, what):
        """`values` (local probes, components, snapshots) on root, one snapshot N or
        all, one component or all, saved to `filename` + `_snapshot_N` or `_all` too.
        Every rank checks N and the component alike, before the collective gather."""
        n_comp, n_snap = values.shape[1], values.shape[2]
        if N is not None and not 0 <= N < n_snap:
            raise IndexError("{}: no snapshot {}, there are {}".format(what, N, n_snap))
        if component is not None and not 0 <= component < n_comp:
            raise IndexError("{}: no component {}, there are {}".format(
                what, component, n_comp))
        if component is not None:
            values = values[:, [component], :]
        if N is not None:
            values = values[:, :, N]
        z = _on_root(self._comm, self.get_probe_ids(), values,
                     self.get_total_number_probes(), root)
        if z is None:
            return None
        if filename:
            np.save(filename + ("_snapshot_" + str(N) if N is not None else "_all"), z)
        return np.squeeze(z)

    def array(self, N=None, filename=None, component=None, root=0):
        """Every probe's values on root: all snapshots or snapshot N, every component or
        one. Collective; None on the other ranks."""
        return self._array(self.values(), N, filename, component, root, "array")


def _build():
    from femtools.jit import cppimport_module
    probe11 = cppimport_module(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp", "probe11.cpp"),
        "femtools.cpp.probe11")

    class Probe(probe11.Probe):

        def __call__(self, *args):
            return self.eval(*args)

        def __len__(self):
            return self.value_size()

        def __getitem__(self, i):
            return self.get_probe_at_snapshot(i)

    class Probes(_Collection, probe11.Probes):
        pass

    class GradProbe(probe11.GradProbe):

        def __call__(self, *args):
            return self.eval(*args)

        def __len__(self):
            return self.value_size()

        def __getitem__(self, i):
            return self.get_probe_at_snapshot(i)

    class GradProbes(_Collection, probe11.GradProbes):

        def grad(self, *args):
            return self.eval_grad(*args)

        def array_grad(self, N=None, filename=None, component=None, root=0):
            """`array` for the gradients: component j*gdim + k is d(value j)/d(x_k)."""
            return self._array(self.grad_values(), N, filename, component, root,
                               "array_grad")

    for cls in (Probe, Probes, GradProbe, GradProbes):
        cls.__module__ = __name__
    _COMPILED.update(probe11=probe11, Probe=Probe, Probes=Probes, GradProbe=GradProbe,
                     GradProbes=GradProbes)
