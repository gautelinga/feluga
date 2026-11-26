"""feluga.probes on DOLFINx: a Function's values, and gradients, at many points,
evaluated many times, with the dolfin classes' semantics and Python API.

    probes = Probes(points.flatten(), V)      # every rank given the same points
    probes(u)                                 # one snapshot
    probes.array()                            # (points, components, snapshots) on root
    probes.array(N=0)                         # one snapshot: (points, components)

    g = GradProbes(points.flatten(), V)
    g.grad(u)                                 # a snapshot of the values and the gradient
    g.array_grad(N=0)                         # component j*gdim + k: d(value j)/d(x_k)

A point is evaluated in one cell: of the owned cells it lies in, on every rank, the one
with the smallest original (input) cell index, and by the rank owning that cell. A point
on a face between cells, or on a partition boundary, is therefore evaluated in the same
cell whatever the number of ranks, though not necessarily in the cell dolfin's search
picks: a discontinuous field may read the other side's value there. A point outside the
mesh is found by no rank and reads 0 in `array`, which gathers over the mesh's
communicator. `dump` writes the files feluga.probes.read_dump reads.

The gradient of a Lagrange field of degree k is, cell by cell, a polynomial of degree
k - 1, so it is interpolated (exactly, on affine cells) into the discontinuous space of
that degree and evaluated at the probes' cells.
"""
import os
import weakref

import numpy as np

from feluga.probes import _Collection


def _value_size(V):
    shape = V.ufl_element().reference_value_shape
    return int(np.prod(shape)) if len(shape) else 1


def _mixed(V):
    """Whether V's element is mixed (not a blocked vector or tensor element)."""
    return bool(getattr(V.ufl_element(), "is_mixed", False))


def _locate(mesh, x):
    """The points x (N, gdim) this rank keeps, as (indices, points padded to 3, cells):
    each point goes to the owned cell holding it with the smallest original cell index
    over all ranks, kept by the rank owning that cell. Collective."""
    from dolfinx import geometry
    from mpi4py import MPI
    comm = mesh.comm
    tdim = mesh.topology.dim
    n_own = mesh.topology.index_map(tdim).size_local
    tree = geometry.bb_tree(mesh, tdim, entities=np.arange(n_own, dtype=np.int32))
    x3 = np.zeros((len(x), 3))
    x3[:, :x.shape[1]] = x
    none = np.iinfo(np.int64).max
    best = np.full(len(x), none, dtype=np.int64)        # smallest original index found
    cells = np.full(len(x), -1, dtype=np.int32)
    if len(x):
        found = geometry.compute_colliding_cells(
            mesh, geometry.compute_collisions_points(tree, x3), x3)
        offsets, links = np.asarray(found.offsets), np.asarray(found.array)
        counts = np.diff(offsets)
        if len(links):
            point = np.repeat(np.arange(len(x)), counts)
            original = np.asarray(mesh.topology.original_cell_index,
                                  dtype=np.int64)[links]
            order = np.lexsort((original, point))          # by point, then index
            first = order[np.r_[0, np.cumsum(counts[counts > 0])[:-1]]]
            hit = point[first]
            best[hit] = original[first]
            cells[hit] = links[first]
    lowest = np.empty_like(best)
    comm.Allreduce(best, lowest, op=MPI.MIN)
    keep = np.nonzero((best == lowest) & (best != none))[0]
    return keep, x3[keep], cells[keep]


def _sync(u):
    """Bring u's DOLFINx array up to date with its backend vector, if it has one (a
    feluga.backend.dolfinx Function or a view of one's component)."""
    if callable(getattr(u, "sync", None)):
        u.sync()
    elif hasattr(u, "_parent"):
        u._parent.sync()
    return u


class Probes(_Collection):
    """Probes of a space V at the points x (flat, gdim per point, the same on every
    rank): `probes(u)` adds a snapshot of u, a Function on V."""

    def __init__(self, x, V):
        self._V = V
        self._comm = V.mesh.comm
        gdim = V.mesh.geometry.dim
        x = np.asarray(x, dtype=float).reshape(-1, gdim)
        self._total = len(x)
        self._ids, self._x, self._cells = _locate(V.mesh, x)
        self._vs = _value_size(V)
        self._snaps = []

    def eval(self, u):
        """One more snapshot of u. Collective (the ghost values brought up to date)."""
        _sync(u)
        u.x.scatter_forward()
        self._snaps.append(self._at(u))

    def _at(self, u):
        if not len(self._ids):
            return np.zeros((0, _value_size(u.function_space)))
        if _mixed(u.function_space):
            # DOLFINx's Function.eval refuses a mixed element (0.11: "Element does not
            # have a reference_value_shape"): each sub-function's values, side by side
            return np.hstack([self._at(u.sub(i))
                              for i in range(u.function_space.num_sub_spaces)])
        return np.asarray(u.eval(self._x, self._cells)).reshape(len(self._ids), -1)

    def values(self):
        """Every local probe's every component at every snapshot, (probes, components,
        snapshots)."""
        if not self._snaps:
            return np.zeros((len(self._ids), self._vs, 0))
        return np.stack(self._snaps, axis=2)

    def clear(self):
        """Forget the snapshots."""
        self._snaps = []

    def number_of_evaluations(self):
        return len(self._snaps)

    def local_size(self):
        return len(self._ids)

    def value_size(self):
        return self._vs

    def get_total_number_probes(self):
        return self._total

    def get_probe_ids(self):
        return [int(i) for i in self._ids]

    def get_probe_id(self, i):
        return int(self._ids[i])

    def coordinates(self):
        """The local probes' points, (probes, 3)."""
        return self._x.copy()

    def __getitem__(self, i):
        """(id, values) of local probe i: (snapshots, components)."""
        values = (np.stack([snap[i] for snap in self._snaps]) if self._snaps
                  else np.zeros((0, self._vs)))
        return int(self._ids[i]), values

    def dump(self, *args):
        """`dump(filename)` or `dump(i, filename)`: each rank's probes to
        filename_rank<r>.probe (as feluga.probes writes them), and on rank 0 the list
        of this dump's files to filename.probes. Collective."""
        filename = args[-1]
        component = args[0] if len(args) == 2 else None
        if component is not None and not 0 <= component < self._vs:
            raise IndexError("dump: no component {}, there are {}".format(
                component, self._vs))
        rank = self._comm.Get_rank()
        mine = None
        if len(self._ids):
            mine = "{}_rank{}.probe".format(os.path.basename(filename), rank)
            values = self.values()
            with open("{}_rank{}.probe".format(filename, rank), "w") as fp:
                for k, pid in enumerate(self._ids):
                    fp.write("Probe id = {}\nNumber of evaluations = {}\nCoordinates:\n"
                             "{:e} {:e} {:e}\n".format(pid, values.shape[2], *self._x[k]))
                    if component is None:
                        fp.write("Values for all components:\n")
                        for s in range(values.shape[2]):
                            fp.write("".join("{:e} ".format(v) for v in values[k, :, s])
                                     + "\n")
                    else:
                        fp.write("Values for component {}\n".format(component))
                        for s in range(values.shape[2]):
                            fp.write("{:e}\n".format(values[k, component, s]))
                        fp.write("\n")
        names = self._comm.gather(mine, root=0)
        if rank == 0:
            with open(filename + ".probes", "w") as handle:
                handle.write("".join(n + "\n" for n in names if n))


class GradProbes(Probes):
    """Probes that also take the gradient: `grad(u)` adds a snapshot of u's values and
    of its gradient (`array_grad`)."""

    def __init__(self, x, V):
        super().__init__(x, V)
        self._grads = []
        self._dg = {}
        self._held = None          # (weakref to the last u, its gradient Expression, g)
        self._probe_cells = np.unique(self._cells).astype(np.int32)

    def _grad_space(self, V):
        """The discontinuous Lagrange space of one degree less than V's, of the
        gradient's shape (V's value shape, then gdim)."""
        from dolfinx import fem
        key = id(V)
        if key not in self._dg:
            e = V.ufl_element()
            gdim = V.mesh.geometry.dim
            shape = tuple(e.reference_value_shape) + (gdim,)
            degree = max(e.embedded_superdegree - 1, 0)
            self._dg[key] = (V, fem.functionspace(V.mesh, ("DG", degree, shape)))
        return self._dg[key][1]

    def _gradient(self, u):
        """grad(u)'s Expression and the Function it is interpolated into, made once for
        the Function u and reused while u is the one probed."""
        held = self._held
        if held is None or held[0]() is not u:
            import ufl
            from dolfinx import fem
            W = self._grad_space(u.function_space)
            held = (weakref.ref(u),
                    fem.Expression(ufl.grad(u), W.element.interpolation_points), fem.Function(W))
            self._held = held
        return held[1], held[2]

    def eval_grad(self, u):
        """A snapshot of u's values and of its gradient. Collective (the ghost values
        brought up to date); the gradient is interpolated on the probes' cells only."""
        _sync(u)
        u.x.scatter_forward()
        self._snaps.append(self._at(u))
        expr, g = self._gradient(u)
        if len(self._probe_cells):
            g.interpolate(expr, cells0=self._probe_cells)
        self._grads.append(self._at(g))

    def grad(self, *args):
        return self.eval_grad(*args)

    def grad_values(self):
        """Every local probe's gradient components at every gradient snapshot,
        (probes, components * gdim, snapshots)."""
        if not self._grads:
            return np.zeros((len(self._ids), self._vs * self._V.mesh.geometry.dim, 0))
        return np.stack(self._grads, axis=2)

    def clear_grad(self):
        self._grads = []

    def number_of_grad_evaluations(self):
        return len(self._grads)

    def array_grad(self, N=None, filename=None, component=None, root=0):
        """`array` for the gradients: component j*gdim + k is d(value j)/d(x_k)."""
        return self._array(self.grad_values(), N, filename, component, root, "array_grad")
