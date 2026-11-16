"""The names of femtools.backend on DOLFINx: dolfin's object API over unique dofs. Code
written for legacy dolfin (spaces with a constrained_domain, Functions with vector(),
DirichletBC, assemble, facet markers, Expressions in C++, PETScKrylovSolver, ...) runs
on DOLFINx with these names, the periodicity included, and no dolfinx_mpc.

    from femtools.backend.dolfinx import *
    V = FunctionSpace(mesh, "CG", 2, constrained_domain=PeriodicBox({0: (0, 1)}))
    u = Function(V); u.vector()[:] = 1.0
    bc = DirichletBC(V, 0.0, markers, 2)          # markers: a BoundaryMarkers
    A = assemble(inner(grad(TrialFunction(V)), grad(TestFunction(V))) * dx)

Every vector the algorithm sees, a Function's `vector()` included, is a PETSc vector
over the space's *unique* dofs (femtools.dolfinx.unique): periodic slaves are folded
onto their masters, as dolfin's `constrained_domain` does, so norms, means, pointwise
products and reciprocals need no masks, and matrices are assembled folded
(N_unique x N_unique).

A Function is a DOLFINx Function whose values live in two layouts: the compact vector,
which is what the algorithm reads and writes, and DOLFINx's own array (owned, ghost
and slave dofs), which the form assembly and the interpolation read. The compact
vector is the truth: `sync()` copies it into DOLFINx's array (one forward scatter)
when it changed since the last copy (PETSc's state of the vector), and assemble calls
it on every coefficient of a form before assembling it. DOLFINx's own ways of writing
the array end in the compact vector too: `u.interpolate(...)`, and `u.x.array[...] =
...` followed by `u.x.scatter_forward()` (as DOLFINx asks after a direct write), on the
Function or on a view `u.sub(i)`. A write to `u.x.array` not followed by
`u.x.scatter_forward()` is not seen.

Where dolfin's method names clash with DOLFINx's properties (`V.mesh()` against
`V.mesh`, `u.function_space()` against `u.function_space`, `mesh.geometry().dim()`
against `mesh.geometry.dim`), code calls the neutral helpers: `space_mesh`,
`function_space`, `geometric_dimension`, `element_degree`, `owned_dofs`, `fixed_dofs`.

Supported: scalar and mixed Lagrange spaces (and DG), a mixed space's components
vector ones too (`VectorElement`: Taylor-Hood as [P2^3, P1] or [P2, P2, P2, P1]),
periodic boxes given as a box spec (`lower`, `upper`, `periodic`), Dirichlet conditions
on facets marked in a `BoundaryMarkers` or by C++ code (or pointwise), on a scalar space
or component or a vector component, with a number, a `Constant` (a vector one on a vector
component) or a Function on the (scalar) space as value; `owned_dofs` of a component by
index or by path ((0, 1): the velocity's y component).
"""
import time
import weakref
from collections import defaultdict
from enum import Enum

import numpy as np
from mpi4py import MPI as _MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import dolfinx.fem.petsc  # noqa: F401  (fem.petsc)
import ufl
from dolfinx import fem
from ufl import (  # noqa: F401
    as_vector, conditional, cos, div, dot, exp, FacetNormal, ge, grad, gt, inner, le, lhs,
    lt, nabla_grad, rhs, sin, split, sqrt, TestFunction, TestFunctions, TrialFunction,
    TrialFunctions)

from femtools import ksp  # noqa: F401
from femtools.couplings import form_couplings  # noqa: F401
from femtools.dolfinx import assembly as _assembly
from femtools.dolfinx.boundary_markers import BoundaryMarkers, compiled_facets  # noqa: F401
from femtools.dolfinx import mixed as _mixed
from femtools.dolfinx.periodic import flat_collapse_map, periodic_box
from femtools.dolfinx.unique import UniqueDofs
from femtools.hdf5_rows import agreed
from femtools.lru import LRU

name = "dolfinx"
dx = ufl.dx
# What dolfin calls a Coefficient: a Function or a Constant (a ufl.Constant in DOLFINx),
# for isinstance checks.
Coefficient = (ufl.Coefficient, ufl.Constant)
DOLFIN_EPS = 3.0e-16
DOLFIN_EPS_LARGE = 1.0e-14

# dolfin's global parameters, those of the form compiler that matter here.
# quadrature_degree is put on every integral that does not set its own, as dolfin does
# with the global one.
parameters = {
    "linear_algebra_backend": "PETSc",
    "form_compiler": {"quadrature_degree": None, "optimize": True, "cpp_optimize": True,
                      "representation": "uflacs", "cpp_optimize_flags": "-O3"},
}


# --- MPI -------------------------------------------------------------------------------

class MPI:
    """dolfin's MPI helpers on mpi4py communicators."""
    comm_world = _MPI.COMM_WORLD
    comm_self = _MPI.COMM_SELF

    @staticmethod
    def rank(comm):
        return comm.rank

    @staticmethod
    def size(comm):
        return comm.size

    @staticmethod
    def barrier(comm):
        comm.Barrier()

    @staticmethod
    def sum(comm, value):
        return comm.allreduce(value, op=_MPI.SUM)

    @staticmethod
    def min(comm, value):
        return comm.allreduce(value, op=_MPI.MIN)

    @staticmethod
    def max(comm, value):
        return comm.allreduce(value, op=_MPI.MAX)


# --- timers ----------------------------------------------------------------------------

class TimingClear(Enum):
    """dolfin's TimingClear: whether list_timings clears the timings it lists."""
    keep = 0
    clear = 1


class TimingType(Enum):
    """dolfin's TimingType: which clock list_timings reports."""
    wall = 0
    user = 1
    system = 2


_TIMINGS = defaultdict(lambda: [0, 0.0])  # task -> [repetitions, total wall time]


class Timer:
    """dolfin's Timer: started when made (with a task name, it is recorded for
    list_timings when stopped or destroyed); stop() returns the elapsed wall time in seconds,
    elapsed() the (wall, user, system) times so far."""

    def __init__(self, task=None):
        self.task = task
        self._total = 0.0
        self._t0 = None
        self.start()

    def start(self):
        self._total = 0.0
        self._t0 = time.perf_counter()

    def stop(self):
        if self._t0 is not None:
            self._total += time.perf_counter() - self._t0
            self._t0 = None
            if self.task:
                entry = _TIMINGS[self.task]
                entry[0] += 1
                entry[1] += self._total
        return self._total

    def __del__(self):
        # dolfin records a running Timer when it is destroyed, too.
        try:
            self.stop()
        except Exception:
            pass

    def elapsed(self):
        running = time.perf_counter() - self._t0 if self._t0 is not None else 0.0
        wall = self._total + running
        return wall, wall, 0.0


def list_timings(clear=TimingClear.keep, types=(TimingType.wall,)):
    """Print the wall times of the named Timers (average over the processes), as
    dolfin's list_timings does; clear them with TimingClear.clear. Collective."""
    comm = MPI.comm_world
    tasks = sorted(set().union(*comm.allgather(list(_TIMINGS))))
    rows = []
    for task in tasks:
        reps, total = _TIMINGS[task] if task in _TIMINGS else (0, 0.0)
        reps = comm.allreduce(reps, op=_MPI.MAX)
        total = comm.allreduce(total, op=_MPI.SUM) / comm.size
        rows.append((task, reps, total / max(reps, 1), total))
    if comm.rank == 0 and rows:
        width = max(len(r[0]) for r in rows)
        print("[MPI_AVG] Summary of timings {} | reps    wall avg    wall tot".format(
            " " * max(width - 28, 0)))
        print("-" * (width + 36))
        for task, reps, avg, total in rows:
            print("{:<{w}} | {:6d}  {:10.6f}  {:10.6f}".format(task, reps, avg, total, w=width))
    if clear == TimingClear.clear:
        _TIMINGS.clear()


# --- elements, meshes and spaces -------------------------------------------------------

_FAMILIES = {"CG": "Lagrange", "P": "Lagrange", "Lagrange": "Lagrange", "Q": "Lagrange",
             "DG": "DG", "Discontinuous Lagrange": "DG"}


def FiniteElement(family, cell, degree):
    """A scalar element by dolfin's family name ('CG', 'DG', ...)."""
    family = _FAMILIES.get(family, family)
    cellname = cell if isinstance(cell, str) else cell.cellname  # a property in UFL 2026
    cellname = cellname() if callable(cellname) else cellname
    if family == "DG":
        return basix.ufl.element("Lagrange", cellname, degree, discontinuous=True)
    return basix.ufl.element(family, cellname, degree)


def VectorElement(family, cell, degree, dim=None):
    """dolfin's VectorElement: `dim` (default the cell's dimension) components of the
    scalar element FiniteElement(family, cell, degree)."""
    base = FiniteElement(family, cell, degree)
    if dim is None:
        dim = base.cell.topological_dimension      # a property in UFL 2026
        dim = dim() if callable(dim) else dim
    n = int(dim)
    return basix.ufl.blocked_element(base, shape=(n,))


def MixedElement(*elements):
    """dolfin's MixedElement: the mixed element of elements given one by one,
    MixedElement(P2, P1), or as a list, MixedElement([P2, P1])."""
    if len(elements) == 1 and isinstance(elements[0], (list, tuple)):
        elements = elements[0]
    return basix.ufl.mixed_element(list(elements))


class BoxSpec:
    """A periodic box: `lower` and `upper` corners and the `periodic` axes (3 each),
    with `tol` the absolute distance from a face that counts as on it. The dolfinx
    counterpart of femtools.periodic.periodic_box_from_bounds, which dolfin problems
    use; any object with these attributes serves."""

    def __init__(self, lower, upper, periodic, tol=1e-10):
        self.lower = [float(v) for v in lower]
        self.upper = [float(v) for v in upper]
        self.periodic = [bool(p) for p in periodic]
        self.tol = float(tol)


def PeriodicBox(bounds, tol=1e-10, map_tol=None, comm=None):
    """The box periodic along each axis in `bounds`, which maps the axis to its
    (lower, upper) bounds: {0: (0, Lx)} is periodic in x alone; `tol` the absolute
    distance from a face that counts as on it. `map_tol` and `comm` are dolfin's (its
    compiled box maps points within map_tol, and compiles on comm); taken here so a call
    serves both backends, and unused: DOLFINx pairs dofs by position within `tol`."""
    lower, upper = ([bounds[i][k] if i in bounds else 0. for i in range(3)] for k in (0, 1))
    return BoxSpec(lower, upper, [i in bounds for i in range(3)], tol)


Mesh = dolfinx.mesh.Mesh
SpatialCoordinate = ufl.SpatialCoordinate


class Point:
    """dolfin's Point(x, y, z=0)."""

    def __init__(self, *x):
        self.x = [float(v) for v in x] + [0.0] * (3 - len(x))

    def __getitem__(self, i):
        return self.x[i]


_DIAGONALS = {"right": "right", "left": "left", "crossed": "crossed",
              "right/left": "right_left", "left/right": "left_right"}


def RectangleMesh(p0, p1, nx, ny, diagonal="right"):
    """dolfin's RectangleMesh(Point, Point, nx, ny, diagonal): the same vertices and
    cells, on MPI.comm_world."""
    kind = getattr(dolfinx.mesh.DiagonalType, _DIAGONALS[diagonal])
    return _first(dolfinx.mesh.create_rectangle(
        MPI.comm_world, [[p0[0], p0[1]], [p1[0], p1[1]]], [int(nx), int(ny)],
        dolfinx.mesh.CellType.triangle, diagonal=kind))


def repair_partition(mesh, box, log=None):
    """The mesh: DOLFINx pairs periodic dofs by their coordinates, on any partition."""
    return mesh


def check_periodic_pairing(box, spaces):
    """Nothing to check: DOLFINx pairs periodic dofs by their coordinates."""


def global_dim(V):
    """The space's unique dofs over all ranks (periodic slaves folded); a component
    W.sub(i)'s alone. Collective for a component."""
    parent = getattr(V, "_parent", None)
    if parent is None:
        return unique_dofs(V).n_global
    component = list(V.component())
    if len(component) != 1:
        raise NotImplementedError("global_dim: a component of a mixed space, not of one of "
                                  "its components")
    return parent.mesh.comm.allreduce(len(owned_dofs(parent, component[0])))


def global_num_cells(mesh):
    """The mesh's cells over all ranks."""
    return mesh.topology.index_map(mesh.topology.dim).size_global


def global_num_vertices(mesh):
    """The mesh's vertices over all ranks."""
    return mesh.topology.index_map(0).size_global


def marker_counts(markers):
    """The exterior facets of each marker over all ranks, as an array (`np.bincount`
    of their markers, as femtools.boundaries.written_marker_counts gives a file's): of
    `markers`, a BoundaryMarkers; each exterior facet counted by the rank owning it.
    Collective."""
    mesh = markers.mesh
    comm = mesh.comm
    n_own = mesh.topology.index_map(mesh.topology.dim - 1).size_local
    ext = np.asarray(markers.boundary)
    values = markers.values[ext[ext < n_own]].astype(np.int64)
    n = comm.allreduce(int(values.max(initial=-1)) + 1, op=_MPI.MAX)
    return comm.allreduce(np.bincount(values, minlength=n))


def read_mesh_file(fname, periodic=None, tol=None):
    """The mesh in the file fname: "mesh" in an HDF5 file (dolfin's layout), or the mesh
    of an XDMF file (`*.xdmf`, the meshers'), numbered as dolfin numbers it
    (femtools.dolfinx.mesh.read_mesh). `periodic` and `tol` are dolfin's concern (its
    partition must be repaired); DOLFINx pairs periodic dofs on any partition."""
    from femtools.dolfinx.mesh import read_mesh
    return _first(read_mesh(fname, MPI.comm_world))


def BoxMesh(p0, p1, nx, ny, nz):
    """dolfin's BoxMesh(Point, Point, nx, ny, nz): the same vertices and tetrahedra
    (DOLFINx's create_box numbers them as dolfin does), on MPI.comm_world."""
    return _first(dolfinx.mesh.create_box(
        MPI.comm_world, [[p0[0], p0[1], p0[2]], [p1[0], p1[1], p1[2]]],
        [int(nx), int(ny), int(nz)], dolfinx.mesh.CellType.tetrahedron))


def UnitSquareMesh(nx, ny, diagonal="right"):
    """dolfin's UnitSquareMesh(nx, ny, diagonal) (RectangleMesh of the unit square)."""
    return RectangleMesh(Point(0., 0.), Point(1., 1.), nx, ny, diagonal)


def mesh_coordinates(mesh):
    """The mesh's vertex coordinates, a writable (n, gdim) view (dolfin's
    mesh.coordinates()): change them before making spaces on the mesh."""
    return mesh.geometry.x[:, :mesh.geometry.dim]


def hmax(mesh):
    """The largest cell diameter over all ranks. Collective."""
    tdim = mesh.topology.dim
    n = mesh.topology.index_map(tdim).size_local
    h = mesh.h(tdim, np.arange(n, dtype=np.int32))  # a method of Mesh in DOLFINx 0.11
    return MPI.max(mesh.comm, float(h.max()) if n else 0.0)


def hmin(mesh):
    """The smallest cell diameter over all ranks. Collective."""
    tdim = mesh.topology.dim
    n = mesh.topology.index_map(tdim).size_local
    h = mesh.h(tdim, np.arange(n, dtype=np.int32))
    return MPI.min(mesh.comm, float(h.min()) if n else float("inf"))


def write_mesh_h5(path, mesh, name="mesh", mode="w"):
    """The mesh in dolfin's HDF5 layout, as the group `name`, the file started over
    ("w") or added to ("a") (femtools.dolfinx.mesh.write_mesh). Collective."""
    from femtools.dolfinx.mesh import write_mesh
    write_mesh(path, mesh, name, mode)


def extent(mesh):
    """`(lower, upper)`: the mesh's bounding box over all ranks, gdim components each."""
    from femtools.hdf5_rows import extent
    gdim = mesh.geometry.dim
    return extent(mesh.geometry.x[:, :gdim], mesh.comm, gdim)


def mark_walls(boundaries, mesh, marker=1, tol=1e-10):
    """Mark the walls of the obstacles inside the mesh's bounding box `marker`: every
    boundary facet except those on the box's faces, which are set to 0 (femtools.box's
    mark_walls)."""
    from femtools.periodic_core import on_planes
    lo, hi = extent(mesh)
    boundaries.mark(lambda x: np.ones(x.shape[1], dtype=bool), marker)
    boundaries.mark(lambda x: on_planes(x.T, lo, hi, range(len(lo)), tol), 0)


class Faces:
    """The boundary where coordinate `axis` equals one of `values` (femtools.box.Faces):
    Faces(1, 0, Ly).mark(markers, 2) marks the faces y = 0 and y = Ly."""

    def __init__(self, axis, *values, tol=1e-10):
        self.axis, self.values, self.tol = axis, values, tol

    def inside_points(self, x):
        on = np.zeros(x.shape[1], dtype=bool)
        for v in self.values:
            on |= np.abs(x[self.axis] - v) < self.tol
        return on

    def mark(self, markers, value):
        markers.mark(self, value)


class SubDomain:
    """dolfin's SubDomain for marking facets: a subclass defines inside(x, on_boundary)
    for one point x, and mark(markers, value) marks the boundary facets whose vertices
    and midpoint are inside (with on_boundary True), as dolfin does. Evaluated point by
    point in Python, which is slow on large meshes; Faces, or a vectorized predicate
    given to BoundaryMarkers.mark, is fast."""

    def inside(self, x, on_boundary):
        raise NotImplementedError

    def inside_points(self, x):
        return np.array([bool(self.inside(x[:, k], True)) for k in range(x.shape[1])],
                        dtype=bool)

    def mark(self, markers, value):
        markers.mark(self, value)


_DEFAULT_MESH = [None]


def set_default_mesh(mesh):
    """The mesh a Constant or an Expression made without one is on (dolfin's belong to
    no mesh; DOLFINx's need one): the driver sets the flow mesh; else the first mesh made
    by this backend or given a space."""
    _DEFAULT_MESH[0] = mesh


def _first(mesh):
    """mesh, the default mesh if there is none yet."""
    if _DEFAULT_MESH[0] is None:
        _DEFAULT_MESH[0] = mesh
    return mesh


class _Space(fem.FunctionSpace):
    """A DOLFINx FunctionSpace that knows its periodic box (`_box`) and, made by
    `sub`, the space it is a component of (`_parent`): a Dirichlet condition on
    W.sub(0) acts through W's unique dofs."""

    def sub(self, i):
        """Component i, made once (DOLFINx makes a new sub-dofmap at every call)."""
        subs = self.__dict__.setdefault("_subs", {})
        if i not in subs:
            subs[i] = _space(super().sub(i), self._box, parent=self)
        return subs[i]

    def unique_dofs(self):
        return unique_dofs(self)


def _space(V, box, parent=None):
    """DOLFINx's space V as a backend space, periodic as `box` (a box spec or None)."""
    S = _Space(V.mesh, V.ufl_element(), V._cpp_object)
    S._box = box
    if parent is not None:
        S._parent = parent
    return S


def FunctionSpace(mesh, family, degree=None, constrained_domain=None):
    """A space as dolfin makes one: FunctionSpace(mesh, 'CG', 2) or
    FunctionSpace(mesh, element), periodic when constrained_domain is a box spec."""
    element = FiniteElement(family, mesh.ufl_cell(), degree) if degree is not None else family
    return _space(fem.functionspace(_first(mesh), element), constrained_domain)


def unique_dofs(V):
    """The UniqueDofs of a space made by FunctionSpace or subspace (built once)."""
    U = getattr(V, "_unique_dofs", None)
    if U is None:
        if hasattr(V, "_parent"):
            raise ValueError("unique_dofs: a component W.sub(i) has W's unique dofs")
        if not hasattr(V, "_box"):
            raise ValueError("a space not made by femtools.backend.dolfinx (FunctionSpace or "
                             "subspace): its periodicity is unknown")
        box = V._box
        # Mixed spaces numbered node by node, as dolfin numbers them, and the nodes in a
        # bandwidth-reducing order: ILU and AMG depend on both.
        if box is None or not any(box.periodic):
            U = UniqueDofs(V, interleave=True, reorder=True)
        else:
            from femtools.periodic_core import box_scale
            gdim = V.mesh.geometry.dim
            # the box's tol is absolute (BoxSpec), periodic_box's relative to box_scale
            tol = max(float(getattr(box, "tol", 1e-10)) / box_scale(box.lower, box.upper),
                      1e-15)
            U = UniqueDofs(V, *periodic_box(V, upper=box.upper[:gdim],
                                            periodic=box.periodic[:gdim],
                                            lower=box.lower[:gdim], tol=tol),
                           interleave=True, reorder=True)
        V._unique_dofs = U
    return U


# Neutral helpers for the names dolfin and DOLFINx spell differently.

def mesh_id(mesh):
    """A key for the mesh, the same while it exists."""
    return id(mesh)


def function_space(u):
    """The space of Function `u`."""
    return u.function_space


def space_mesh(V):
    """The mesh of space `V`."""
    return V.mesh


def mesh_comm(mesh):
    """The mesh's MPI communicator (mpi4py)."""
    return mesh.comm


def geometric_dimension(mesh):
    """The dimension of the mesh's coordinates."""
    return mesh.geometry.dim


def element_degree(V):
    """The polynomial degree of space `V`'s element."""
    return V.ufl_element().embedded_superdegree


def is_lagrange(V):
    """Whether V's element is scalar continuous Lagrange."""
    el = V.ufl_element()
    return (el.num_sub_elements == 0 and el.reference_value_shape == ()
            and el.family_name == "P" and not el.discontinuous)


def owned_dofs(W, i):
    """The owned dofs of component i of the mixed space W (an index, or a path: (0, 1)
    is W.sub(0).sub(1)), as global indices of its vectors."""
    U = unique_dofs(W)
    lo, hi = U.owned_range
    c = np.unique(U.compact[_mixed.component_map(W, i)])
    return c[(c >= lo) & (c < hi)].astype(PETSc.IntType)


def fixed_dofs(W, bcs):
    """The owned dofs of W that the Dirichlet conditions `bcs` constrain, as global
    indices of its vectors."""
    return np.unique(np.concatenate(
        [bc.global_rows() for bc in bcs] or [np.zeros(0, PETSc.IntType)])).astype(
            PETSc.IntType)


# --- vectors and matrices --------------------------------------------------------------

class Vector:
    """dolfin's GenericVector API on a PETSc vector over unique dofs. Vector(x) is a
    copy of x, Vector(petsc_vec) wraps one."""

    def __init__(self, x=None):
        if isinstance(x, Vector):
            self._v = x._v.copy()
        elif isinstance(x, PETSc.Vec):
            self._v = x
        else:
            raise TypeError("Vector(x): a Vector to copy or a PETSc Vec")

    # access
    def vec(self):
        return self._v

    def mpi_comm(self):
        return self._v.getComm().tompi4py()

    def size(self):
        return self._v.getSize()

    def local_size(self):
        return self._v.getLocalSize()

    def __len__(self):
        return self._v.getLocalSize()

    def local_range(self):
        return self._v.getOwnershipRange()

    def get_local(self):
        return self._v.getArray(readonly=True).copy()

    def set_local(self, values):
        with self._v as a:
            a[:] = values

    def apply(self, mode=None):
        pass

    def update_ghost_values(self):
        pass

    def copy(self):
        return Vector(self)

    def __getitem__(self, index):
        return self._v.getArray(readonly=True)[index].copy()

    def __setitem__(self, index, value):
        if isinstance(value, Vector):
            value = value._v.getArray(readonly=True)
        with self._v as a:
            a[index] = value

    # algebra
    def zero(self):
        self._v.zeroEntries()

    def axpy(self, a, x):
        self._v.axpy(a, x._v)

    def norm(self, norm_type="l2"):
        return self._v.norm(_VEC_NORMS[norm_type.lower()])

    def sum(self):
        return self._v.sum()

    def inner(self, x):
        return self._v.dot(x._v)

    def max(self):
        return self._v.max()[1]

    def min(self):
        return self._v.min()[1]

    def _new(self):
        """A new Vector of the same layout, its values not set."""
        return Vector(self._v.duplicate())

    def __mul__(self, other):
        if not isinstance(other, Vector):
            out = self.copy()
            out._v.scale(float(other))
            return out
        out = self._new()
        out._v.pointwiseMult(self._v, other._v)
        return out

    __rmul__ = __mul__

    def __imul__(self, other):
        if isinstance(other, Vector):
            self._v.pointwiseMult(self._v, other._v)
        else:
            self._v.scale(float(other))
        return self

    def __truediv__(self, other):
        return self * (1.0 / float(other))

    def __itruediv__(self, other):
        self._v.scale(1.0 / float(other))
        return self

    def __add__(self, other):
        return self._plus(1.0, other)

    def __sub__(self, other):
        return self._plus(-1.0, other)

    def _plus(self, a, other):
        """self + a*other (a Vector or a number), a new Vector."""
        if isinstance(other, Vector):
            out = self._new()
            out._v.waxpy(a, other._v, self._v)
        else:
            out = self.copy()
            out._v.shift(a * float(other))
        return out

    def __iadd__(self, other):
        if isinstance(other, Vector):
            self._v.axpy(1.0, other._v)
        else:
            self._v.shift(float(other))
        return self

    def __isub__(self, other):
        if isinstance(other, Vector):
            self._v.axpy(-1.0, other._v)
        else:
            self._v.shift(-float(other))
        return self

    def __neg__(self):
        return self * -1.0


_VEC_NORMS = {"l1": PETSc.NormType.NORM_1, "l2": PETSc.NormType.NORM_2,
              "linf": PETSc.NormType.NORM_INFINITY}
_MAT_NORMS = {"l1": PETSc.NormType.NORM_1, "frobenius": PETSc.NormType.FROBENIUS,
              "linf": PETSc.NormType.NORM_INFINITY}


class Matrix:
    """dolfin's GenericMatrix API on a PETSc matrix over unique dofs. Matrix(A) is a
    copy of A (values, pattern and local-to-global maps), Matrix(petsc_mat) wraps one."""

    def __init__(self, A):
        if isinstance(A, Matrix):
            self._m = A._m.duplicate(copy=True)
            # its couplings and conditions (femtools.dolfinx.assembly), which the PETSc
            # copy does not carry: assembled into, the copy is filtered as A is
            filt = A._m.getAttr(_assembly._FILTER)
            if filt is not None:
                self._m.setAttr(_assembly._FILTER, filt)
        elif isinstance(A, PETSc.Mat):
            self._m = A
        else:
            raise TypeError("Matrix(A): a Matrix to copy or a PETSc Mat")
        # dolfin keeps a row's pattern when a Dirichlet condition zeroes it, so that
        # the matrix can be assembled into again.
        self._m.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)

    def mat(self):
        return self._m

    def mpi_comm(self):
        return self._m.getComm().tompi4py()

    def size(self, dim):
        return self._m.getSize()[dim]

    def zero(self):
        self._m.zeroEntries()

    def axpy(self, a, B, same_nonzero_pattern):
        structure = (PETSc.Mat.Structure.SAME_NONZERO_PATTERN if same_nonzero_pattern
                     else PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        self._m.axpy(a, B._m, structure)

    def norm(self, norm_type="frobenius"):
        return self._m.norm(_MAT_NORMS[norm_type.lower()])

    def __mul__(self, x):
        y = self._m.createVecLeft()
        self._m.mult(x._v, y)
        return Vector(y)

    def set_nullspace(self, null):
        self._m.setNullSpace(null.petsc)


def as_backend_type(x):
    """x itself: this backend's Vectors and Matrices are already PETSc-backed (`vec()`,
    `mat()`), as dolfin's as_backend_type gives them."""
    return x


# --- functions and constants -----------------------------------------------------------

class _Values(dolfinx.la.Vector):
    """DOLFINx's array of a backend Function (`u.x`), whose `scatter_forward()` -- the
    call that ends a direct write in DOLFINx (`u.x.array[...] = ...`) -- also copies the
    values into the Function's compact vector (`restrict`)."""

    def __init__(self, cpp, owner):
        super().__init__(cpp)
        self._owner = weakref.ref(owner)

    @property
    def array(self):
        """DOLFINx's local array, its owned entries first brought up to the compact
        vector's values (locally, no communication): a write through it then starts
        from the Function's values, not from what the array held before."""
        owner = self._owner()
        if owner is not None:
            owner._refresh_owned()
        return self._cpp_object.array

    def scatter_forward(self):
        super().scatter_forward()
        owner = self._owner()
        if owner is not None:
            owner.restrict(ghosts_fresh=True)


def _point_evaluation(V):
    """Whether the dofs of space V are its values at its dof coordinates: a scalar or
    blocked space of a Lagrange-type element, not mixed and not a component."""
    if V.component():
        return False
    e = V.ufl_element()
    if e.block_size > 1:
        e = e.sub_elements[0]
    try:
        return bool(e.basix_element.interpolation_is_identity)
    except (AttributeError, NotImplementedError):
        return False


def _interpolate(u, f, *args, **kwargs):
    """DOLFINx's u.interpolate(f). A callable f into a space whose dofs are point
    values is evaluated once per dof, at the dof coordinates; DOLFINx would evaluate it
    at every cell's points, several times as many."""
    V = u.function_space
    if callable(f) and not isinstance(f, fem.Function) and not args and not kwargs \
            and _point_evaluation(V):
        x = V.tabulate_dof_coordinates()          # (dofs, 3), contiguous
        values = np.asarray(f(x.T), dtype=u.x.array.dtype)
        u.x.array[:] = values.T.ravel() if values.ndim > 1 else values
        return
    fem.Function.interpolate(u, f, *args, **kwargs)


class _WritesBack(fem.Function):
    """What a backend Function and its component views share: DOLFINx's ways of writing
    the array (`interpolate`, `x.array[...] = ...` then `x.scatter_forward()`) end in the
    compact vector too. (A subclass, not a mixin: FFCx finds a coefficient's handler
    along the first bases.)"""

    def _track(self, owner):
        self._x = _Values(self._cpp_object.x, owner)

    def interpolate(self, u0, *args, **kwargs):
        owner = self._owner()
        owner._refresh_owned()        # a component's interpolation keeps the others' values
        _interpolate(self, _interpolant(u0), *args, **kwargs)
        owner.restrict()
        return self


class Function(_WritesBack):
    """A DOLFINx Function with dolfin's `vector()`: the compact vector over the space's
    unique dofs (see the module's docstring for the two layouts)."""

    def __init__(self, V, name=None):
        super().__init__(V, name="f" if name is None else name)
        self._unique = unique_dofs(V)
        self._vector = Vector(self._unique.create_vector())
        self._synced = None        # the compact vector's PETSc state DOLFINx's array has
        self._owned_state = None   # ... the state its owned entries have (`_refresh_owned`)
        self._track(self)

    def _owner(self):
        return self

    def vector(self):
        return self._vector

    def _stale(self):
        """Whether the compact vector changed since DOLFINx's array had it (this rank)."""
        return self._vector._v.stateGet() != self._synced

    def _prolong(self):
        v = self._vector._v
        self._unique.prolong(v, self)
        self._synced = self._owned_state = v.stateGet()

    def sync(self):
        """The compact vector into DOLFINx's array (owned, ghost and slave dofs), when it
        changed since on any rank (PETSc counts every write to a vector: its state).
        Collective: the prolongation is a scatter, so every rank takes part when one
        rank's vector changed (a write on some ranks only)."""
        if agreed(self._unique.comm, self._stale()):
            self._prolong()
        return self

    def _refresh_owned(self):
        """The owned entries of DOLFINx's array from the compact vector, when it changed
        since: local, no communication (ghosts and slaves wait for `sync`)."""
        state = self._vector._v.stateGet()
        if state != self._synced and state != self._owned_state:
            self._cpp_object.x.array[self._unique.owned_unique] = \
                self._vector._v.getArray(readonly=True)
            self._owned_state = state

    def restrict(self, ghosts_fresh=False):
        """DOLFINx's array into the compact vector, after it was written directly. The
        array counts as the vector's prolongation only when its ghosts were just updated
        (`ghosts_fresh`: a `scatter_forward`) and the space has no periodic slaves (on any
        rank): else it is prolonged again at the next use -- a component's interpolation
        leaves the other components' ghosts as they were, and slaves hold what was
        written at their own positions, not their masters'."""
        v = self._vector._v
        self._unique.restrict(self, v)
        v.stateIncrease()
        self._owned_state = v.stateGet()
        self._synced = self._owned_state if ghosts_fresh and not self._unique.any_slave \
            else None
        return self

    def sub(self, i):
        """A view of component i, whose values are this Function's: writes to it reach
        the compact vector as the Function's own do."""
        self.sync()
        return _ComponentView(self, i)

    def split(self, deepcopy=False):
        """The components, as dolfin's split: views onto this Function, or with `deepcopy`
        Functions of their own on the components' collapsed spaces (copies)."""
        n = self.function_space.num_sub_spaces
        if not deepcopy:
            return tuple(self.sub(i) for i in range(n))
        return tuple(subfunction(self, i, Function(subspace(self.function_space, i)))
                     for i in range(n))

    def rename(self, name, label=None):
        self.name = name

    def assign(self, other):
        """The values of `other`, a Function on the same space (dolfin's assign)."""
        if not (isinstance(other, Function) and other.function_space is self.function_space):
            raise NotImplementedError("assign: a Function on the same space only")
        other._vector._v.copy(self._vector._v)


class _ComponentView(_WritesBack):
    """Component i of a backend Function, a view of its array (DOLFINx's `sub`)."""

    def __init__(self, parent, i):
        super().__init__(parent.function_space.sub(i), parent.x,
                         name="{}_{}".format(parent.name, i))
        self._parent = parent
        self._track(parent)

    def _owner(self):
        return self._parent


def _function_of(f):
    """The backend Function whose arrays f's are: f, or a component view's parent."""
    if isinstance(f, Function):
        return f
    return getattr(f, "_parent", None)


def sync(f):
    """Bring DOLFINx's array of f up to date with its compact vector (if f is a
    backend Function, or a view of one's component); returns f. Collective."""
    if isinstance(f, Function):
        f.sync()
    elif hasattr(f, "_parent"):
        f._parent.sync()
    return f


def collect_garbage():
    """Destroy the PETSc objects Python has dropped, which petsc4py keeps until then
    (femtools.garbage). Collective: call it at the same point on every rank, such as
    once a time step."""
    from femtools.garbage import cleanup
    cleanup()


def constant_on(value, mesh):
    """A Constant for forms on `mesh` (the default mesh would serve forms on another
    one ill: DOLFINx's Constants belong to a mesh)."""
    return Constant(value, mesh)


class Constant(fem.Constant):
    """dolfin's Constant(value): on the last mesh a space was made on, unless given."""

    def __init__(self, value, mesh=None):
        mesh = _DEFAULT_MESH[0] if mesh is None else mesh
        if mesh is None:
            raise RuntimeError("Constant: no mesh yet (make a FunctionSpace first, or "
                               "pass mesh=)")
        super().__init__(mesh, np.asarray(value, dtype=PETSc.ScalarType))

    def values(self):
        return np.atleast_1d(np.asarray(self.value, dtype=float)).ravel().copy()

    def follow(self, source):
        """Take source's value (a Constant, on another mesh) at every assembly with this
        one. Returns self."""
        self._source = source
        self.value = source.value
        return self

    def assign(self, value):
        """The value of a number, an array or another Constant."""
        self.value = value.value if isinstance(value, fem.Constant) else value


def _expression_space(mesh, element):
    """The space Expressions in `element` on `mesh` are interpolated into, one for all
    of them (with its unique dofs). Kept on the mesh, so it goes with it."""
    spaces = mesh.__dict__.setdefault("_expression_spaces", {})
    if element not in spaces:
        spaces[element] = _space(fem.functionspace(mesh, element), None)
    return spaces[element]


class Expression(Function):
    """dolfin's Expression(cpp_code, degree=..., **parameters): C++ code in x[0], x[1],
    x[2] and the parameters, compiled as dolfin compiles it
    (femtools.dolfinx.expression.CompiledExpression: <cmath>, near(), DOLFIN_EPS, pi;
    the conditional and logical operators and integer division as in C++); a tuple of
    strings is a vector. The parameters are attributes (expr.t = 1.0). Making one is
    collective (rank 0 compiles, once per code; the library is cached).

    interpolate and set_subfunction evaluate the code at the target's dofs. In a form it
    is, as in dolfin, its interpolant of the given degree (Lagrange, on `domain`, else
    the last mesh a space was made on), or in the given `element` (FiniteElement),
    computed again when a parameter has changed."""

    def __init__(self, cpp_code, degree=2, element=None, domain=None, **params):
        from femtools.dolfinx.expression import CompiledExpression
        self._share(CompiledExpression(cpp_code, MPI.comm_world, **params), element)
        mesh = _DEFAULT_MESH[0] if domain is None else domain
        if mesh is None:
            raise RuntimeError("Expression: no mesh yet (make a FunctionSpace first)")
        self._on(mesh, degree)

    def on_mesh(self, mesh):
        """This Expression on `mesh` (the dual mesh's phase-field mesh): the same compiled
        code and parameters, interpolated there; dolfin's Expressions belong to no mesh."""
        twin = self._twins.get(id(mesh))
        if twin is None:
            twin = object.__new__(Expression)
            twin._share(self._c, self._element)
            twin._on(mesh, self._degree)
            self._twins[id(mesh)] = twin
        return twin

    def _share(self, compiled, element):
        for key, value in (("_c", compiled), ("_dirty", True), ("_twins", {}),
                           ("_element", element)):
            object.__setattr__(self, key, value)

    def _on(self, mesh, degree):
        object.__setattr__(self, "_degree", degree)
        n = self._c.value_size
        if self._element is not None:
            el = self._element
            if (n > 1) != bool(el.reference_value_shape):
                raise ValueError("Expression: element and code differ in value shape")
        else:
            if degree == 0:
                family, degree = "DG", 0
            else:
                family, degree = "Lagrange", degree if degree else 2
            el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree,
                                   discontinuous=family == "DG",
                                   shape=(n,) if n > 1 else ())
        super().__init__(_expression_space(mesh, el), name="Expression")

    @property
    def _user_parameters(self):
        return self._c.parameters

    def __getattr__(self, key):
        c = self.__dict__.get("_c")
        if c is not None and key in c._names:
            return getattr(c, key)
        raise AttributeError(key)

    def __setattr__(self, key, value):
        c = self.__dict__.get("_c")
        if c is not None and key in c._names:
            setattr(c, key, value)
            object.__setattr__(self, "_dirty", True)
            for twin in self.__dict__.get("_twins", {}).values():
                object.__setattr__(twin, "_dirty", True)
        else:
            object.__setattr__(self, key, value)

    def evaluate(self, x):
        """The values at points x (an array of shape (3, n))."""
        return self._c(x)

    def sync(self):
        if self._dirty:
            _interpolate(self, self.evaluate)
            self.restrict()
            object.__setattr__(self, "_dirty", False)
        return self


# --- assembly --------------------------------------------------------------------------

CAPACITY = 256


def _held(obj, name, capacity):
    """An LRU kept as attribute `name` of `obj` (a mesh or a space), made on first use.
    A cache held by what its entries are built on goes with it: entry and holder form a
    cycle Python's collector frees once nothing else holds them, where a module-level
    cache would keep every mesh it ever saw (a refinement ladder's every level). `obj`
    None (a form on no mesh found): one module-level LRU per name."""
    if obj is None:
        return _UNHELD.setdefault(name, LRU(capacity))
    cache = obj.__dict__.get(name)
    if cache is None:
        cache = LRU(capacity)
        setattr(obj, name, cache)
    return cache


_UNHELD = {}


def _form_mesh(form):
    """The mesh of a UFL form's arguments or coefficients (None if it has neither)."""
    for a in form.arguments():
        return a.ufl_function_space().mesh
    for c in form.coefficients():
        V = getattr(c, "function_space", None)
        if V is not None and not callable(V):
            return V.mesh
    return None


def _for_dolfinx(form):
    """The form as DOLFINx compiles it: the global quadrature degree on every integral
    that sets none (as dolfin does), and the BoundaryMarkers its measures were made with
    as their current meshtags (dolfin reads its facet MeshFunction at each assembly)."""
    degree = parameters["form_compiler"]["quadrature_degree"]
    integrals = []
    for itg in form.integrals():
        kw = {}
        if degree is not None:
            kw["metadata"] = dict({"quadrature_degree": degree}, **itg.metadata())
        if isinstance(itg.subdomain_data(), BoundaryMarkers):
            kw["subdomain_data"] = itg.subdomain_data().meshtags()
        integrals.append(itg.reconstruct(**kw) if kw else itg)
    return ufl.Form(integrals)


def _jit_options():
    """DOLFINx's JIT options from dolfin's form_compiler parameters: cpp_optimize_flags
    are the C compiler's flags. (dolfin's other ones, optimize, cpp_optimize and
    representation, choose among FFC's code generators; FFCx has one.)"""
    flags = parameters["form_compiler"].get("cpp_optimize_flags")
    return {"cffi_extra_compile_args": flags.split()} if flags else {}


class _Compiled:
    def __init__(self, form):
        # the BoundaryMarkers the integrals are over, as they were when compiled
        self.markers = [(itg.subdomain_data(), itg.subdomain_data().version)
                        for itg in form.integrals()
                        if isinstance(itg.subdomain_data(), BoundaryMarkers)]
        # the global quadrature degree it was compiled at: a change compiles it again, as
        # dolfin's JIT signature includes it
        self.degree = parameters["form_compiler"]["quadrature_degree"]
        self.cpp = fem.form(_for_dolfinx(form), jit_options=_jit_options())
        self.coefficients = [c for c in form.coefficients()
                             if isinstance(c, Function) or hasattr(c, "_parent")]
        # Constants that follow another (a copy on another mesh: Constant.follow)
        self.constants = [c for c in form.constants() if hasattr(c, "_source")]
        args = sorted(form.arguments(), key=lambda a: a.number())
        self.spaces = [a.ufl_function_space() for a in args]
        self.rank = len(args)
        self.comm = self.cpp.mesh.comm


def compiled(form):
    """The compiled form of a UFL form, made once per equal form (while among the
    CAPACITY most recently used): a form made anew each step is compiled once. Looked
    up by the form object first, then by its content. The caches are the mesh's (`_held`):
    they go with it."""
    mesh = _form_mesh(form)
    by_object = _held(mesh, "_femtools_forms", CAPACITY)       # (form, its _Compiled)
    by_content = _held(mesh, "_femtools_form_contents", CAPACITY)   # by equal form
    held = by_object.get(id(form))
    entry = held[1] if held is not None and held[0] is form else None
    if entry is None:
        entry = by_content.get(form)
    if entry is None or any(m.version != v for m, v in entry.markers) or \
            entry.degree != parameters["form_compiler"]["quadrature_degree"]:
        entry = by_content[form] = _Compiled(form)
        if by_content.evicted == 1 and entry.comm.rank == 0:
            import sys
            print("femtools: more than {} distinct forms compiled. Each holds DOLFINx's "
                  "integration entities; a form made with other values each time (a "
                  "Python number where a Constant belongs) is compiled again every "
                  "time.".format(CAPACITY), file=sys.stderr)
    by_object[id(form)] = (form, entry)   # the form held: its id is not reused
    # the coefficients brought up to date: decided on every rank together, in one
    # reduction (a prolongation is a scatter, and a write may have been on some ranks)
    owners = list({id(o): o for o in (_function_of(c) for c in entry.coefficients)
                   if o is not None}.values())
    for o in owners:
        # A Function whose class has its own sync first runs it, as assembly always did:
        # an Expression's code evaluated into its arrays, or a copy that follows another
        # Function brought up to date. Called on every rank, as compiled() is.
        if type(o).sync is not Function.sync:
            o.sync()
    if owners:
        stale = np.array([o._stale() for o in owners], dtype=np.int64)
        if entry.comm.size > 1:
            from femtools.hdf5_rows import _int_reduce
            stale = _int_reduce(entry.comm, stale, "max")
        for o, s in zip(owners, stale):
            if s:
                o._prolong()
    for c in entry.constants:
        c.value = c._source.value
    return entry


def assemble(form, tensor=None, couplings=None, bcs=()):
    """dolfin's assemble: a Matrix, a Vector or a number, into `tensor` if given (a
    Matrix made for the same spaces, or a Vector over the test space's unique dofs). A
    Matrix made here with `couplings` (None, "form" or a mask: femtools.couplings)
    stores only the blocks of the coupled components (and the diagonal), and keeps them
    for later assemblies into it; a nonzero outside them raises. With Dirichlet
    conditions `bcs` (on a square form) the rows and columns of their dofs hold only the
    diagonal, the number of cells holding the dof: the conditions applied symmetrically,
    as dolfin's assemble_system applies them."""
    c = compiled(form)
    if c.rank == 0:
        return c.comm.allreduce(fem.assemble_scalar(c.cpp), op=_MPI.SUM)
    if c.rank == 1:
        U = unique_dofs(c.spaces[0])
        b = Vector(U.create_vector()) if tensor is None else tensor
        _assembly.assemble_vector(b._v, c.cpp, U)
        return b
    if tensor is None:
        from femtools.couplings import resolve
        tensor = Matrix(_assembly.create_matrix(c.cpp, unique_dofs(c.spaces[0]),
                                                unique_dofs(c.spaces[1]),
                                                couplings=resolve(couplings, form),
                                                bcs=tuple(bcs)))
    elif couplings is not None or bcs:
        # the same on every rank: the arguments are
        raise ValueError("assemble: couplings and bcs are the matrix's, given when it is made; "
                         "assemble into it without them")
    _assembly.assemble_matrix(tensor._m, c.cpp)
    return tensor


def assemble_matrix(form, bcs=()):
    """The matrix of the bilinear `form` with the Dirichlet conditions `bcs` applied,
    assembled once per equal form and conditions (femtools.fast_assembly.constant_matrix):
    the same Matrix is returned to every caller (kept by the form's mesh, `_held`)."""
    from femtools.fast_assembly import constant_matrix
    return constant_matrix(form, bcs, assemble,
                           cache=_held(_form_mesh(form), "_femtools_matrices", 64))


# --- boundaries and Dirichlet conditions -----------------------------------------------

# ufl.Measure, taking a BoundaryMarkers as subdomain data, read when a form is compiled.
Measure = ufl.Measure


class DirichletBC:
    """dolfin's DirichletBC(V, value, markers, marker) on a scalar space, a scalar
    component W.sub(i) of a mixed one, or a vector component (Taylor-Hood's velocity
    W.sub(0), each of its components in turn); `value` a number, a Constant (a vector one
    on a vector component, or a tuple), or (on a whole scalar space) a Function on V;
    DirichletBC(bc) copies one. It acts on the
    compact vectors and matrices: `apply(A)` sets the constrained rows to the
    identity's (keeping the pattern, as dolfin does), `apply(b)` sets the values,
    `apply(A, b)` both. A periodic slave on the boundary is its master (on the
    boundary too, where the data agree)."""

    METHODS = ("topological", "geometric", "pointwise")

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], DirichletBC):
            other = args[0]
            self.V, self.value, self._local = other.V, other.value, other._local
            self._unique, self._rows, self._comp = other._unique, other._rows, other._comp
            return
        if len(args) in (3, 4) and isinstance(args[2], str):
            # dolfin's DirichletBC(V, value, "C++ code"[, method]): the facets where the
            # code holds, compiled (compiled_facets); "pointwise", the dofs where it holds
            V, value, code = args[:3]
            method = args[3] if len(args) == 4 else "topological"
            if method not in self.METHODS:
                raise ValueError("DirichletBC: method {!r}, not one of {}".format(
                    method, self.METHODS))
        elif len(args) in (4, 5) and isinstance(args[2], BoundaryMarkers):
            V, value, markers, marker = args[:4]
            method = args[4] if len(args) == 5 else "topological"
            if method not in ("topological", "geometric"):
                raise ValueError("DirichletBC: method {!r} on facet markers, not "
                                 "'topological' or 'geometric'".format(method))
            if markers.mesh is not V.mesh:
                raise ValueError("DirichletBC: the markers are on another mesh than the "
                                 "space")
        else:
            raise NotImplementedError("DirichletBC(V, value, markers, marker[, method]) or "
                                      "DirichletBC(V, value, \"C++ code\"[, method]) only")
        # a vector component (num_sub_spaces > 0, numbered as its mixed space) acts
        # through its scalar components, one after the other
        comps = [V.sub(j) for j in range(V.num_sub_spaces)] if V.num_sub_spaces else [V]
        if V.dofmap.index_map_bs != 1 or any(
                c.num_sub_spaces or c.dofmap.index_map_bs != 1 for c in comps):
            raise NotImplementedError("DirichletBC on a scalar space or component, or a "
                                      "vector component of a mixed space")
        self.V = V
        self.value = value
        # a component's dofs are numbered as its mixed space's (its root's)
        root = V
        while hasattr(root, "_parent"):
            root = root._parent
        self._unique = U = unique_dofs(root)
        # For Lagrange elements dolfin's "geometric" and "topological" agree: the dofs on
        # the marked facets' closures.
        if method != "pointwise":
            facets = (compiled_facets(V.mesh, code) if isinstance(args[2], str)
                      else markers.where(marker))
        rows = []
        for c in comps:
            dofs = (compiled_dofs(c, code) if method == "pointwise" else
                    fem.locate_dofs_topological(c, V.mesh.topology.dim - 1, facets))
            rows.append(_owned_rows(U, dofs))
        self._local = np.concatenate(rows)
        # each row's component, for a vector value (None on a scalar space)
        self._comp = (np.concatenate([np.full(len(r), j, dtype=np.int64)
                                      for j, r in enumerate(rows)])
                      if V.num_sub_spaces else None)
        self._rows = None

    def function_space(self):
        return self.V

    def homogenize(self):
        self.value = 0.0

    def global_rows(self):
        """The constrained owned rows, as global indices of the compact vectors (made
        once)."""
        if self._rows is None:
            self._rows = (self._local + self._unique.owned_range[0]).astype(PETSc.IntType)
        return self._rows

    def _value(self):
        """The values at the constrained owned dofs: an array in the order of `_local`
        for a Function or a vector value, else one number."""
        g = self.value
        if isinstance(g, Function):
            if g.function_space is not self.V or self._comp is not None:
                raise NotImplementedError("DirichletBC: a Function value on another space")
            return g.vector()._v.getArray(readonly=True)[self._local]
        v = np.asarray(g.value if isinstance(g, fem.Constant) else g, dtype=float).ravel()
        if v.size == 1:
            return float(v[0])
        if self._comp is None or v.size != self.V.num_sub_spaces:
            raise ValueError("DirichletBC: a value of {} components on a space of {}".format(
                v.size, self.V.num_sub_spaces or 1))
        return v[self._comp]

    def values(self):
        """The values at the constrained owned dofs (in the order of `_local`)."""
        return np.broadcast_to(self._value(), self._local.shape).copy()

    def get_boundary_values(self):
        """{local dof: value}, the owned constrained dofs (in the compact numbering)."""
        return dict(zip(self._local.tolist(), self.values().tolist()))

    def apply(self, *tensors):
        for t in tensors:
            if isinstance(t, Matrix):
                # the rows are this rank's own: PETSc need not route them to owners
                t._m.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)
                t._m.zeroRows(self.global_rows(), diag=1.0)
            elif isinstance(t, Vector):
                with t._v as a:
                    a[self._local] = self._value()
            else:
                raise TypeError("DirichletBC.apply: a Matrix or a Vector")


def compiled_dofs(V, code):
    """The local dofs of V (a space, or a component W.sub(i), whose dofs are W's) where
    `code` holds at their coordinates, as dolfin's pointwise DirichletBC finds them
    (on_boundary false)."""
    from femtools.dolfinx.expression import CompiledExpression
    if hasattr(V, "_parent"):
        Vc, parent = V.collapse()
        x, dofs = Vc.tabulate_dof_coordinates(), np.asarray(parent, dtype=np.int32)
    else:
        x = V.tabulate_dof_coordinates()
        dofs = np.arange(len(x), dtype=np.int32)
    params = {"on_boundary": 0.0} if "on_boundary" in code else {}
    expr = CompiledExpression(code, V.mesh.comm, **params)
    inside = np.asarray(expr(x.T), dtype=float) != 0.0 if len(x) else np.zeros(0, bool)
    return dofs[inside]


def _owned_rows(U, dofs):
    """The rows (local indices of the compact vector) of local dofs `dofs` of U's space,
    on the ranks that own them: a periodic slave's master may be owned by a rank that
    does not see the slave, so every rank's dofs go to their owner (one reverse scatter).
    Collective."""
    plain = U.plain_vector()
    with plain.localForm() as loc:
        loc.zeroEntries()
        with loc as a:
            a[np.asarray(dofs, dtype=np.int64)] = 1.0
    hits = U.fold(plain, U.create_vector())
    rows = np.flatnonzero(hits.getArray(readonly=True) > 0.0).astype(PETSc.IntType)
    hits.destroy()
    return rows


def apply_bc(bc, *tensors):
    """bc.apply(*tensors); `bc` may be a list or tuple of conditions, applied in their
    order (as femtools.fast_bc.apply on dolfin)."""
    for one in (bc if isinstance(bc, (list, tuple)) else [bc]):
        one.apply(*tensors)


# --- interpolation, mixed components, null spaces --------------------------------------

def interpolate(f, V, out=None):
    """dolfin.interpolate(f, V), into out (a Function on V, made if None)."""
    out = Function(V) if out is None else out
    _interpolate(out, _interpolant(f))
    return out.restrict()


def _interpolant(f):
    """What DOLFINx's Function.interpolate takes for f: an Expression's or a constant's
    values at points, or a Function (brought up to date)."""
    if isinstance(f, Expression):
        return f.evaluate
    if isinstance(f, (fem.Constant, int, float)):
        value = np.asarray(f.value if isinstance(f, fem.Constant) else f, dtype=float)
        if value.size != 1:
            raise NotImplementedError("interpolate: a scalar constant only")
        return lambda x: np.full(x.shape[1], float(value))
    return sync(f)


def subspace(W, i):
    """The collapsed space of component i of the mixed space W (made once,
    femtools.dolfinx.mixed), periodic as W is."""
    S = _mixed.subspace(W, i)
    if not hasattr(S, "_box"):
        S._box = W._box
    return S


def subfunction(q, i, out=None):
    """Component i of the mixed Function q as a Function on its collapsed space: `out`
    if given, else the same Function for every call on the same space and component
    (its values last until the next such call). Returns it."""
    W = q.function_space
    if out is None:
        held = _held(W, "_femtools_subfunctions", 64)          # W's: it goes with W
        out = held.get(i)
        if out is None:
            out = held[i] = Function(subspace(W, i))
    q.sync()
    _mixed.subfunction(q, i, out=out)
    return out.restrict()


def set_subfunction(q, i, value):
    """Set component i of the mixed Function q to `value`: a Function on its
    collapsed space, or anything Function.interpolate takes (a callable of the
    coordinates, a Function on another space of the mesh, an Expression)."""
    S = subspace(q.function_space, i)
    if not (isinstance(value, fem.Function) and value.function_space is S):
        value = Function(S).interpolate(value)
    else:
        sync(value)
    q.sync()
    _mixed.set_subfunction(q, i, value)     # ends in q.x.scatter_forward: q.restrict


def nodal_values(f, V):
    """The values of f at V's owned (unique) dofs, as an array: interpolate(f, V)'s
    get_local(), into a Function kept by the space (it goes with V)."""
    out = V.__dict__.get("_femtools_values")
    if out is None:
        out = Function(V)
        setattr(V, "_femtools_values", out)
    return interpolate(f, V, out=out).vector().get_local()


class AssignedVectorFunction(Function):
    """A vector Function on the vector space of the scalar Functions `u` (all on one
    space, periodic as it is); calling it copies their values in and returns it
    (femtools.mixed.AssignedVectorFunction)."""

    def __init__(self, u, name="Assigned Vector Function"):
        V = u[0].function_space
        Vv = fem.functionspace(V.mesh, basix.ufl.blocked_element(V.ufl_element(),
                                                                 shape=(len(u),)))
        super().__init__(_space(Vv, V._box), name=name)
        self.u = list(u)
        self._maps = [flat_collapse_map(Vv.sub(i).collapse()[1]) for i in range(len(u))]

    def __call__(self):
        for m, ui in zip(self._maps, self.u):
            self.x.array[m] = sync(ui).x.array
        return self.restrict()


class _NullSpace:
    """dolfin's VectorSpaceBasis of the constants, on a PETSc NullSpace."""

    def __init__(self, petsc):
        self.petsc = petsc

    def orthogonalize(self, x):
        self.petsc.remove(x._v)


def constant_nullspace(A, V):
    """The constants on the scalar space V, or on component V = W.sub(i) of a mixed space
    W (its dofs alone, the others zero: a Stokes pressure's), as a null space attached to
    A."""
    parent = getattr(V, "_parent", None)
    if parent is None:
        null = _NullSpace(_assembly.constant_nullspace(unique_dofs(V)))
    else:
        component = list(V.component())
        if len(component) != 1:
            raise NotImplementedError("constant_nullspace: a component of a mixed space, "
                                      "not of one of its components")
        U = unique_dofs(parent)
        v = U.create_vector()
        v.getArray()[owned_dofs(parent, component[0]) - U.owned_range[0]] = 1.0
        v.normalize()
        null = _NullSpace(PETSc.NullSpace().create(vectors=[v], comm=U.comm))
    A.set_nullspace(null)
    return null


def norm(x, norm_type="L2", mesh=None):
    """dolfin's norm: of a Vector, "l1", "l2" or "linf"; of a Function, the integral norm
    "L2", "H1" or "H10" (the name in either case). `mesh` is dolfin's, for a norm of an
    Expression; this backend's Expressions are Functions on a mesh of their own, so it is
    taken and not needed. Collective."""
    t = norm_type.lower()
    if isinstance(x, fem.Function):
        sync(x)
        if t == "l2":
            form = ufl.inner(x, x) * ufl.dx
        elif t == "h1":
            form = (ufl.inner(x, x) + ufl.inner(ufl.grad(x), ufl.grad(x))) * ufl.dx
        elif t == "h10":
            form = ufl.inner(ufl.grad(x), ufl.grad(x)) * ufl.dx
        else:
            raise ValueError("norm of a Function: L2, H1 or H10, not {!r}".format(norm_type))
        return float(np.sqrt(assemble(form)))
    return x.norm(t)


def normalize(x, normalization_type="average"):
    """Subtract the average of x from it (dolfin's normalize); returns the average."""
    if normalization_type != "average":
        raise NotImplementedError("normalize: 'average' only")
    average = x.sum() / x.size()
    x._v.shift(-average)
    return average


# --- solvers ---------------------------------------------------------------------------

class PETScKrylovSolver:
    """dolfin's PETScKrylovSolver on a petsc4py KSP: `ksp()`, `parameters` (applied at
    every solve, as dolfin does), set_reuse_preconditioner, set_operator(A),
    set_operators(A, P), and solve(A, x, b) or solve(x, b) on the operators set."""

    _COUNT = [0]

    def __init__(self, method=None, preconditioner=None):
        self._ksp = PETSc.KSP().create(MPI.comm_world)
        self.parameters = {}
        self._reuse = False
        self._operators = None        # (A, P), Matrices, from set_operator(s)
        if method is not None:
            # dolfin's PETScKrylovSolver(method, preconditioner), by femtools.ksp's names
            self._COUNT[0] += 1
            ksp.configure(self._ksp, "krylov_{}_".format(self._COUNT[0]),
                          ksp.solver_options(method, preconditioner or "default"))

    def ksp(self):
        return self._ksp

    def set_reuse_preconditioner(self, reuse):
        self._reuse = bool(reuse)

    def set_operator(self, A):
        """A the operator and the preconditioner's matrix, for solve(x, b)."""
        self._operators = (A, A)

    def set_operators(self, A, P):
        """A the operator, P the matrix the preconditioner is built from."""
        self._operators = (A, P)

    def solve(self, *args):
        """solve(A, x, b), or solve(x, b) with the operators set; returns the iterations.
        Raises unless it converged (parameters["error_on_nonconvergence"])."""
        if len(args) == 3:
            A, x, b = args
            P = A
        elif len(args) == 2:
            if self._operators is None:
                raise RuntimeError("PETScKrylovSolver.solve(x, b): no operator set "
                                   "(set_operator or set_operators)")
            (A, P), (x, b) = self._operators, args
        else:
            raise TypeError("PETScKrylovSolver.solve(A, x, b) or solve(x, b)")
        k = self._ksp
        k.setOperators(A._m, P._m)
        k.getPC().setReusePreconditioner(self._reuse)  # what KSPSetReusePreconditioner does
        ksp.apply_parameters(k, self.parameters)
        k.solve(b._v, x._v)
        reason = k.getConvergedReason()
        its = k.getIterationNumber()
        if reason < 0 and self.parameters.get("error_on_nonconvergence", True):
            raise RuntimeError("PETScKrylovSolver: the solve did not converge (PETSc reason "
                               "{}, {} iterations)".format(reason, its))
        if self.parameters.get("report") and MPI.comm_world.rank == 0:
            print("PETSc Krylov solver ({}, {}) converged in {} iterations.".format(
                k.getType(), k.getPC().getType(), its))
        return its


# --- linear variational problems ----------------------------------------------------

def solve(equation, u, bcs=None):
    """dolfin's solve(a == L, u, bcs) for a linear problem: the system assembled, each
    condition applied to its rows (bc.apply(A, b), as dolfin's LinearVariationalSolver
    does), and solved by LU (MUMPS), factorized at every call as dolfin does. The
    matrix and the solver are kept per bilinear form (equal forms share them; the 8 most
    recent, by the form's mesh)."""
    a, L = equation.lhs, equation.rhs
    bcs = [] if bcs is None else (list(bcs) if isinstance(bcs, (list, tuple)) else [bcs])
    linear = _held(_form_mesh(a), "_femtools_linear", 8)       # the mesh's: it goes with it
    entry = linear.get(a)
    if entry is None:
        A = assemble(a)
        solver = PETSc.KSP().create(MPI.comm_world)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.getPC().setFactorSolverType("mumps")
        entry = linear[a] = (A, solver)
    else:
        assemble(a, tensor=entry[0])
    A, solver = entry
    b = assemble(L)
    for bc in bcs:
        bc.apply(A, b)
    solver.setOperators(A._m)
    solver.solve(b._v, u.vector()._v)
    if solver.getConvergedReason() < 0:
        raise RuntimeError("solve: LU failed (PETSc reason {})".format(
            solver.getConvergedReason()))
    return u


# --- output ----------------------------------------------------------------------------

class XDMFFile:
    """dolfin's XDMFFile(comm, path) for what the driver writes: facet markers (a
    BoundaryMarkers) and Functions, each with its mesh."""

    def __init__(self, comm, path):
        self.comm, self.path = comm, path
        self._file = self._series = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for f in (self._file, self._series):
            if f is not None:
                f.close()
        self._file = self._series = None

    def write(self, obj, t=None):
        """Facet markers (a BoundaryMarkers) through DOLFINx's XDMFFile; a Function, or a
        tuple of them on one node set (`write((u, p))`), as femtools.xdmf.XDMFFile2 writes
        them (femtools.dolfinx.xdmf: P2 kept)."""
        if isinstance(obj, BoundaryMarkers):
            if self._file is None:
                self._file = dolfinx.io.XDMFFile(self.comm, self.path, "w")
                self._file.write_mesh(obj.mesh)
            self._file.write_meshtags(obj.meshtags(), obj.mesh.geometry)
        else:
            if self._series is None:
                from femtools.dolfinx.xdmf import XDMFSeries
                self._series = XDMFSeries(self.comm, self.path)
            fields = (tuple(sync(f) for f in obj) if isinstance(obj, (tuple, list))
                      else sync(obj))
            self._series.write(fields, 0.0 if t is None else float(t))


class TimeSeriesFile:
    """A field's time series, `write(function, t)` and `close()`: XDMF in the layout
    dolfin's TimeSeriesFile (femtools.xdmf.XDMFFile2) writes, word for word
    (femtools.dolfinx.xdmf.XDMFSeries): a P2 field keeps its midside nodes, the mesh is
    stored once, and both backends' series read, merge and continue alike."""

    def __init__(self, comm, path):
        from femtools.dolfinx.xdmf import XDMFSeries
        self.path = path
        self._series = XDMFSeries(comm, path)

    def write(self, obj, t):
        self._series.write(sync(obj), float(t))

    def close(self):
        self._series.close()


# --- checkpoints: dolfin's HDF5 layout (femtools.dolfinx.checkpoint) --------------------

def space_id(V):
    """A key for space V, the same while V exists."""
    return id(V)


def write_dof_tables(path, V, name="dofs", mode="w"):
    """V's dof tables in dolfin's layout (femtools.dolfin_layout.write_dofs)."""
    from femtools.dolfinx.checkpoint import write_dofs
    write_dofs(path, V, unique_dofs(V), name, mode)


def write_function_h5(path, u, name, mode="a", dofs=None):
    """u in dolfin's layout, as dolfin's HDF5File.write writes it
    (femtools.dolfin_layout.write_function): dolfin reads it, on any number of ranks."""
    from femtools.dolfinx.checkpoint import write_function
    V = u.function_space
    write_function(path, u.vector().get_local(), V, unique_dofs(V), name, mode, dofs)


def write_functions_h5(path, functions, mode="w", dofs=None):
    """Several Functions, `(u, name)` each, in one file opened once: what
    write_function_h5 writes for each in turn, the first in `mode` and the rest appended
    (femtools.dolfin_layout.write_functions_of)."""
    from femtools.dolfinx.checkpoint import write_functions
    write_functions(path, [(u.vector().get_local(), u.function_space,
                            unique_dofs(u.function_space), name) for u, name in functions],
                    mode, dofs)


def read_function_h5(path, u, name):
    """u from the Function `name` in path, in dolfin's layout (written by dolfin or by
    write_function_h5, on any number of ranks)."""
    from femtools.dolfinx.checkpoint import read_function
    V = u.function_space
    u.vector().set_local(read_function(path, name.strip("/"), V, unique_dofs(V)))
    return u


def _into_components(components, read):
    """The vector Function `read(u)` reads (u on the vector space of the scalar
    Functions `components`, all on one space) into `components`."""
    u = AssignedVectorFunction(components)
    read(u)
    u.sync()
    for m, c in zip(u._maps, components):
        c.x.array[:] = u.x.array[m]
        c.restrict()


def read_vector_function_h5(path, name, components):
    """The vector Function `name` in path (dolfin's layout) into the scalar Functions
    `components` (all on one space, which gives the vector space its element and
    periodicity)."""
    _into_components(components, lambda u: read_function_h5(path, u, name))


def read_series_step(path, name, counter, u):
    """Step `counter` of field `name` of an XDMF time series (femtools.xdmf.XDMFFile2's
    layout, either backend's) into u (femtools.dolfinx.xdmf_read.read_nodal)."""
    from femtools.dolfinx.xdmf_read import read_nodal
    read_nodal(path, name, counter, u)
    return u.restrict()


def read_vector_series_step(path, name, counter, components):
    """Step `counter` of the vector field `name` of an XDMF time series into the scalar
    Functions `components`, all on one space."""
    _into_components(components, lambda u: read_series_step(path, name, counter, u))


def weighted_gradient_matrix(V, S, i):
    """The matrix of the derivative along x_i of a Function on S, onto V, as femtools
    .gradient makes it on dolfin: the cells' mean gradients averaged onto V's dofs, a
    dof's cells (those of its row in the DG0-to-V mass matrix) weighted by
    w_c^2 / sum(w), w_c = 1/|c|."""
    DG = FunctionSpace(V.mesh, "DG", 0)
    G = assemble(TrialFunction(DG) * TestFunction(V) * dx)._m
    w = assemble(TestFunction(DG) * dx)._v
    w.reciprocal()
    indptr, cols, _ = G.getValuesCSR()
    need = np.unique(cols).astype(PETSc.IntType)
    local = PETSc.Vec().createSeq(len(need), comm=PETSc.COMM_SELF)
    scatter = PETSc.Scatter().create(w, PETSc.IS().createGeneral(need, comm=PETSc.COMM_SELF),
                                     local, None)
    scatter.scatter(w, local, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    wc = local.getArray(readonly=True)[np.searchsorted(need, cols)]
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    total = np.bincount(rows, weights=wc, minlength=len(indptr) - 1)
    A = G.duplicate()
    A.setValuesCSR(indptr, cols, wc * wc / total[rows])
    A.assemble()
    dP = assemble(TrialFunction(S).dx(i) * TestFunction(DG) * dx)._m
    return Matrix(A.matMult(dP))


from femtools.backend import NAMES  # noqa: E402
__all__ = list(NAMES)
