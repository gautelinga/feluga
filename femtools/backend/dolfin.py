"""The names of femtools.backend on legacy dolfin: dolfin's own, femtools' dolfin tools
(fast assembly, Dirichlet conditions and interpolation, mixed components, null spaces,
Krylov solvers, boundary markers, periodic boxes, dolfin's HDF5 layout), and the neutral
helpers that femtools.backend.dolfinx provides under the same names.
"""
import numpy as np

from femtools.hdf5_rows import sorted_unique
from dolfin import (  # noqa: F401
    as_backend_type, as_vector, BoxMesh, conditional, Constant, cos, DirichletBC, div,
    DOLFIN_EPS, DOLFIN_EPS_LARGE, dot, exp, Expression, FacetNormal, FiniteElement,
    VectorElement,
    Function, FunctionSpace, ge, grad, gt, IndexMap, inner, le, lhs, list_timings, lt,
    Matrix, Mesh, MixedElement, MPI, nabla_grad, norm, normalize, parameters,
    PETScKrylovSolver, Point, RectangleMesh, rhs, sin, solve, SpatialCoordinate, split,
    sqrt, SubDomain, TestFunction, TestFunctions, Timer, TimingClear, TimingType,
    TrialFunction, TrialFunctions, UnitSquareMesh, Vector)

from femtools import ksp  # noqa: F401
from femtools.boundaries import BoundaryMarkers  # noqa: F401
from femtools.box import Faces  # noqa: F401
from femtools.box import mark_walls  # noqa: F401
from femtools.dolfin_layout import read_function as read_function_h5
from femtools.dolfin_layout import write_dofs as write_dof_tables  # noqa: F401
from femtools.dolfin_layout import write_function as write_function_h5  # noqa: F401
from femtools.dolfin_layout import write_functions as write_functions_h5  # noqa: F401
from femtools.dolfin_layout import write_mesh as write_mesh_h5  # noqa: F401
from femtools.couplings import form_couplings  # noqa: F401
from femtools.fast_assembly import assemble as _fast_assemble
from femtools.fast_assembly import assemble_matrix  # noqa: F401
from femtools.fast_bc import apply as apply_bc  # noqa: F401
from femtools.fast_interpolation import interpolate as _fast_interpolate
from femtools.fast_interpolation import values as nodal_values  # noqa: F401
from femtools.mixed import (  # noqa: F401
    AssignedVectorFunction, set_subfunction, subfunction, subspace)
from femtools.nullspace import constant_nullspace  # noqa: F401
from femtools.periodic import extent  # noqa: F401
# Periodic boundary conditions on a box, from each periodic axis's (lower, upper)
# bounds, e.g. PeriodicBox({0: (0, Lx)}) for periodicity in x.
from femtools.periodic import periodic_box_from_bounds as PeriodicBox  # noqa: F401
from femtools.xdmf import Unsupported, XDMFFile2

# The UFL dolfin was built with: ufl_legacy in recent installations, ufl in older ones.
try:
    import ufl_legacy as ufl
    if not isinstance(Constant(0.), ufl.Coefficient):
        raise ImportError
except ImportError:
    import ufl

name = "dolfin"
Coefficient = ufl.Coefficient
Measure = ufl.Measure
dx = ufl.dx


class XDMFFile(XDMFFile2):
    """femtools.xdmf.XDMFFile2, dolfin's XDMFFile with a P2 field's midside nodes kept and
    several fields on one grid (its docstring lists its deliberate differences); a
    Function it does not lay out (DG, P3, a whole mixed Function: femtools.xdmf
    .Unsupported) is written by dolfin's own XDMFFile, as dolfin writes it, when it is the
    file's first write: the file is then dolfin's, and every later write goes to dolfin's
    writer too. After a grid of XDMFFile2's, such a Function is refused (one file, one
    layout); so is a checkpoint in dolfin's file."""

    _dolfin = None          # dolfin's XDMFFile, once a write needed it

    def write(self, obj, *args, time=None, encoding=None):
        import dolfin as df
        if self._dolfin is None:
            fresh = self._doc is None
            try:
                return super().write(obj, *args, time=time, encoding=encoding)
            except Unsupported as e:
                if not fresh:
                    raise ValueError("{}: {}, and this file holds XDMFFile2's grids; write "
                                     "it to a file of its own".format(self.path, e)) from e
                if not isinstance(obj, df.Function):
                    raise
                self._doc = None        # made by the refused write, holding nothing
                self._dolfin = df.XDMFFile(self._mpi_comm(), self.path)
                for key, value in self.parameters.items():
                    self._dolfin.parameters[key] = value
        from femtools.xdmf import Encoding, _merge_write_arg, _split_write_args
        t, e = _split_write_args(args)
        time = _merge_write_arg(time, t, "time")
        Encoding.of(_merge_write_arg(encoding, e, "encoding"))    # HDF5 alone, as XDMFFile2
        return self._dolfin.write(obj) if time is None else self._dolfin.write(obj, time)

    def write_checkpoint(self, *args, **kwargs):
        if self._dolfin is not None:
            raise RuntimeError("{} already holds dolfin's visualisation grids, and a "
                               "checkpoint has a file layout of its own; use another file"
                               .format(self.path))
        return super().write_checkpoint(*args, **kwargs)

    def close(self, compress_checkpoint=True):
        if self._dolfin is not None:
            self._dolfin.close()
            self._dolfin = None
        return super().close(compress_checkpoint)


def set_default_mesh(mesh):
    """Nothing: dolfin's Constants and Expressions belong to no mesh."""


def constant_on(value, mesh):
    """A Constant for forms on `mesh`: dolfin's belong to no mesh."""
    return Constant(value)


def mesh_id(mesh):
    """A key for the mesh, the same while it exists."""
    return mesh.id()


def space_id(V):
    """A key for space V, the same while it exists."""
    return V.id()


def function_space(u):
    """The space of Function `u`."""
    return u.function_space()


def space_mesh(V):
    """The mesh of space `V`."""
    return V.mesh()


def Probes(x, V):
    """femtools.probes.Probes(x, V): V's Functions at the points x (flat, the same on
    every rank), evaluated many times (built at the first use, collectively)."""
    from femtools import probes
    return probes.Probes(x, V)


def GradProbes(x, V):
    """femtools.probes.GradProbes(x, V): Probes that take the gradient too."""
    from femtools import probes
    return probes.GradProbes(x, V)


def cell_values(f):
    """The DG0 Function f's values on the owned cells, in local cell order: one per cell
    for a scalar field, (cells, components) for a vector one."""
    V = f.function_space()
    e = V.ufl_element()
    if e.family() not in ("Discontinuous Lagrange", "DG") or e.degree() != 0:
        raise ValueError("cell_values: a DG0 Function, not {}".format(e))
    mesh = V.mesh()
    tdim = mesh.topology().dim()
    n_own = mesh.topology().ghost_offset(tdim)
    bs = max(int(np.prod(e.value_shape())), 1)
    dofs = np.asarray(V.dofmap().entity_dofs(mesh, tdim), dtype=np.int64)
    values = f.vector().get_local()[dofs.reshape(-1, bs)[:n_own]]
    return values[:, 0] if bs == 1 else values


def mesh_comm(mesh):
    """The mesh's MPI communicator (mpi4py)."""
    return mesh.mpi_comm()


def geometric_dimension(mesh):
    """The dimension of the mesh's coordinates."""
    return mesh.geometry().dim()


def element_degree(V):
    """The polynomial degree of space `V`'s element."""
    return V.ufl_element().degree()


def is_lagrange(V):
    """Whether V's element is scalar continuous Lagrange."""
    el = V.ufl_element()
    return el.family() in ("Lagrange", "Q") and el.value_shape() == ()


def owned_dofs(W, i):
    """The owned dofs of component i of the mixed space W (an index, or a path: (0, 1)
    is W.sub(0).sub(1)), as global indices of its vectors."""
    from petsc4py import PETSc
    from femtools.couplings import sub_at
    return np.asarray(sub_at(W, i).dofmap().dofs(), dtype=PETSc.IntType)


def fixed_dofs(W, bcs):
    """The owned dofs of space W that the Dirichlet conditions `bcs` constrain, as
    global indices of its vectors, sorted, each once, PETSc's integer type (as on
    DOLFINx): every one, also those whose boundary facets another rank holds (gathered,
    as dolfin's `DirichletBC.apply` gathers them; until 2026-11-15 a rank's own facets'
    alone, condition after condition). Collective."""
    from petsc4py import PETSc
    from femtools.couplings import dolfin_fixed_owned
    start = W.dofmap().ownership_range()[0]
    found = [np.zeros(0, np.int64)]
    for bc in bcs:
        found.append(start + dolfin_fixed_owned(W, bc))
    return sorted_unique(np.concatenate(found)).astype(PETSc.IntType)


def sync(f):
    """f, its ghost values brought up to date: after its `sync()` if it has one (a Function
    whose values follow another's), else its vector's ghost update. A vector written
    through PETSc (a KSP solve into `as_backend_type(f.vector()).vec()`) holds new owned
    values and stale ghosts, which assembly reads; DOLFINx's backend tracks that itself, so
    `sync(f)` after such a write is the neutral spelling. Collective."""
    s = getattr(f, "sync", None)
    if s is not None:
        s()
    elif callable(getattr(f, "vector", None)):
        as_backend_type(f.vector()).update_ghost_values()
    return f


def collect_garbage():
    """Destroy the PETSc objects Python has dropped, which petsc4py keeps until then
    (femtools.garbage). Collective: call it at the same point on every rank, such as
    once a time step."""
    from femtools.garbage import cleanup
    cleanup()


def assemble(form, tensor=None, couplings=None, bcs=()):
    """femtools.fast_assembly's assemble (compiled once per form), after bringing the
    form's Functions that follow another up to date (`sync`), as on DOLFINx. A matrix made
    here with `couplings` (None, "form" or a mask: femtools.couplings) stores only the
    blocks of the coupled components; a nonzero outside them is PETSc's error. With
    Dirichlet conditions `bcs` (a bilinear form, square) the matrix holds only the
    diagonal in their dofs' rows and columns, and is assembled with them applied
    symmetrically (dolfin's SystemAssembler, a zero right-hand side: the diagonal the
    number of cells holding the dof), its forms compiled once."""
    for c in form.coefficients():
        sync(c)
    if tensor is None and (couplings is not None or bcs):
        from femtools.couplings import dolfin_matrix, resolve, ones
        mask = resolve(couplings, form)
        tensor = dolfin_matrix(form, ones(form) if mask is None else mask, bcs)
    elif tensor is not None:
        if couplings is not None or bcs:
            raise ValueError("assemble: couplings and bcs are the matrix's, given when it is "
                             "made; assemble into it without them")
        # a matrix made with conditions is filled with them (assemble_system): without,
        # their rows would take the element entries, which its pattern refuses
        bcs = _matrix_bcs(tensor)
    if bcs:
        _system(form, tensor, bcs).assemble(tensor, _SYSTEMS[id(tensor)][4])
        return tensor
    return _fast_assemble(form, tensor)


# dolfin's SystemAssembler of a matrix with conditions, made once per (matrix, form): the
# form and a zero right-hand side compiled once (femtools.fast_assembly), as the rest of
# assemble is -- assemble_system would build and look them up again at every call
_SYSTEMS = None


def _system(form, tensor, bcs):
    """The SystemAssembler filling `tensor` with `form` under `bcs` (kept, with its
    right-hand side's vector, while among the most recently used)."""
    global _SYSTEMS
    import dolfin as df
    from femtools.fast_assembly import compiled
    from femtools.lru import LRU
    if _SYSTEMS is None:
        _SYSTEMS = LRU(64)
    held = _SYSTEMS.get(id(tensor))
    if held is None or held[0] is not tensor or held[1] is not form:
        v = sorted(form.arguments(), key=lambda x: x.number())[0]
        zero = df.Constant(np.zeros(v.ufl_shape)) if v.ufl_shape else df.Constant(0.0)
        L = df.inner(zero, v) * df.dx(domain=v.ufl_domain())
        assembler = df.cpp.fem.SystemAssembler(compiled(form), compiled(L), list(bcs))
        held = (tensor, form, L, assembler, df.PETScVector(tensor.mpi_comm()))
        _SYSTEMS[id(tensor)] = held
    return held[3]


def _matrix_bcs(tensor):
    """The Dirichlet conditions a couplings matrix was made with (femtools.couplings), or
    ()."""
    mat = getattr(tensor, "mat", None)
    return () if mat is None or not hasattr(tensor, "mpi_comm") else \
        (mat().getAttr("femtools_bcs") or ())


def interpolate(f, V, out=None):
    """f interpolated into V, into `out` if given (a Function on V): a Function by
    femtools.fast_interpolation (a transfer matrix built once), a Function on another
    mesh by dolfin's LagrangeInterpolator (V's interpolation points must lie in that
    mesh), an Expression by dolfin. Collective."""
    if isinstance(f, Function):
        if f.function_space().mesh().id() != V.mesh().id():
            from dolfin import LagrangeInterpolator
            out = Function(V) if out is None else out
            out.vector().zero()
            LagrangeInterpolator.interpolate(out, f)
            return out
        return _fast_interpolate(f, V, out=out)
    from dolfin import interpolate as dolfin_interpolate
    g = dolfin_interpolate(f, V)
    if out is None:
        return g
    out.vector().zero()
    out.vector().axpy(1., g.vector())
    return out


def mesh_coordinates(mesh):
    """The mesh's vertex coordinates, writable."""
    return mesh.coordinates()


def hmax(mesh):
    """The largest cell diameter over all ranks (dolfin's mesh.hmax() is this rank's).
    Collective."""
    from mpi4py import MPI
    return mesh.mpi_comm().allreduce(mesh.hmax() if mesh.num_cells() else 0.0, op=MPI.MAX)


def hmin(mesh):
    """The smallest cell diameter over all ranks (dolfin's mesh.hmin() is this rank's).
    Collective."""
    from mpi4py import MPI
    return mesh.mpi_comm().allreduce(mesh.hmin() if mesh.num_cells() else float("inf"),
                                     op=MPI.MIN)


def global_dim(V):
    """The space's dofs over all ranks (V.dim()); a component's alone, W.sub(i) or
    W.sub(i).sub(j). Of a collapsed space (`subspace`), its dofs without the periodic
    constraint, where DOLFINx's collapsed spaces keep it. Collective on DOLFINx."""
    return V.dim()


def global_num_cells(mesh):
    """The mesh's cells over all ranks."""
    return mesh.num_entities_global(mesh.topology().dim())


def global_num_vertices(mesh):
    """The mesh's vertices over all ranks."""
    return mesh.num_entities_global(0)


def marker_counts(markers):
    """The number of exterior facets carrying each marker of `markers` (a facet
    MeshFunction, BoundaryMarkers) over all ranks, as an array indexed by marker
    (`np.bincount`, as femtools.boundaries.written_marker_counts counts a file's). dolfin's
    "on_boundary" decides which facets are exterior; a mesh with ghost cells is refused,
    and so is a negative marker (a size_t of 2**63 or more, a wrapped -1) on any rank, on
    every rank. Collective."""
    from dolfin import CompiledSubDomain, MeshFunction
    from femtools.boundaries import marker_bincount
    mesh = markers.mesh()
    tdim = mesh.topology().dim()
    comm = mesh.mpi_comm()
    if comm.allreduce(int(mesh.topology().ghost_offset(tdim) < mesh.num_cells())):
        raise RuntimeError("marker_counts: a mesh with ghost cells")
    exterior = MeshFunction("size_t", mesh, tdim - 1, 0)
    CompiledSubDomain("on_boundary").mark(exterior, 1)
    return marker_bincount(markers.array()[exterior.array() == 1], comm)


def refine_cells(mesh, markers):
    """The mesh with the cells marked (a boolean per owned cell, or per owned then ghost
    cell) refined: Plaza refinement (femtools.plaza_periodic, no periodic axes), whose
    ties between equally long edges are broken by position, so that DOLFINx's
    refine_cells makes the same cells. Markers of another length are refused on every
    rank, as on DOLFINx. Collective."""
    import dolfin as df
    from femtools.hdf5_rows import agreed
    from femtools.plaza_periodic import refine_periodic
    tdim = mesh.topology().dim()
    n_own, n_all = mesh.topology().ghost_offset(tdim), mesh.num_cells()
    markers = np.asarray(markers, dtype=bool).ravel()
    if agreed(mesh.mpi_comm(), len(markers) not in (n_own, n_all)):
        raise ValueError("refine_cells: boolean markers need one entry a local cell "
                         "(owned, or owned then ghost)")
    flags = df.MeshFunction("bool", mesh, tdim, False)
    flags.array()[:n_own] = markers[:n_own]
    lower, upper = extent(mesh)
    return refine_periodic(mesh, flags, list(upper), [], lower=list(lower))[0]


def read_mesh_file(fname, periodic=None, tol=None):
    """The mesh in the file fname: "mesh" in an HDF5 file (dolfin's layout), or the mesh
    of an XDMF file (`*.xdmf`, the meshers').

    `periodic` gives the axes the problem identifies. Under MPI the mesh is then read
    in blocks and partitioned by femtools.periodic_partition, so that no process holds
    a vertex of a periodic face without knowing that it is periodic (dolfin's periodic
    dofmap would give that vertex a wrong dof); `tol`, how near a face a vertex is on it
    (absolute; default femtools.box.FACE_TOL), as the PeriodicBox's. Without `periodic`,
    dolfin reads the mesh and `repair_partition` can fix the partition afterwards.
    """
    comm = MPI.comm_world
    if periodic is not None and comm.size > 1:
        from femtools.box import FACE_TOL
        from femtools.periodic_partition import read_mesh
        # L=None: the upper faces are at the largest coordinates, all the partition needs.
        return read_mesh(fname, None, periodic, comm, log=_root_print(comm),
                         tol=FACE_TOL if tol is None else tol)
    mesh = Mesh()
    if fname.endswith(".xdmf"):
        from dolfin import XDMFFile
        with XDMFFile(mesh.mpi_comm(), fname) as f:
            f.read(mesh)
        return mesh
    from dolfin import HDF5File
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    return mesh


def _root_print(comm):
    def log(s):
        if comm.rank == 0:
            print(s)
    return log


def repair_partition(mesh, box, log=None):
    """The mesh, its cells moved where dolfin's partition left a process holding a
    vertex of a periodic face of `box` (a PeriodicBox) without knowing that it is
    periodic; the periodic dofmap would give that vertex a wrong dof. Most partitions
    need no move, and the mesh is then returned as it is. Collective."""
    comm = mesh.mpi_comm()
    if comm.size == 1 or not hasattr(box, "upper"):
        return mesh
    from femtools.periodic_partition import repaired
    return repaired(mesh, box.upper, box.periodic, comm, log=log or _root_print(comm),
                    tol=getattr(box, "tol", None))


def check_periodic_pairing(box, spaces):
    """Raise femtools.periodic_partition.PairingError if dolfin's periodic dofmap gave
    a node on a periodic face of `box` (a PeriodicBox) a dof of its own instead of its
    partner's, in any of `spaces` of degree 1 or 2. Collective."""
    mesh = spaces[0].mesh()
    comm = mesh.mpi_comm()
    if comm.size == 1 or not hasattr(box, "upper"):
        return
    from femtools.periodic_partition import check_space_pairing
    for V in spaces:
        if V.ufl_element().degree() <= 2:
            check_space_pairing(V, box.upper, box.periodic, comm, what=str(V.ufl_element()),
                                lower=box.lower, tol=getattr(box, "tol", None))


def weighted_gradient_matrix(V, S, i):
    """The matrix of the derivative along x_i of a Function on S, onto V: the cells'
    mean gradients averaged onto V's dofs (femtools.gradient)."""
    from femtools.gradient import compiled_gradient_module
    DG = FunctionSpace(V.mesh(), "DG", 0)
    G = assemble(TrialFunction(DG) * TestFunction(V) * dx())
    dP = assemble(TrialFunction(S).dx(i) * TestFunction(DG) * dx())
    return compiled_gradient_module.compute_weighted_gradient_matrix(G, dP, Function(DG))


def TimeSeriesFile(comm, path):
    """A field's time series, `write(function, t)` and `close()`: XDMF that keeps the
    midside nodes of a P2 field, compressed (femtools.xdmf.XDMFFile2), written
    incrementally."""
    from femtools.xdmf import XDMFFile2
    f = XDMFFile2(comm, path)
    f.parameters["rewrite_function_mesh"] = False
    f.parameters["flush_output"] = True
    return f


def _into_components(components, read):
    """The vector Function `read(uv)` reads into the scalar Functions `components` (all
    on one space, which gives the vector space its element and periodicity)."""
    from dolfin import FunctionAssigner, VectorFunctionSpace
    V = components[0].function_space()
    el = V.ufl_element()
    uv = Function(VectorFunctionSpace(V.mesh(), el.family(), el.degree(),
                                      constrained_domain=V.dofmap().constrained_domain))
    read(uv)
    FunctionAssigner([V] * len(components), uv.function_space()).assign(list(components), uv)


def read_vector_function_h5(path, name, components):
    """The vector Function `name` in path (dolfin's HDF5 layout) into the scalar
    Functions `components`, all on one space."""
    _into_components(components, lambda uv: read_function_h5(path, uv, name))


def read_series_step(path, name, counter, u):
    """Step `counter` of field `name` of an XDMF time series (femtools.xdmf.XDMFFile2's
    layout, written by either backend) into u."""
    from femtools.xdmf import XDMFFile2
    with XDMFFile2(u.function_space().mesh().mpi_comm(), path) as xdmf:
        xdmf.read_checkpoint(u, name, counter)
    return u


def read_vector_series_step(path, name, counter, components):
    """Step `counter` of the vector field `name` of an XDMF time series into the scalar
    Functions `components`, all on one space."""
    _into_components(components, lambda uv: read_series_step(path, name, counter, uv))


from femtools.backend import NAMES  # noqa: E402
__all__ = list(NAMES)
