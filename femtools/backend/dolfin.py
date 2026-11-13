"""The names of femtools.backend on legacy dolfin: dolfin's own, femtools' dolfin tools
(fast assembly, Dirichlet conditions and interpolation, mixed components, null spaces,
Krylov solvers, boundary markers, periodic boxes, dolfin's HDF5 layout), and the neutral
helpers that femtools.backend.dolfinx provides under the same names.
"""
import numpy as np
from dolfin import (  # noqa: F401
    as_backend_type, as_vector, BoxMesh, conditional, Constant, cos, DirichletBC, div,
    DOLFIN_EPS, DOLFIN_EPS_LARGE, dot, exp, Expression, FacetNormal, FiniteElement,
    Function, FunctionSpace, ge, grad, gt, IndexMap, inner, le, lhs, list_timings, lt,
    Matrix, Mesh, MixedElement, MPI, nabla_grad, norm, normalize, parameters,
    PETScKrylovSolver, Point, RectangleMesh, rhs, sin, solve, SpatialCoordinate, split,
    sqrt, SubDomain, TestFunction, TestFunctions, Timer, TimingClear, TimingType,
    TrialFunction, TrialFunctions, UnitSquareMesh, Vector, XDMFFile)

from femtools import ksp  # noqa: F401
from femtools.boundaries import BoundaryMarkers  # noqa: F401
from femtools.box import Faces  # noqa: F401
from femtools.box import mark_walls  # noqa: F401
from femtools.dolfin_layout import read_function as read_function_h5
from femtools.dolfin_layout import write_dofs as write_dof_tables  # noqa: F401
from femtools.dolfin_layout import write_function as write_function_h5  # noqa: F401
from femtools.dolfin_layout import write_mesh as write_mesh_h5  # noqa: F401
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


def set_default_mesh(mesh):
    """Nothing: dolfin's Constants and Expressions belong to no mesh."""


def mesh_id(mesh):
    """A key for the mesh, the same while it exists."""
    return mesh.id()


def space_id(V):
    """A key for space V, the same while it exists."""
    return V.id()


def function_space(u):
    return u.function_space()


def space_mesh(V):
    return V.mesh()


def mesh_comm(mesh):
    """The mesh's MPI communicator (mpi4py)."""
    return mesh.mpi_comm()


def geometric_dimension(mesh):
    return mesh.geometry().dim()


def element_degree(V):
    return V.ufl_element().degree()


def is_lagrange(V):
    """Whether V's element is scalar continuous Lagrange."""
    el = V.ufl_element()
    return el.family() in ("Lagrange", "Q") and el.value_shape() == ()


def owned_dofs(W, i):
    """The owned dofs of component i of the mixed space W, as global indices of its
    vectors."""
    from petsc4py import PETSc
    return np.asarray(W.sub(i).dofmap().dofs(), dtype=PETSc.IntType)


def fixed_dofs(W, bcs):
    """The owned dofs of space W that the Dirichlet conditions `bcs` constrain, as
    global indices of its vectors."""
    start = W.dofmap().ownership_range()[0]
    owned = W.dofmap().index_map().size(IndexMap.MapSize.OWNED)
    return np.array([start + k for bc in bcs for k in bc.get_boundary_values() if k < owned],
                    dtype=np.int64)


def sync(f):
    """f, after its `sync()` if it has one (a Function whose values follow another's);
    dolfin's Functions themselves have one layout and need none."""
    s = getattr(f, "sync", None)
    if s is not None:
        s()
    return f


def collect_garbage():
    """Destroy the PETSc objects Python has dropped, which petsc4py keeps until then
    (femtools.garbage). Collective: call it at the same point on every rank, such as
    once a time step."""
    from femtools.garbage import cleanup
    cleanup()


def assemble(form, tensor=None):
    """femtools.fast_assembly's assemble (compiled once per form), after bringing the
    form's Functions that follow another up to date (`sync`), as on DOLFINx."""
    for c in form.coefficients():
        sync(c)
    return _fast_assemble(form, tensor)


def interpolate(f, V, out=None):
    """f interpolated into V, into `out` if given (a Function on V): a Function by
    femtools.fast_interpolation (a transfer matrix built once), an Expression by
    dolfin."""
    if isinstance(f, Function):
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
    """The largest cell diameter over all ranks."""
    return mesh.hmax()


def read_mesh_file(fname, periodic=None):
    """The mesh stored as "mesh" in the HDF5 file fname.

    `periodic` gives the axes the problem identifies. Under MPI the mesh is then read
    in blocks and partitioned by femtools.periodic_partition, so that no process holds
    a vertex of a periodic face without knowing that it is periodic (dolfin's periodic
    dofmap would give that vertex a wrong dof). Without it, dolfin reads the mesh and
    `repair_partition` can fix the partition afterwards.
    """
    comm = MPI.comm_world
    if periodic is not None and comm.size > 1:
        from femtools.box import FACE_TOL
        from femtools.periodic_partition import read_mesh
        # L=None: the upper faces are at the largest coordinates, all the partition needs.
        return read_mesh(fname, None, periodic, comm, log=_root_print(comm), tol=FACE_TOL)
    from dolfin import HDF5File
    mesh = Mesh()
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
