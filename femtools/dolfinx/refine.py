"""Plaza refinement of a DOLFINx mesh with periodic faces bisected alike: the DOLFINx port
of `femtools.plaza_periodic` and `femtools.refine_periodic`, with their signatures.

    refine_periodic(mesh, markers, box, axes)    periodic Plaza (`plaza_periodic`'s)
    refine_congruent(mesh, markers, box, axes)   the same, then the congruence guard
    congruent(mesh, box, axes)                   whether identified faces carry the same nodes
    check_congruent(mesh, box, axes)             the same, or exit with the refusal
    compute_box(mesh)                            the box's upper corner, over the ranks
    TooManyCells                                 raised by max_cells / max_rank_cells

The same mesh refined here and by `femtools.plaza_periodic` on dolfin gives the same cells:
the tie rule, the 2D uniform-split test, the partner pairing and the mark exchange are one
C++ header (femtools/cpp/plaza_periodic_rules.h) compiled into both, and the congruence
check is `femtools.periodic_core`'s, as dolfin's is.

DOLFINx's `refine` knows nothing of periodic faces: refine near one face and not its
partner, and the slave dofs there have no master (and on tetrahedra two faces with the
same marks can still be cut differently, DOLFINx breaking ties between equally long edges
on global vertex indices, which the two faces do not share). This is DOLFINx's own
refinement -- its enforce_rules, compute_refinement and create_mesh, so it scales as
`dolfinx.mesh.refine` does -- with plaza_periodic's two changes: ties between longest
edges broken by position, the same on both faces of a pair, and the marking propagated
to a joint fixed point with the periodic partners. Also from plaza_periodic: `frozen`
edges never bisected, and `max_cells` / `max_rank_cells` refusing a refinement before it
is made (`TooManyCells`).

The children stay on their parent's rank (DOLFINx's identity partitioner) unless
`redistribute`; the parent-cell map (`parent`, new local cell -> old local cell) is
returned when they stay. The C++ (femtools/dolfinx/cpp) is built on first use with CMake
against the installed DOLFINx and nanobind, into femtools' cache folder
(FEMTOOLS_JIT_DIR, else ~/.cache/femtools), rank 0 building while the others wait.
"""
import hashlib
import importlib.util
import os
import subprocess
import sys

import numpy as np

from femtools.periodic_core import TooManyCells  # noqa: F401  (raised here)

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp")
_MODULE = [None]
TOL = 1e-9


def _module(comm):
    """The compiled extension, built once per source and installation."""
    if _MODULE[0] is not None:
        return _MODULE[0]
    import dolfinx
    sources = [os.path.join(HERE, f) for f in ("plaza_periodic.cpp", "CMakeLists.txt")]
    sources.append(os.path.join(HERE, "..", "..", "cpp", "plaza_periodic_rules.h"))
    h = hashlib.sha1()
    for path in sources:
        with open(path, "rb") as f:
            h.update(f.read())
    h.update((dolfinx.__version__ + sys.version).encode())
    root = os.environ.get("FEMTOOLS_JIT_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "femtools")
    build = os.path.join(root, "plaza_periodic_x_" + h.hexdigest()[:16])
    error = None
    if comm.rank == 0 and not os.path.exists(os.path.join(build, "done")):
        os.makedirs(build, exist_ok=True)
        steps = [["cmake", "-S", HERE, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                  "-DPython_EXECUTABLE=" + sys.executable],
                 ["cmake", "--build", build, "-j", str(min(8, os.cpu_count() or 1))]]
        for step in steps:
            proc = subprocess.run(step, capture_output=True, text=True)
            if proc.returncode:
                error = (proc.stdout + proc.stderr)[-4000:]
                break
        else:
            open(os.path.join(build, "done"), "w").close()
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError("femtools.dolfinx.refine: building the extension failed:\n" + error)
    (so,) = [f for f in os.listdir(build) if f.startswith("plaza_periodic_x") and f.endswith(".so")]
    spec = importlib.util.spec_from_file_location("plaza_periodic_x", os.path.join(build, so))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE[0] = module
    return module


def _cells(mesh, markers):
    """Local cell indices from None (every owned cell), indices, or a boolean per owned
    cell."""
    tdim = mesh.topology.dim
    if markers is None:
        return np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    markers = np.asarray(markers)
    if markers.dtype == bool:
        markers = np.flatnonzero(markers)
    return np.ascontiguousarray(markers, dtype=np.int32)


def _lower(mesh, axes, lower):
    from femtools.dolfinx.periodic import box_extent
    from femtools.periodic_core import lower_corner
    return lower_corner(None if lower is not None else box_extent(mesh), axes, lower)


def _say(mesh, text):
    if mesh.comm.rank == 0:
        print(text, flush=True)


def refine_periodic(mesh, markers, box, axes, redistribute=True, tol=1e-10,
                    max_cells=None, max_rank_cells=None, frozen=None, report=None,
                    lower=None):
    """`dolfinx.mesh.refine` of the marked cells, with the identified faces bisected
    alike: `femtools.plaza_periodic.refine_periodic`'s signature and cells.

    `markers`: None (every cell), local cell indices, or a boolean per owned cell.
    `box` is the periodic box's upper corner and `lower` its lower one (the origin when
    None, which a mesh elsewhere is refused), `axes` the periodic axes; `tol` how near a
    face a point must be. 2D and 3D, any rank count.

    Returns `(new_mesh, npair, parent_cell)`: the refined mesh, the number of periodic
    partner edges paired (summed over ranks, both ways) and the parent-cell map, each
    local cell of the new mesh to the local cell of `mesh` it was cut from -- None when
    `redistribute` on several ranks (the children then leave their parents' ranks).
    `frozen`: local edges never to bisect; `max_cells`, `max_rank_cells`: raise
    TooManyCells rather than make more children than that, in all or on the most loaded
    rank. `report`, a dict, receives `unmarked` (marked cells left so by the freeze) and
    `frozen_cells`. Collective."""
    import ufl
    from dolfinx.mesh import Mesh
    from femtools.periodic_core import xyz
    comm = mesh.comm
    tdim = mesh.topology.dim
    cap, rank_cap = int(max_cells or 0), int(max_rank_cells or 0)
    if cap < 0 or rank_cap < 0:
        raise ValueError("max_cells and max_rank_cells must be positive, got {} and {}"
                         .format(max_cells, max_rank_cells))
    lo = _lower(mesh, axes, lower)
    hi = xyz(np.asarray(box, float))
    cells = _cells(mesh, markers)
    frozen = (np.zeros(0, np.int32) if frozen is None
              else np.ascontiguousarray(np.asarray(frozen).ravel(), np.int32))
    module = _module(comm)
    new, parent, npair, n_children, n_rank, n_unmarked, n_frozen = module.refine_periodic(
        mesh._cpp_object, cells, [float(v) for v in lo], [float(v) for v in hi],
        [int(a) for a in axes], float(tol), bool(redistribute), cap, rank_cap, frozen)
    if report is not None:
        report.update(unmarked=int(n_unmarked), frozen_cells=int(n_frozen))
    if new is None:
        raise TooManyCells(n_children, cap, n_rank, rank_cap)
    fine = Mesh(new, ufl.Mesh(mesh.ufl_domain().ufl_coordinate_element()))
    if redistribute and comm.size > 1:
        return fine, int(npair), None
    # DOLFINx's parent map is in the order the children were made, which create_mesh
    # reorders: a child's place in it is its original cell index, less the cells made on
    # the ranks before (the children stay on their parent's rank).
    parent = np.asarray(parent)
    first = comm.exscan(len(parent)) or 0
    n_new = fine.topology.index_map(tdim).size_local
    made = np.asarray(fine.topology.original_cell_index[:n_new], dtype=np.int64) - first
    return fine, int(npair), parent[made].astype(np.int64)


def _vertices_and_edges(mesh, edges=True):
    """Every local vertex's position (owned and ghost, topology numbering) and the local
    edges as vertex pairs."""
    import dolfinx.mesh as dm
    topology = mesh.topology
    nv = topology.index_map(0).size_local + topology.index_map(0).num_ghosts
    topology.create_connectivity(0, topology.dim)
    nodes = dm.entities_to_geometry(mesh, 0, np.arange(nv, dtype=np.int32)).reshape(-1)
    x = mesh.geometry.x[nodes]
    if not edges:
        return x, None
    topology.create_entities(1)
    topology.create_connectivity(1, 0)
    return x, topology.connectivity(1, 0).array.reshape(-1, 2)


def congruent(mesh, box, axes, exempt=None, lower=None):
    """Whether the identified faces still carry the same nodes, edges included
    (`femtools.periodic_core.faces_congruent_arrays`, which dolfin's check is). `exempt`
    is a bool per local vertex then per local edge (DOLFINx's numbering), or an (m, 3)
    array of positions, of nodes allowed no partner; `lower` the box's lower corner."""
    from femtools.periodic_core import faces_congruent_arrays
    lower = _lower(mesh, axes, lower)
    x, e = _vertices_and_edges(mesh)
    ok, _, _, _ = faces_congruent_arrays(mesh.comm, x, e, box, axes, TOL, exempt, lower)
    return ok


def check_congruent(mesh, box, axes, exempt=None, lower=None):
    """`congruent`, or exit with the refinement's refusal."""
    if not congruent(mesh, box, axes, exempt, lower):
        raise SystemExit(
            "the identified faces came out of the refinement not matching. Do "
            "not solve on this mesh: a periodic dof with no partner is left "
            "without its constraint.")


def compute_box(mesh):
    """The periodic box's upper corner off the mesh itself, over the mesh's ranks."""
    from femtools.dolfinx.periodic import box_extent
    return box_extent(mesh)[1][:mesh.geometry.dim]


def refine_congruent(mesh, cell_markers, box, axes, redistribute=False, max_cells=None,
                     exempt=None, max_rank_cells=None, frozen=None, report=None,
                     deferred=False, lower=None):
    """`femtools.refine_periodic.refine_congruent` on DOLFINx: the periodic refinement,
    with its output checked before it is trusted (unless `deferred`, when the caller
    runs `check_congruent` itself). Returns `(new_mesh, npair, parent_cell)`. `exempt`:
    a callable `(new_mesh, parent_cell) ->` the nodes allowed no partner. Without
    periodic axes, plain `dolfinx.mesh.refine` (which can neither count nor freeze)."""
    _say(mesh, "refining")
    if axes:
        new_mesh, npair, parent_cell = refine_periodic(
            mesh, cell_markers, box, axes, redistribute=redistribute, max_cells=max_cells,
            max_rank_cells=max_rank_cells, frozen=frozen, report=report, lower=lower)
        _say(mesh, "periodic Plaza: {} identified edge pairs".format(npair))
        if deferred:
            if exempt is not None:
                raise ValueError("refine_congruent: a deferred guard takes its exempt "
                                 "nodes at check_congruent")
        else:
            check_congruent(new_mesh, box, axes,
                            None if exempt is None else exempt(new_mesh, parent_cell),
                            lower)
        return new_mesh, npair, parent_cell
    if max_cells or max_rank_cells or frozen is not None:
        raise ValueError("max_cells and frozen need periodic axes: dolfinx.mesh.refine "
                         "can neither count nor freeze before it refines")
    import dolfinx.mesh as dm
    cells = _cells(mesh, cell_markers)
    mesh.topology.create_entities(1)
    edges = np.unique(dm.compute_incident_entities(mesh.topology, cells, mesh.topology.dim, 1))
    new_mesh, _, _ = dm.refine(mesh, edges)
    return new_mesh, 0, None
