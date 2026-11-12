"""Plaza refinement of a DOLFINx mesh with periodic faces bisected alike: the DOLFINx port
of `femtools.plaza_periodic`.

    fine, npair, parent = refine_periodic(mesh, cells, upper=(Lx, Ly), periodic=(True, False))

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
returned when they do. The C++ (femtools/dolfinx/cpp) is built on first use with CMake
against the installed DOLFINx and nanobind, into femtools' cache folder
(FEMTOOLS_JIT_DIR, else ~/.cache/femtools), rank 0 building while the others wait.
"""
import hashlib
import importlib.util
import os
import subprocess
import sys

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp")
_MODULE = [None]


class TooManyCells(Exception):
    """The refinement would make more cells than allowed (nothing was refined)."""

    def __init__(self, n_children, cap, n_rank, rank_cap):
        self.n_children, self.cap, self.n_rank, self.rank_cap = n_children, cap, n_rank, rank_cap
        super().__init__("refine_periodic: {} cells ({} on the most loaded rank), beyond the "
                         "limits {} and {}".format(n_children, n_rank, cap or "-", rank_cap or "-"))


def _module(comm):
    """The compiled extension, built once per source and installation."""
    if _MODULE[0] is not None:
        return _MODULE[0]
    import dolfinx
    sources = [os.path.join(HERE, f) for f in ("plaza_periodic.cpp", "CMakeLists.txt")]
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


def _cells(mesh, cells):
    """Local cell indices from indices or a boolean marker per local (owned) cell."""
    cells = np.asarray(cells)
    if cells.dtype == bool:
        cells = np.flatnonzero(cells)
    return np.ascontiguousarray(cells, dtype=np.int32)


def refine_periodic(mesh, cells=None, upper=None, periodic=None, lower=None, tol=1e-10,
                    redistribute=False, max_cells=None, max_rank_cells=None, frozen=None,
                    report=None):
    """`dolfinx.mesh.refine` of the cells `cells` (local indices, or a boolean per owned
    cell; None: every cell), with the faces of the periodic box identified along
    `periodic` (default every axis) bisected alike. `upper`, `lower`: the box's corners
    (default the mesh's bounding box); `tol`: how near a face a point must be.

    Returns `(new_mesh, npair, parent)`: the refined mesh, the number of periodic partner
    edges paired (summed over ranks, both ways) and the parent-cell map, each local cell
    of the new mesh to the local cell of `mesh` it was cut from (None when redistributed
    on several ranks; dolfinx.mesh.refine's own map is in the order the cells were made,
    not the new mesh's). `frozen`: local edges never to bisect;
    `max_cells`, `max_rank_cells`: raise TooManyCells rather than make more children
    than that, in all or on the most loaded rank. `report`, a dict, receives
    `unmarked` (marked cells left so by the freeze) and `frozen_cells`. Collective."""
    import ufl
    from dolfinx.mesh import Mesh
    from femtools.dolfinx.periodic import box_extent
    comm = mesh.comm
    tdim, gdim = mesh.topology.dim, mesh.geometry.dim
    lo, hi = box_extent(mesh)
    lo = lo if lower is None else np.r_[np.asarray(lower, float), np.zeros(3)][:3]
    hi = hi if upper is None else np.r_[np.asarray(upper, float), np.zeros(3)][:3]
    periodic = [True] * gdim if periodic is None else list(periodic)
    axes = [a for a, p in enumerate(periodic) if p]
    n_own = mesh.topology.index_map(tdim).size_local
    cells = np.arange(n_own, dtype=np.int32) if cells is None else _cells(mesh, cells)
    frozen = np.zeros(0, np.int32) if frozen is None else np.ascontiguousarray(frozen, np.int32)
    cap, rank_cap = int(max_cells or 0), int(max_rank_cells or 0)
    module = _module(comm)
    new, parent, npair, n_children, n_rank, n_unmarked, n_frozen = module.refine_periodic(
        mesh._cpp_object, cells, [float(v) for v in lo], [float(v) for v in hi], axes,
        float(tol), bool(redistribute), cap, rank_cap, frozen)
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
    return fine, int(npair), parent[made]
