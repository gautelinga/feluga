"""Worker for tests/test_dolfinx_unique.py: the preallocation of `create_matrix`, which
enumerates the dofs each integration entity couples, against the pattern of a plain
DOLFINx assembly folded onto unique dofs; (d_nnz, o_nnz) must be equal entry by entry.
And the values of `assemble_matrix`'s own inserter (every assembly after the first)
against those of MatSetValuesLocal (the first), and its refusal of a form whose values
fall outside the matrix's pattern.
Scalar P1 and P2, mixed P2xP2 and Taylor-Hood spaces, periodic in no axis, in x, in x
and y; cell, exterior-facet, interior-facet and cell-subdomain integrals; square and
rectangular forms; 2D and 3D; the pairs enumerated at once and in small batches.
Writes {case: mismatching rows over all ranks, plus 1 if the values differ} and
{"refused": whether every rank refused} as JSON (rank 0).

    [mpirun -n k] python3 tests/dolfinx_pattern_worker.py out.json
"""
import json
import sys

import numpy as np
import ufl
import basix.ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

from feluga.dolfinx import assembly
from feluga.dolfinx.periodic import periodic_box
from feluga.dolfinx.unique import UniqueDofs

comm = MPI.COMM_WORLD


def assembled_preallocation(a, rows, cols):
    """(d_nnz, o_nnz) of the folded matrix of `a` from the pattern of one plain DOLFINx
    assembly folded onto unique dofs. The fold is one to one except at periodic slaves,
    so only the entries in a slave's row or column, or in a column not known on this
    rank, can meet another: those alone are sent to the owner of their row and made
    unique there, with the other entries of the rows they reach. Collective."""
    from dolfinx.fem import petsc
    from petsc4py import PETSc
    from feluga.hdf5_rows import route, unique_rows

    comm = rows.comm
    A = petsc.create_matrix(a)
    petsc.assemble_matrix(A, a)  # values do not matter, the pattern does
    A.assemble()
    indptr, gcols = A.getValuesCSR()[:2]
    A.destroy()
    n = len(indptr) - 1                         # the space's owned dofs: the rows
    lrow = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    del indptr
    # An owned row may hold columns that are not local dofs here (entries another rank
    # assembled for its ghost row): their owners know them.
    lcol = cols.local_of_global(gcols)
    known = lcol >= 0
    ccol = np.empty(len(gcols), np.int64)
    ccol[known] = cols.compact[lcol[known]]
    ccol[~known] = cols.compact_of_global(gcols[~known])
    del gcols
    loose = ~known
    loose[known] = cols.is_slave[lcol[known]]
    del lcol, known
    loose |= rows.is_slave[lrow]
    clean = ~loose
    clo, chi = cols.owned_range
    start, n_owned = rows.start, rows.n_owned
    # the clean entries: unique, each in the owned row of a local row that is no slave
    own = ~rows.is_slave[:n]
    row_of = rows.compact[:n][own] - start
    diag = (ccol >= clo) & (ccol < chi)
    d_nnz = np.zeros(n_owned, np.int64)
    o_nnz = np.zeros(n_owned, np.int64)
    d_nnz[row_of] = np.bincount(lrow[clean & diag], minlength=n)[own]
    o_nnz[row_of] = np.bincount(lrow[clean & ~diag], minlength=n)[own]
    del diag
    pairs = np.stack([rows.compact[lrow[loose]], ccol[loose]], axis=1)
    if comm.size > 1:
        starts = np.array(comm.allgather(start), dtype=np.int64)
        owner = np.searchsorted(starts, pairs[:, 0], side="right") - 1
        pairs = route(comm, owner, pairs).received[0]
    if len(pairs):
        # the rows the loose entries reach, recounted with their clean entries
        touched = np.zeros(n_owned, bool)
        touched[pairs[:, 0] - start] = True
        reached = np.zeros(n, bool)
        reached[rows.owned_unique[touched]] = True
        also = clean & reached[lrow]
        pairs = unique_rows(np.concatenate(
            [pairs, np.stack([rows.compact[lrow[also]], ccol[also]], axis=1)]))
        r = pairs[:, 0] - start
        diag = (pairs[:, 1] >= clo) & (pairs[:, 1] < chi)
        d_nnz[touched] = 0
        o_nnz[touched] = 0
        d_nnz += np.bincount(r[diag], minlength=n_owned)
        o_nnz += np.bincount(r[~diag], minlength=n_owned)
    return d_nnz.astype(PETSc.IntType), o_nnz.astype(PETSc.IntType)


def spaces(msh):
    cell = msh.basix_cell()
    gdim = msh.geometry.dim
    P1 = basix.ufl.element("Lagrange", cell, 1)
    P2 = basix.ufl.element("Lagrange", cell, 2)
    P2v = basix.ufl.element("Lagrange", cell, 2, shape=(gdim,))
    return {"P1": fem.functionspace(msh, P1), "P2": fem.functionspace(msh, P2),
            "P2xP2": fem.functionspace(msh, basix.ufl.mixed_element([P2, P2])),
            "TH": fem.functionspace(msh, basix.ufl.mixed_element([P2v, P1]))}


def forms(msh, R, C):
    """{name: compiled bilinear form} with test space R, trial space C."""
    tdim = msh.topology.dim
    u, v = ufl.TrialFunction(C), ufl.TestFunction(R)
    cells = np.arange(msh.topology.index_map(tdim).size_local, dtype=np.int32)
    mid = mesh.compute_midpoints(msh, tdim, cells)
    half = mesh.meshtags(msh, tdim, cells, np.where(mid[:, 1] < 0.5, 1, 2).astype(np.int32))
    dxh = ufl.Measure("dx", domain=msh, subdomain_data=half)
    w = ufl.split(u)[0] if C.element.num_sub_elements else u
    q = ufl.split(v)[-1] if R.element.num_sub_elements else v
    if len(w.ufl_shape):
        w = w[0]
    out = {"all": fem.form(ufl.inner(u, v) * ufl.dx)} if R is C else {}
    return {**out, "cells": fem.form(w * q * ufl.dx),
            "ext_facets": fem.form(w * q * ufl.ds),
            "int_facets": fem.form(ufl.jump(w) * ufl.jump(q) * ufl.dS),
            "cells_jumps": fem.form(w * q * ufl.dx + ufl.jump(w) * ufl.jump(q) * ufl.dS),
            "subdomain": fem.form(w * q * dxh(1))}


def compare(a, rows, cols):
    """Rows whose counts differ, over all ranks: with the pairs enumerated all at once,
    and a few at a time (many batches, many row ranges)."""
    ref = assembled_preallocation(a, rows, cols)
    bad = 0
    for chunk in (assembly._CHUNK, 97):
        default, assembly._CHUNK = assembly._CHUNK, chunk
        try:
            new = assembly._preallocation(a, rows, cols)
        finally:
            assembly._CHUNK = default
        bad += int(np.count_nonzero((new[0] != ref[0]) | (new[1] != ref[1])))
    K = assembly.assemble_matrix(assembly.create_matrix(a, rows, cols), a)
    first = K.copy()
    assembly.assemble_matrix(K, a)
    K.axpy(-1, first)
    differ = K.norm(PETSc.NormType.NORM_INFINITY) > 1e-14 * max(
        first.norm(PETSc.NormType.NORM_INFINITY), 1)
    return comm.allreduce(bad) + int(differ)


def refused(msh, V):
    """Whether a form coupling across interior facets, assembled into the matrix of one
    coupling within cells only, is refused on every rank."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    U = UniqueDofs(V)
    K = assembly.assemble_matrix(assembly.create_matrix(fem.form(u * v * ufl.dx), U),
                                 fem.form(u * v * ufl.dx))
    try:
        assembly.assemble_matrix(K, fem.form(ufl.jump(u) * ufl.jump(v) * ufl.dS))
    except RuntimeError as e:
        return comm.allreduce("outside the matrix's pattern" in str(e), op=MPI.LAND)
    return False


res = {}
for dim, msh in ((2, mesh.create_rectangle(comm, [[0, 0], [1, 1]], [7, 9])),
                 (3, mesh.create_box(comm, [[0, 0, 0], [1, 1, 1]], [3, 4, 3]))):
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    S = spaces(msh)
    for per in ("none", "x", "xy"):
        periodic = {"none": None, "x": (True, False, False), "xy": (True, True, False)}[per]
        U = {k: UniqueDofs(V) if periodic is None
             else UniqueDofs(V, *periodic_box(V, periodic=periodic[:dim]))
             for k, V in S.items()}
        pairs = [(k, k) for k in S] + [("P2", "P1"), ("TH", "P1"), ("P1", "P2xP2")]
        for rk, ck in pairs:
            names = forms(msh, S[rk], S[ck])
            if rk != ck:  # rectangular: fewer integral kinds
                names = {k: names[k] for k in ("cells", "cells_jumps", "ext_facets")}
            for name, a in names.items():
                res["{}d/{}/{}x{}/{}".format(dim, per, rk, ck, name)] = compare(
                    a, U[rk], U[ck])

    if dim == 2:
        res_refused = refused(msh, S["P1"])

if comm.rank == 0:
    res["refused"] = res_refused
    with open(sys.argv[1], "w") as fh:
        json.dump(res, fh)
