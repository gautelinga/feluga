"""SHA-256 digests of what a dolfin-layout HDF5 mesh or Function *is*, read in blocks.

    mesh_digest("mesh.h5")                 # coordinates, topology, cell labels
    function_digest(h5["u"], comm)         # vector, dofmap, cell labels; on any rank count

Of the arrays, not the file's bytes (the same mesh written twice need not be
byte-identical) and not its path: each array's name, dtype and shape go in before its
data, and the data is read in blocks of rows, so no process holds the whole array. numpy
and h5py; mpi4py only where a communicator is passed.
"""
import hashlib

import numpy as np

MESH_DATASETS = ("mesh/coordinates", "mesh/topology", "mesh/cell_indices")
FUNCTION_DATASETS = ("vector_0", "cell_dofs", "x_cell_dofs", "cells")


def mesh_digest(path, block=1 << 20):
    """The hex SHA-256 of a dolfin HDF5 mesh's coordinates, topology and labelling
    (`MESH_DATASETS`, `cell_indices` where there is one), `block` rows at a time. A file
    without coordinates or topology is refused: every non-mesh file would otherwise hash
    alike, and match each other."""
    import h5py
    digest = hashlib.sha256()
    with h5py.File(path, "r") as h5:
        missing = [n for n in MESH_DATASETS[:2] if n not in h5]
        if missing:
            raise ValueError("{} is not a dolfin mesh: no {}".format(
                path, " or ".join(missing)))
        for name in MESH_DATASETS:
            if name not in h5:
                continue
            data = h5[name]
            digest.update("{} {} {}".format(name, data.dtype.str, data.shape).encode())
            for start in range(0, data.shape[0], block):
                digest.update(np.ascontiguousarray(data[start:start + block]).tobytes())
    return digest.hexdigest()


def function_digest(group, comm=None, block=1 << 22):
    """The hex SHA-256 of a stored Function, the h5py group `group` (dolfin's layout:
    `FUNCTION_DATASETS`, its vector, dofmap and row labels).

    Each dataset is cut into blocks of `block` rows, block j hashed by rank j % P, and the
    blocks' digests are hashed in order with each dataset's name (`<group>/<dataset>`),
    dtype and shape: the same value on any rank count, the read and the hashing spread
    over the ranks. The block size is part of the value, so a stored digest should name
    it. Collective; every rank gets the value."""
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    prefix = group.name.strip("/")
    jobs = [(name, start) for name in FUNCTION_DATASETS
            for start in range(0, group[name].shape[0], block)]
    mine = {j: hashlib.sha256(np.ascontiguousarray(group[name][start:start + block])
                              .tobytes()).digest()
            for j, (name, start) in enumerate(jobs) if j % size == rank}
    if size > 1:
        for part in comm.allgather(mine):
            mine.update(part)
    digest = hashlib.sha256()
    j = 0
    for name in FUNCTION_DATASETS:
        data = group[name]
        digest.update("{}/{} {} {}".format(prefix, name, data.dtype.str, data.shape).encode())
        while j < len(jobs) and jobs[j][0] == name:
            digest.update(mine[j])
            j += 1
    return digest.hexdigest()
