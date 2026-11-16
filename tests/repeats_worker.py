"""tests/test_compress.py's worker: write_ordered on 2 ranks where only rank 0's block gets
a repeated row (rank 1's block gets each row once): every rank must take the same
collectives (until 2026-11-15 rank 1 returned early and rank 0 waited in the repeats'
check). Prints "ok" from rank 0 when the file is the array."""
import os
import sys

import numpy as np
from mpi4py import MPI

from femtools import compress, hdf5_rows

comm = MPI.COMM_WORLD
n = 2 * compress.chunk_rows((1 << 21,))           # two chunks: a block a rank
half = n // 2
rows = np.r_[np.arange(half), 0] if comm.rank == 0 else np.arange(half, n)
path = os.path.join(sys.argv[1], "repeats.h5")
hdf5_rows.write_ordered(path, "x", rows, rows.astype(float), n, comm, mode="w")
if comm.rank == 0:
    import h5py
    with h5py.File(path, "r") as h5:
        print("ok" if np.array_equal(h5["x"][:], np.arange(n, dtype=float)) else "wrong",
              flush=True)
