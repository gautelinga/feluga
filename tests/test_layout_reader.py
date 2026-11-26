"""`feluga.layout_reader`: a mesh and functions dolfin wrote, read with numpy and h5py.

    python3 -m pytest tests/test_layout_reader.py

On a unit box written by dolfin's `HDF5File` on one rank and on three (rows in each
rank's order, so neither file is in global cell order): a quadratic on P2 comes back at
every P2 node, midsides included, to rounding, and a linear on P1 at the vertices
(`read_p1`) and at the midsides (`read`); the streamed reader (`label_volumes`,
`cell_blocks`) is the whole one with the function file's rows permuted and blocks smaller
than the mesh; a file labelling a cell twice, or one the mesh lacks, is refused, and a
dofmap read wrongly (two node columns swapped) is caught by the shared-node check; a
field holding its vertex values (`VERTEX_VALUES`) is returned as it is.
"""
import os
import shutil
import sys

import numpy as np
import pytest

from tests import support

h5py = pytest.importorskip("h5py")

from feluga.layout_reader import LayoutReader  # noqa: E402


U = ("0.1 + 0.5*x[0] - 0.25*x[1] + x[0]*x[0] - 0.5*x[1]*x[2]",
     "-0.2 + x[2] + 0.75*x[1]*x[1] - 0.3*x[0]*x[1]",
     "1.0 + 0.25*x[0] - 0.5*x[2] + 0.4*x[2]*x[2] + 0.2*x[0]*x[2]")
P = "2.0 - x[0] + 0.5*x[1] + 0.25*x[2]"

WRITER = r"""
import sys
import dolfin as df
df.set_log_level(40)
U, P, out = eval(sys.argv[1]), sys.argv[2], sys.argv[3]
comm = df.MPI.comm_world
mesh = df.BoxMesh(comm, df.Point(0, 0, 0), df.Point(1, 1, 2), 3, 3, 5)
u = df.interpolate(df.Expression(U, degree=2), df.VectorFunctionSpace(mesh, "Lagrange", 2))
p = df.interpolate(df.Expression(P, degree=1), df.FunctionSpace(mesh, "Lagrange", 1))
with df.HDF5File(comm, out + "/mesh.h5", "w") as h5:
    h5.write(mesh, "mesh")
with df.HDF5File(comm, out + "/up.h5", "w") as h5:
    h5.write(u, "u")
    h5.write(p, "p")
"""


def u_exact(x):
    return np.stack([0.1 + 0.5*x[:, 0] - 0.25*x[:, 1] + x[:, 0]**2 - 0.5*x[:, 1]*x[:, 2],
                     -0.2 + x[:, 2] + 0.75*x[:, 1]**2 - 0.3*x[:, 0]*x[:, 1],
                     1.0 + 0.25*x[:, 0] - 0.5*x[:, 2] + 0.4*x[:, 2]**2 + 0.2*x[:, 0]*x[:, 2]],
                    axis=1)


def p_exact(x):
    return 2.0 - x[:, 0] + 0.5*x[:, 1] + 0.25*x[:, 2]


@pytest.fixture(scope="module", params=[1, pytest.param(3, marks=pytest.mark.mpi)])
def files(request, tmp_path_factory):
    pytest.importorskip("dolfin")
    out = str(tmp_path_factory.mktemp("layout{}".format(request.param)))
    support.run([sys.executable, "-c", WRITER, repr(U), P, out], ranks=request.param)
    return os.path.join(out, "mesh.h5"), os.path.join(out, "up.h5")


def test_p2_and_p1_come_back_at_every_node(files):
    reader = LayoutReader(*files)
    coords, conn, (u, p), names = reader.read(["u", "p"])
    assert names == ["u", "p"] and conn.shape[1] == 10
    assert np.abs(u - u_exact(coords)).max() < 1e-13
    assert np.abs(p - p_exact(coords)).max() < 1e-13           # midsides too: P1 is linear
    vx, topo, pv = reader.read_p1("p")
    assert np.abs(pv - p_exact(vx)).max() < 1e-13
    assert np.array_equal(topo, conn[:, :4])
    with pytest.raises(ValueError, match="degree 2, not 1"):
        reader.read_p1("u")


def test_the_streamed_reader_is_the_whole_one(files, tmp_path):
    mesh_path, field_path = files
    path = str(tmp_path / "permuted.h5")
    shutil.copyfile(field_path, path)
    with h5py.File(path, "r+") as h5:
        cells = np.asarray(h5["u/cells"])
        order = np.random.default_rng(1).permutation(len(cells))
        h5["u/cells"][...] = cells[order]
        h5["u/cell_dofs"][...] = np.asarray(h5["u/cell_dofs"]).reshape(len(cells), -1)[order].ravel()
    reader = LayoutReader(mesh_path, path)
    vx, topo = reader.read_mesh()
    rows, vec, _, comps, nodes = reader.cell_rows("u", len(topo))
    x = vx[topo]
    volume = np.abs(np.einsum("ij,ij->i", x[:, 1] - x[:, 0],
                              np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))) / 6.0
    where = np.empty(int(reader.cell_id.max()) + 1, dtype=np.int64)
    where[reader.cell_id] = np.arange(len(topo))
    mine = where[cells[order]]                    # the mesh row of each function row
    volumes, v_fluid, box = reader.label_volumes(chunk=13)
    got = list(reader.cell_blocks("u", volumes, chunk=17))
    assert np.array_equal(np.concatenate([v for v, _ in got]), volume[mine])
    assert np.array_equal(np.concatenate([u for _, u in got]),
                          vec[rows[mine]].reshape(len(mine), comps, nodes))
    assert v_fluid == pytest.approx(2.0, rel=1e-13)
    assert np.allclose(box, [1, 1, 2])
    for name, ids, message in (("twice", cells[order][[1] * 2 + list(range(2, len(cells)))],
                                "exactly once"),
                               ("missing", np.r_[cells.max() + 5, cells[order][1:]],
                                "names cells the mesh file does not")):
        bad = str(tmp_path / (name + ".h5"))
        shutil.copyfile(path, bad)
        with h5py.File(bad, "r+") as h5:
            h5["u/cells"][...] = ids
        with pytest.raises(ValueError, match=message):
            list(LayoutReader(mesh_path, bad).cell_blocks("u", volumes, chunk=17))
        with pytest.raises(ValueError, match=message):
            LayoutReader(mesh_path, bad).read(["u"])


def test_a_dofmap_read_wrongly_is_caught(files, tmp_path):
    """Two vertex columns of every cell's u0 block swapped: the cells sharing a vertex no
    longer agree on it."""
    mesh_path, field_path = files
    bad = str(tmp_path / "swapped.h5")
    shutil.copyfile(field_path, bad)
    with h5py.File(bad, "r+") as h5:
        dofs = np.asarray(h5["u/cell_dofs"]).reshape(-1, 30)
        dofs[:, [0, 1]] = dofs[:, [1, 0]]
        h5["u/cell_dofs"][...] = dofs.ravel()
    with pytest.raises(ValueError, match="cells disagree"):
        LayoutReader(mesh_path, bad).read(["u"])


def test_vertex_values_are_taken_as_they_are(files, tmp_path):
    mesh_path, field_path = files
    path = str(tmp_path / "vv.h5")
    shutil.copyfile(field_path, path)
    n = len(LayoutReader(mesh_path).read_mesh()[0])
    with h5py.File(path, "r+") as h5:
        h5["p/vv"] = np.arange(n, dtype=float)

    class Reader(LayoutReader):
        VERTEX_VALUES = "vv"
    assert np.array_equal(Reader(mesh_path, path).read_p1("p")[2], np.arange(n))
    # the base class does not look for them
    assert np.abs(LayoutReader(mesh_path, path).read_p1("p")[2]
                  - p_exact(LayoutReader(mesh_path).read_mesh()[0])).max() < 1e-13
    with h5py.File(path, "r+") as h5:
        del h5["p/vv"]
        h5["p/vv"] = np.zeros(n - 1)
    with pytest.raises(ValueError, match="vertex values"):
        Reader(mesh_path, path).read_p1("p")
