"""`femtools.dolfinx`: assembly over unique dofs (periodic slaves folded onto masters).

Against plain DOLFINx without periodicity (identical), against PᵀAP with an explicit P
(square, rectangular and mixed forms, forms over cells, interior facets and a subdomain;
right-hand sides) with exact preallocation, against dolfinx_mpc (a periodic solve with
nonzero Dirichlet data on the corners, applied by the backend's DirichletBC), on one rank
and on three; and against dolfin's constrained_domain in one process (unique dofs, a
periodic field, the operator). The preallocation, counted from the dofmaps, against a
plain assembly's pattern folded onto unique dofs (spaces, periodicity, integral kinds,
rectangular forms, 2D and 3D) on one to four ranks.

    python3 -m pytest tests/test_dolfinx_unique.py [-m mpi]
"""
import json
import os
import sys

import pytest

pytest.importorskip("dolfinx")
pytestmark = [pytest.mark.dolfinx]

from tests import support  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(worker, tmp_path, ranks):
    out = tmp_path / f"{worker}.{ranks}.json"
    support.run([sys.executable, os.path.join(HERE, worker + ".py"), str(out)], ranks)
    return json.loads(out.read_text())


def _check_unique(r):
    assert r["plain_matrix"] == 0.0 and r["plain_rhs"] == 0.0
    for name in ("fold_square", "fold_rect", "fold_mixed"):
        assert r[name] < 1e-14, name
        assert r[name + "_exact"], name
    for name in ("same_cells", "interior_facets", "cell_subdomain"):
        assert r["share_" + name] < 1e-14, name
        assert r["share_" + name + "_exact"], name
    assert r["fold_rhs"] < 1e-14
    assert r["slaves_V"] == 49 and r["unique_dofs_V"] == 1176
    assert r["roundtrip"] == 0.0
    if r["vs_dolfinx_mpc"] is not None:
        assert r["vs_dolfinx_mpc"] < 1e-12
    assert r["nullspace"]
    assert r["interleaved"]
    assert r["fold_mixed_interleaved"] < 1e-14 and r["fold_mixed_interleaved_exact"]
    assert r["interleaved_roundtrip"] == 0.0
    assert r["fold_mixed_reordered"] < 1e-14 and r["fold_mixed_reordered_exact"]
    assert r["reordered_interleaved"]
    assert r["reordered_roundtrip"] == 0.0
    assert r["reordered_dg0"]
    for th in ("th", "thv", "thv3"):           # [P2, P2, P1]; [vector P2, P1] in 2D, 3D
        assert r[th + "_fold"] < 1e-14 and r[th + "_fold_exact"], th
        assert r[th + "_grouped"], th
        assert r[th + "_nodes_together"] and r[th + "_vertex_nodes_whole"], th
        assert r[th + "_roundtrip"] == 0.0, th
        assert r[th + "_reference_graph_same_order"], th
    assert r["p3p2_kept"]                       # P3's two edge dofs against P2's one
    assert r["band_reordered"] < r["band_own_order"] / 2, (r["band_reordered"],
                                                           r["band_own_order"])


def test_unique_dofs_serial(tmp_path):
    _check_unique(_run("dolfinx_unique_worker", tmp_path, 1))


@pytest.mark.mpi
def test_unique_dofs_three_ranks(tmp_path):
    serial = _run("dolfinx_unique_worker", tmp_path, 1)
    parallel = _run("dolfinx_unique_worker", tmp_path, 3)
    _check_unique(parallel)
    assert parallel["solution_norm"] == pytest.approx(serial["solution_norm"], rel=1e-12)


def _check_pattern(r):
    assert r.pop("refused") is True
    assert len(r) == 198
    assert not {k: v for k, v in r.items() if v}


def test_preallocation_serial(tmp_path):
    _check_pattern(_run("dolfinx_pattern_worker", tmp_path, 1))


@pytest.mark.mpi
@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_preallocation_parallel(tmp_path, ranks):
    _check_pattern(_run("dolfinx_pattern_worker", tmp_path, ranks))


@pytest.mark.dolfin
def test_same_unique_dofs_as_dolfin(tmp_path):
    pytest.importorskip("dolfin")
    r = _run("dolfinx_cross_worker", tmp_path, 1)
    assert r["dofs"][0] == r["dofs"][1]
    assert r["norm"] < 1e-14 and r["sum"] < 1e-14
    assert r["operator"] < 1e-13
