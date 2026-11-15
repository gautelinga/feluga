"""`femtools.ksp`: a KSP set up from options under a prefix, dolfin's names translated,
a preconditioner PETSc did not build refused; options set for a block and restored.

    python3 -m pytest tests/test_ksp.py
"""
import pytest

PETSc = pytest.importorskip("petsc4py.PETSc")

from femtools.ksp import configure, scoped, solver_options  # noqa: E402


def ksp():
    return PETSc.KSP().create(PETSc.COMM_SELF)


def test_dolfin_names_are_translated():
    assert solver_options("bicgstab", "hypre_amg", {"ksp_rtol": 1e-8}) == dict(
        ksp_type="bcgs", pc_type="hypre", pc_hypre_type="boomeramg", ksp_rtol=1e-8)
    assert solver_options("cg", "jacobi") == dict(ksp_type="cg", pc_type="jacobi")
    assert solver_options("gmres", "schur", pc_options={"pc_type": "python"}) == dict(
        ksp_type="gmres", pc_type="python")


def test_configure_sets_up_what_is_asked_and_clears_the_prefix():
    k = ksp()
    configure(k, "t1_", solver_options("bicgstab", "bjacobi", {"ksp_rtol": 1e-3}))
    assert (k.getType(), k.getPC().getType()) == ("bcgs", "bjacobi")
    assert k.getTolerances()[0] == pytest.approx(1e-3)
    configure(k, "t1_", solver_options("cg", "jacobi"))
    assert "t1_ksp_rtol" not in PETSc.Options().getAll()


def test_configure_refuses_what_petsc_did_not_build():
    """A KSP that kept its own type (here one that ignores the database) is refused;
    a name PETSc does not know fails in PETSc."""
    class Ignoring:
        def setOptionsPrefix(self, prefix):
            pass

        def setFromOptions(self):
            pass

        def getType(self):
            return "gmres"

        def getPC(self):
            return self

    with pytest.raises(ValueError, match="PETSc set up gmres \\+ gmres, not the requested"):
        configure(Ignoring(), "t2_", dict(ksp_type="cg", pc_type="jacobi"))
    with pytest.raises(PETSc.Error):
        configure(ksp(), "t2_", dict(ksp_type="cg", pc_type="no_such_pc"))


def test_scoped_restores_the_database():
    db = PETSc.Options()
    db["t3_ksp_rtol"] = 0.5
    with scoped("t3_", {"ksp_rtol": 1e-6, "ksp_monitor": None}):
        assert db.getAll()["t3_ksp_rtol"] == "1e-06" and db.hasName("t3_ksp_monitor")
    assert db.getAll()["t3_ksp_rtol"] == "0.5" and not db.hasName("t3_ksp_monitor")
    db.delValue("t3_ksp_rtol")


def test_defaults_leave_what_the_database_holds():
    from femtools.ksp import defaults
    db = PETSc.Options()
    db["tdef_pc_hypre_boomeramg_P_max"] = 2
    try:
        got = defaults("tdef_", {"pc_hypre_boomeramg_P_max": 0, "pc_hypre_boomeramg_agg_nl": 0})
        assert got == {"tdef_pc_hypre_boomeramg_agg_nl": 0}
    finally:
        db.delValue("tdef_pc_hypre_boomeramg_P_max")


def test_the_classical_set_is_taken_by_boomeramg():
    """configure() refuses a preconditioner PETSc did not build; hypre reads the set."""
    from femtools.ksp import BOOMERAMG, BOOMERAMG_CLASSICAL
    k = ksp()
    A = PETSc.Mat().createAIJ((4, 4), comm=PETSc.COMM_SELF)
    A.setUp()
    for i in range(4):
        A[i, i] = 2.0
    A.assemble()
    k.setOperators(A)
    options = dict(ksp_type="cg", **BOOMERAMG, pc_hypre_boomeramg_strong_threshold=0.7,
                   **BOOMERAMG_CLASSICAL)
    try:
        configure(k, "tcls_", options)
        k.setUp()
        assert k.getPC().getHYPREType() == "boomeramg"
    finally:
        configure(ksp(), "tcls_", {})


def test_a_diagonal_matrix():
    import numpy as np
    from mpi4py import MPI
    from femtools.ksp import diagonal_matrix
    D = diagonal_matrix([1.0, 2.0, 3.0], MPI.COMM_SELF)
    assert D.getSize() == (3, 3) and D.getInfo()["nz_used"] == 3
    assert np.array_equal(D.getDiagonal().getArray(), [1.0, 2.0, 3.0])
