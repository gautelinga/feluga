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
