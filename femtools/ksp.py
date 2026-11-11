"""PETSc Krylov solvers set up from PETSc options under a prefix.

    from femtools.ksp import configure, solver_options
    options = solver_options("bicgstab", "hypre_amg", {"ksp_rtol": 1e-8})
    configure(solver.ksp(), "velocity_", options)

`solver_options` takes a method and a preconditioner by PETSc's names or by dolfin's
(`KSP_NAMES`, `PC_NAMES`: dolfin's 'bicgstab' is PETSc's 'bcgs', 'hypre_amg' is hypre's
BoomerAMG). `configure` sets the options in PETSc's database under the prefix, removes
any an earlier solver left there, and raises when PETSc set up another method or
preconditioner than the one asked for: an option PETSc does not know, or one
misspelt, is otherwise ignored without a word, and the solver runs with its default.

The options stay in the database after `configure`, since some are read only when the
preconditioner is set up, at the first solve (the sub_ options of bjacobi and asm,
say). `scoped` sets options for the length of a `with` block instead and then puts
back what the database held before, for a solve that must not leave its options to
later ones in the process.
"""
from contextlib import contextmanager

# dolfin's names for Krylov methods and preconditioners that PETSc names differently.
# Any other name is given to PETSc as it is.
KSP_NAMES = {"bicgstab": "bcgs", "default": "gmres"}
BOOMERAMG = {"pc_type": "hypre", "pc_hypre_type": "boomeramg"}
PC_NAMES = {
    "default": {"pc_type": "bjacobi"},
    "amg": BOOMERAMG,
    "hypre_amg": BOOMERAMG,
    "hypre_euclid": {"pc_type": "hypre", "pc_hypre_type": "euclid"},
    "hypre_parasails": {"pc_type": "hypre", "pc_hypre_type": "parasails"},
    "petsc_amg": {"pc_type": "gamg"},
    "ml_amg": {"pc_type": "ml"},
    "additive_schwarz": {"pc_type": "asm"},
}
# BoomerAMG settings for 3D: cheaper coarsening and interpolation, which in 3D make
# each solve of a Poisson problem several times faster for twice the iterations.
BOOMERAMG_3D = {"pc_hypre_boomeramg_strong_threshold": 0.5,
                "pc_hypre_boomeramg_coarsen_type": "HMIS",
                "pc_hypre_boomeramg_interp_type": "ext+i",
                "pc_hypre_boomeramg_P_max": 4,
                "pc_hypre_boomeramg_agg_nl": 1}


def solver_options(solver_type, preconditioner_type, petsc_options=None, pc_options=None):
    """PETSc options for a method and a preconditioner named by PETSc or by dolfin,
    then `petsc_options`, any further ones. `pc_options`, when given, replace the
    preconditioner's."""
    options = dict(ksp_type=KSP_NAMES.get(solver_type, solver_type))
    options.update(pc_options or PC_NAMES.get(preconditioner_type,
                                              {"pc_type": preconditioner_type}))
    options.update(petsc_options or {})
    return options


def configure(ksp, prefix, options):
    """Set up the petsc4py KSP `ksp` from `options`, PETSc options given under
    `prefix` (with its trailing underscore), after removing those an earlier solver
    left under it; raise ValueError when PETSc did not take the requested ksp_type
    and pc_type. Every option whose name starts with `prefix` is removed, so also
    those of a solver whose prefix extends it: 'u_' clears 'u_inner_ksp_type' (an
    inner solver's options, as a preconditioner's sub-solvers are meant to be)."""
    from petsc4py import PETSc
    db = PETSc.Options()
    for key in db.getAll():
        if key.startswith(prefix):
            db.delValue(key)
    for key, value in options.items():
        db[prefix + key] = value
    ksp.setOptionsPrefix(prefix)
    ksp.setFromOptions()
    got = ksp.getType(), ksp.getPC().getType()
    want = options.get("ksp_type", got[0]), options.get("pc_type", got[1])
    if got != want:
        raise ValueError("{}: PETSc set up {} + {}, not the requested {} + {}".format(
            prefix.rstrip("_"), *got, *want))


@contextmanager
def scoped(prefix, options):
    """Set `options` under `prefix` for the `with` block, then restore the database:
    an option that was there before gets its value back, one that was not is
    removed. A value of None sets a flag (an option without a value)."""
    from petsc4py import PETSc
    db = PETSc.Options()
    before = db.getAll()
    names = [prefix + key for key in options]
    try:
        for name, value in zip(names, options.values()):
            db[name] = value
        yield
    finally:
        for name in names:
            if name not in before:
                db.delValue(name)
            else:
                db[name] = before[name]
