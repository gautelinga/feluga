"""The null space of constants, for a Krylov solve of a problem with a pure Neumann
pressure (or any field determined only up to a constant).

    null = constant_nullspace(A, Q)          # a pressure space
    null = constant_nullspace(A, W.sub(1))   # the pressure in a mixed space
    null.orthogonalize(b)                    # a right-hand side made consistent

The vector is one on the dofs of `V` and zero elsewhere, normalized, laid out as A's
columns. On a component of a mixed space it is set through the component's dofmap;
`Function(W).sub(1).vector()` is a view onto the whole mixed vector, and setting it
would make the null space the constants on every component.
"""


def constant_nullspace(A, V):
    """The constants on `V` (a space, or a component W.sub(i) of the space A acts
    on) as a dolfin VectorSpaceBasis, attached to A, a dolfin matrix or a petsc4py
    Mat."""
    import dolfin as df
    if not hasattr(A, "init_vector"):
        A = df.PETScMatrix(A)
    x = df.Vector(A.mpi_comm())
    A.init_vector(x, 1)
    V.dofmap().set(x, 1.0)
    x *= 1.0 / x.norm("l2")
    null = df.VectorSpaceBasis([x])
    df.as_backend_type(A).set_nullspace(null)
    return null
