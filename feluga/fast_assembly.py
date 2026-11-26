"""dolfin.assemble for a form assembled again and again, compiled once.

    from feluga.fast_assembly import assemble
    a = mu * inner(grad(u), grad(v)) * dx      # build the form once ...
    A = assemble(a)                            # ... and assemble it every step
    assemble(a, tensor=A)

dolfin.assemble turns a UFL form into a dolfin Form on every call: it computes the
form's signature and looks the generated code up in the JIT cache before it
assembles. Here that is done once per form object, which is kept, and later calls
go straight to dolfin's Assembler. The form's coefficients are read at every call,
so a Function changed in place is assembled with its new values.

The saving needs the same form object each time: a form built in the call, as in
assemble(mu * inner(grad(u), grad(v)) * dx), is a new object every time, and is
compiled every time too (the cache holds the CAPACITY most recently used forms). The
form compiler parameters are those in force when a form is first assembled.

`assemble_matrix(form, bcs)` is for a matrix that never changes, such as a mass
matrix: it is assembled, with the Dirichlet conditions applied, once per form and
conditions, and the same matrix returned to every caller (copy it to change it).
"""
from feluga.lru import LRU

# The compiled forms with their UFL forms (held, so that an id is not reused while
# it is a key). UFL forms cannot be referenced weakly, so the cache is bounded
# instead: a form built anew in every call, which this cannot help, cycles through
# it instead of filling memory.
CAPACITY = 256
_COMPILED = LRU(CAPACITY)


def compiled(form):
    """The dolfin Form for a UFL form, made on the first call and kept while it is
    among the CAPACITY most recently used."""
    import dolfin as df
    entry = _COMPILED.get(id(form))
    if entry is None:
        entry = _COMPILED[id(form)] = (form, df.Form(form))
    return entry[1]


def assemble(form, tensor=None):
    """dolfin.assemble(form, tensor) with the form compiled once (compiled): a
    matrix, a vector or a scalar, as dolfin.assemble gives. Without a tensor, the
    first assembly makes one, as dolfin.assemble does."""
    import dolfin as df
    if tensor is None:
        return df.assemble(form)
    df.cpp.fem.Assembler().assemble(tensor, compiled(form))
    return tensor.get_scalar_value() if isinstance(tensor, df.cpp.la.Scalar) else tensor


# Constant matrices by (form, conditions). UFL forms compare by their expressions, so
# a form built again from the same spaces' arguments finds the same matrix.
MATRIX_CAPACITY = 64
_MATRICES = LRU(MATRIX_CAPACITY)


def _rows(bc):
    """What applying Dirichlet condition `bc` does to a matrix: its constrained rows,
    which become the identity's whatever the condition's values; as bytes of the sorted
    local indices."""
    import numpy as np
    rows = getattr(bc, "global_rows", None)       # feluga.backend.dolfinx's
    if rows is not None:
        r = np.asarray(rows(), dtype=np.int64)
    else:                                         # dolfin's: its local dofs
        r = np.fromiter(bc.get_boundary_values().keys(), dtype=np.int64)
    return np.sort(r).tobytes()


def _comm(form):
    """The communicator of the form's mesh (mpi4py), on either backend."""
    mesh = form.ufl_domain().ufl_cargo()
    return mesh.mpi_comm() if callable(getattr(mesh, "mpi_comm", None)) else mesh.comm


def constant_matrix(form, bcs, assemble, cache=_MATRICES):
    """The matrix of the bilinear `form` (`assemble(form)`, either backend's) with the
    Dirichlet conditions `bcs` applied, made on the first call for an equal form and
    conditions that constrain the same rows, and returned again on later ones (while
    among the `cache`'s most recently used). Conditions differing only in their values
    give the same matrix. Collective: the ranks agree on whether it is cached."""
    from mpi4py import MPI
    key = (form, tuple(_rows(bc) for bc in bcs))
    A = cache.get(key)
    if not _comm(form).allreduce(A is not None, op=MPI.LAND):
        A = None
    if A is None:
        if len(form.arguments()) != 2:
            raise ValueError("assemble_matrix takes a bilinear form")
        A = assemble(form)
        for bc in bcs:
            bc.apply(A)
        cache[key] = A
    return A


def assemble_matrix(form, bcs=()):
    """The matrix of the bilinear `form` with the DirichletBCs `bcs` applied, assembled
    once per equal form and conditions (`constant_matrix`)."""
    return constant_matrix(form, bcs, assemble)
