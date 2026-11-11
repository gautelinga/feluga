"""dolfin.assemble for a form assembled again and again, compiled once.

    from femtools.fast_assembly import assemble
    a = mu * inner(grad(u), grad(v)) * dx      # build the form once ...
    A = assemble(a)                            # ... and assemble it every step
    assemble(a, tensor=A)

dolfin.assemble turns a UFL form into a dolfin Form on every call: it computes the
form's signature and looks the generated code up in the JIT cache before it
assembles. Here that is done once per form object, which is kept, and later calls
go straight to dolfin's Assembler. The form's coefficients are read at every call,
so a Function changed in place is assembled with its new values. On a P2 velocity
space with 33k cells this saves 3.6 ms of 16 ms for a viscous term with two
coefficients, less for simpler forms.

The saving needs the same form object each time: a form built in the call, as in
assemble(mu * inner(grad(u), grad(v)) * dx), is a new object every time, and is
compiled every time too (the cache holds the CAPACITY most recently used forms). The
form compiler parameters are those in force when a form is first assembled.

`assemble_matrix(form, bcs)` is for a matrix that never changes, such as a mass
matrix: it is assembled, with the Dirichlet conditions applied, once per form and
conditions, and the same matrix returned to every caller (copy it to change it).
"""
from femtools.lru import LRU

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


def assemble_matrix(form, bcs=()):
    """The matrix of the bilinear `form` with the DirichletBCs `bcs` applied,
    assembled on the first call for an equal form and the same conditions and
    returned again on later ones (while among the MATRIX_CAPACITY most recently
    used)."""
    key = (form, tuple(bcs))
    A = _MATRICES.get(key)
    if A is None:
        if len(form.arguments()) != 2:
            raise ValueError("assemble_matrix takes a bilinear form")
        A = assemble(form)
        for bc in bcs:
            bc.apply(A)
        _MATRICES[key] = A
    return A
