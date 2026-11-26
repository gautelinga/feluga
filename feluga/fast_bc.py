"""DirichletBC.apply on a vector, with the constrained dofs found once.

    from feluga.fast_bc import apply
    apply(bc, b)          # bc.apply(b)
    apply(bc, A)          # bc.apply(A), by dolfin
    apply(bc, A, b)       # bc.apply(A, b): A by dolfin, b here
    apply([bc1, bc2], b)  # bc1.apply(b); bc2.apply(b), as one setting of the vector

DirichletBC.apply searches the boundary facets and tabulates their dofs at every call.
Here that is done once per condition: one dolfin apply onto a vector of NaNs gives
this rank's owned constrained dofs, and a later apply sets those entries and updates
the ghost values, as dolfin does.

The values follow the condition's value at every call when it is a Constant (its
current values) or a Function on the condition's space, changed in place or not (its
values at the dofs). Any other value, an Expression for one, and a condition on a
component of a space, are left to dolfin's apply at every call, as is a matrix, whose
rows dolfin sets keeping its nonzero pattern. The dofs are found once per condition
object, so a condition whose markers or mesh change in place needs a new object,
and one made anew for every apply costs more than dolfin's apply. The cache keeps the
256 most recently used conditions, with what they hold.

Several conditions given as a list or tuple are applied in their order, as one call
each would apply them (a later one's value wins on a dof both constrain), with one
setting of the vector's values and one update of its ghosts for the cached ones in a
row.
"""
import numpy as np

from feluga.lru import LRU

_CACHE = LRU(256)


class _Cached:
    def __init__(self, bc):
        import dolfin as df
        self.bc = bc                                # held, so that its id stays unique
        V = bc.function_space()
        g = bc.value()
        self.kind, self.value_id = None, g.id()
        if len(V.component()):
            return
        if isinstance(g, df.cpp.function.Constant):
            self.kind = "constant"
            # The component of each constrained dof: the value of a copy of the
            # condition whose value is the component's number.
            marker = df.DirichletBC(bc)
            n = g.value_size()
            marker.set_value(df.Constant(np.arange(n, dtype=float) if n > 1 else 0.))
        elif isinstance(g, df.cpp.function.Function) and g.function_space().id() == V.id():
            self.kind = "function"
            marker = bc
        else:
            return
        from petsc4py import PETSc
        x = df.Function(df.FunctionSpace(V)).vector()
        x.set_local(np.full(x.local_size(), np.nan))
        x.apply("insert")
        marker.apply(x)
        values = x.get_local()
        self.dofs = np.flatnonzero(~np.isnan(values))
        if self.kind == "constant":
            self.component = values[self.dofs].astype(int)
        self.rows = (V.dofmap().ownership_range()[0] + self.dofs).astype(PETSc.IntType)

    def entries(self):
        """`(rows, values)`: the owned global rows the condition sets and their current
        values; None for a condition left to dolfin."""
        g = self.bc.value()
        if g.id() != self.value_id:                 # set_value gave it a new value
            self.__init__(self.bc)
        if self.kind is None:
            return None
        if self.kind == "constant":
            values = g.values()[self.component]
        else:                                       # the owned values, read in place
            from dolfin import as_backend_type
            values = as_backend_type(g.vector()).vec().getArray(readonly=True)[self.dofs]
        return self.rows, values


def _cached(bc):
    entry = _CACHE.get(id(bc))
    if entry is None:
        entry = _CACHE[id(bc)] = _Cached(bc)
    return entry


def _apply_vector(bcs, b):
    """The conditions `bcs` applied to the vector b in their order: the cached ones in
    a row set in one call and the ghosts updated once, the others by dolfin."""
    from dolfin import as_backend_type
    rows, values = [], []

    def flush():
        if rows:
            as_backend_type(b).vec().setValues(np.concatenate(rows), np.concatenate(values))
            b.apply("insert")
            del rows[:], values[:]
    for bc in bcs:
        found = _cached(bc).entries()
        if found is None:
            flush()
            bc.apply(b)
        else:
            rows.append(found[0])
            values.append(found[1])
    flush()


def apply(bc, *tensors):
    """bc.apply(*tensors): a vector through the cached dofs, a matrix by dolfin. `bc`
    may be a list or tuple of conditions, applied in their order."""
    from dolfin.cpp.la import GenericMatrix, GenericVector
    bcs = list(bc) if isinstance(bc, (list, tuple)) else [bc]
    if len(tensors) == 1 and isinstance(tensors[0], GenericVector):
        _apply_vector(bcs, tensors[0])
    elif len(tensors) == 2 and isinstance(tensors[0], GenericMatrix) \
            and isinstance(tensors[1], GenericVector):
        for one in bcs:
            one.apply(tensors[0])
        _apply_vector(bcs, tensors[1])
    else:
        for one in bcs:
            one.apply(*tensors)
