"""Components of mixed Functions as Functions of their own, and vector Functions put
together from scalar ones.

    phi = subfunction(q, 0)          # component 0 of q, on its collapsed space
    set_subfunction(q, 0, value)     # and back
    uv = AssignedVectorFunction(as_vector((u0, u1)))
    uv()                             # the components copied in

Collapsing a subspace builds a dofmap, and a FunctionAssigner builds maps of its own,
so both are built once per mixed space and component and shared by every caller (the
CAPACITY most recently used are kept). Imports dolfin.
"""
import dolfin as df

from femtools.lru import LRU

CAPACITY = 64
_COMPONENTS = LRU(CAPACITY)


class _Component:
    """Component i of a mixed space W, collapsed into a space of its own, with a
    Function on it and the assigners to it from W and back."""

    def __init__(self, W, i):
        self.W = W      # held, so that W.id() is not reused while it is a key
        self.space = W.sub(i).collapse()
        self.f = df.Function(self.space)
        self.get = df.FunctionAssigner(self.space, W.sub(i))
        self.put = df.FunctionAssigner(W.sub(i), self.space)


def _component(W, i):
    key = (W.id(), i)
    c = _COMPONENTS.get(key)
    if c is None:
        c = _COMPONENTS[key] = _Component(W, i)
    return c


def subspace(W, i):
    """The collapsed space of component i of the mixed space W (built once)."""
    return _component(W, i).space


def subfunction(q, i):
    """Component i of the mixed Function q, copied into a Function on its collapsed
    space (subspace). The same Function is returned, with the current values, on
    every call for the same mixed space and component."""
    c = _component(q.function_space(), i)
    c.get.assign(c.f, q.sub(i))
    return c.f


def set_subfunction(q, i, value):
    """Set component i of the mixed Function q to `value`: a Function on the
    component's collapsed space (subspace), or anything else Function.interpolate
    takes, such as an Expression or a Function on another space, which is
    interpolated there first."""
    c = _component(q.function_space(), i)
    if not (isinstance(value, df.Function) and value.function_space().id() == c.space.id()):
        c.f.interpolate(value)
        value = c.f
    c.put.assign(q.sub(i), value)


class AssignedVectorFunction(df.Function):
    """A vector Function on the vector space of the scalar Functions in `u` (a
    ListTensor, such as as_vector((u0, u1))): same family, degree and constrained
    domain. Calling it copies the components in."""

    def __init__(self, u, name="Assigned Vector Function"):
        V = u[0].function_space()
        el = V.ufl_element()
        Vv = df.VectorFunctionSpace(V.mesh(), el.family(), el.degree(),
                                    constrained_domain=V.dofmap().constrained_domain)
        df.Function.__init__(self, Vv, name=name)
        self.u = u
        self.fa = [df.FunctionAssigner(Vv.sub(i), V) for i in range(len(u))]

    def __call__(self):
        for i, ui in enumerate(self.u):
            self.fa[i].assign(self.sub(i), ui)
