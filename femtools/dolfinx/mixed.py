"""Components of mixed DOLFINx Functions as Functions of their own, and vector Functions
put together from scalar ones: the DOLFINx counterpart of `femtools.mixed`, with the
same names.

    phi = subfunction(q, 0)          # component 0 of q, on its collapsed space
    set_subfunction(q, 0, value)     # and back (a Function there, or anything
                                     # Function.interpolate takes, such as a callable)
    uv = AssignedVectorFunction([u0, u1])
    uv()                             # the components copied in

Each component's collapsed space and dof map are built once per mixed space and component
and shared by every caller (the CAPACITY most recently used are kept). Values move as
plain index copies over owned and ghost dofs, so Functions holding periodic slaves (see
femtools.dolfinx.unique) carry them along.
"""
import numpy as np

from femtools.dolfinx.periodic import flat_collapse_map
from femtools.lru import LRU

CAPACITY = 64
_COMPONENTS = LRU(CAPACITY)


class _Component:
    """Component i of mixed space W, collapsed, with a Function on it and its dof map
    into W (collapsed local dof -> W local dof)."""

    def __init__(self, W, i):
        from dolfinx import fem

        self.W = W  # held, so that id(W) is not reused while it is a key
        self.space, dofs = W.sub(i).collapse()
        self.map = flat_collapse_map(dofs)
        self.f = fem.Function(self.space)


def _component(W, i):
    key = (id(W), i)
    c = _COMPONENTS.get(key)
    if c is None or c.W is not W:
        c = _COMPONENTS[key] = _Component(W, i)
    return c


def subspace(W, i):
    """The collapsed space of component i of the mixed space W (built once)."""
    return _component(W, i).space


def subfunction(q, i):
    """Component i of the mixed Function q, copied into a Function on its collapsed
    space. The same Function is returned, with the current values, on every call for
    the same mixed space and component."""
    c = _component(q.function_space, i)
    c.f.x.array[:] = q.x.array[c.map]
    return c.f


def set_subfunction(q, i, value):
    """Set component i of the mixed Function q to `value`: a Function on the
    component's collapsed space, or anything Function.interpolate takes (a callable of
    the coordinates, a Function on another space of the same mesh), interpolated there
    first."""
    c = _component(q.function_space, i)
    if not (hasattr(value, "function_space") and value.function_space is c.space):
        c.f.interpolate(value)
        value = c.f
    q.x.array[c.map] = value.x.array
    q.x.scatter_forward()


class AssignedVectorFunction:
    """A vector Function on the vector space of the scalar Functions `u` (all on one
    space). Calling it copies the components in; `.function` is the vector Function."""

    def __init__(self, u, name="Assigned Vector Function"):
        import basix.ufl
        from dolfinx import fem

        V = u[0].function_space
        el = basix.ufl.blocked_element(V.ufl_element(), shape=(len(u),))
        Vv = fem.functionspace(V.mesh, el)
        self.function = fem.Function(Vv, name=name)
        self.u = list(u)
        self.maps = [flat_collapse_map(Vv.sub(i).collapse()[1]) for i in range(len(u))]

    def __call__(self):
        for m, ui in zip(self.maps, self.u):
            self.function.x.array[m] = ui.x.array
        self.function.x.scatter_forward()
        return self.function
