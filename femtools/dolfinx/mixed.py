"""Components of mixed DOLFINx Functions as Functions of their own: the DOLFINx
counterpart of `femtools.mixed`'s subspace and subfunction.

    S = subspace(W, 0)               # component 0's collapsed space (shared)
    subfunction(q, 0, out)           # component 0 of q into `out`, a Function on S
    set_subfunction(q, 0, value)     # and back, from a Function on S

Each component's collapsed space and dof map are built once per mixed space and component
and shared by every caller; they are kept on the mixed space, and go with it. Values move as
plain index copies over owned and ghost dofs, so Functions holding periodic slaves (see
femtools.dolfinx.unique) carry them along.
"""
import numpy as np

from femtools.dolfinx.periodic import flat_collapse_map


class _Component:
    """Component i of mixed space W, collapsed, and its dof map into W (collapsed local
    dof -> W local dof)."""

    def __init__(self, W, i):
        self.space, dofs = W.sub(i).collapse()
        self.map = flat_collapse_map(dofs)


def _component(W, i):
    """Kept in W's own dict: W, its components and its mesh form a cycle the collector
    frees with W, where a module-level cache would keep every mesh it saw."""
    held = W.__dict__.setdefault("_femtools_components", {})
    c = held.get(i)
    if c is None:
        c = held[i] = _Component(W, i)
    return c


def subspace(W, i):
    """The collapsed space of component i of the mixed space W (built once)."""
    return _component(W, i).space


def component_map(W, i):
    """The local dofs of W of component i, in the order of subspace(W, i)'s."""
    return _component(W, i).map


def subfunction(q, i, out):
    """Component i of the mixed Function q, copied into `out`, a Function on
    subspace(W, i). Returns it."""
    c = _component(q.function_space, i)
    if out.function_space is not c.space:
        raise ValueError("subfunction: out must be on subspace(W, {})".format(i))
    np.take(q.x.array, c.map, out=out.x.array)
    return out


def set_subfunction(q, i, value):
    """Set component i of the mixed Function q to `value`, a Function on
    subspace(W, i)."""
    c = _component(q.function_space, i)
    if value.function_space is not c.space:
        raise ValueError("set_subfunction: value must be on subspace(W, {})".format(i))
    q.x.array[c.map] = value.x.array
    q.x.scatter_forward()
