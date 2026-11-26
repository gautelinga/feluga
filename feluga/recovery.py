"""A continuous gradient recovered from a P1 (or P2) field by lumped-mass averaging
(Zienkiewicz-Zhu), on dolfin and feluga.backend.dolfinx.

    V = recovery_space(mesh, L, periodic)            # periodic vector P1 (dolfin)
    G = recovered_gradient(delta, V)                  # a Function on V
    g = recovered_vertex_gradient(mesh, delta, S)     # (vertices, 3): S periodic scalar P1

The recovery space is the **periodic** P1 space, not the plain one, and that is not a
detail: on a plain space the two sides of a periodic face recover independently, each
patch seeing only its own side's cells, so the recovered gradient there is one-sided
exactly where the solution is smooth. Constraining the space makes the assembly sum both
sides onto the one dof, which is the patch the geometry has. A boundary that is not
periodic (an inlet, an outlet) stays one-sided: the classical ZZ boundary degradation.

From stokes-beadpack (`ade/chi_indicator.py`), whose tolerance criterion and field
structure measures take it; the dolfin code is unchanged, so its numbers are.
"""
import numpy as np

from feluga.backend import is_dolfinx


def recovery_space(mesh, L, periodic, exempt=None, vector=True):
    """The space a recovered field is averaged into, on legacy dolfin: periodic vector P1
    (`vector`), or its scalar space, the box `L` periodic along `periodic`; its pairing
    checked (`feluga.periodic_partition.check_space_pairing`, `exempt` a slab's cut).
    On DOLFINx make it with feluga.backend.dolfinx's FunctionSpace and PeriodicBox."""
    import dolfin as df
    from feluga.periodic import periodic_box
    from feluga.periodic_partition import check_space_pairing
    space = df.VectorFunctionSpace if vector else df.FunctionSpace
    V = space(mesh, "Lagrange", 1, constrained_domain=periodic_box(L, 1e-10, periodic))
    check_space_pairing(V, L, periodic, mesh.mpi_comm(), "the recovery space", exempt)
    return V


def recovered_gradient(delta_, Vv):
    """The ZZ gradient of `delta_` on the (periodic) vector P1 space `Vv`: each dof the
    lumped average of grad(delta_) over its patch. A Function on Vv."""
    if is_dolfinx(Vv):
        return _gradient_dolfinx(delta_, Vv)
    import dolfin as df
    v = df.TestFunction(Vv)
    lump = df.assemble(df.dot(df.Constant((1.0, 1.0, 1.0)), v) * df.dx)
    G = df.Function(Vv, name="recovered_gradient")
    G.vector().set_local(
        df.assemble(df.dot(df.grad(delta_), v) * df.dx).get_local()
        / lump.get_local())
    G.vector().apply("insert")
    return G


def recovered_vertex_gradient(mesh, delta_, S):
    """`recovered_gradient` at the vertices, (vertices, 3): each component averaged on its
    own in the periodic scalar P1 space `S` -- the same lumped averages, a third of the
    dofs at a time, and no vector space of its own. Legacy dolfin: in the mesh's vertex
    order; DOLFINx: in its geometry's node order (`mesh_coordinates`)."""
    if is_dolfinx(S):
        return _vertex_gradient_dolfinx(mesh, delta_, S)
    import dolfin as df
    w = df.TestFunction(S)
    lump = df.assemble(w * df.dx).get_local()
    g = df.Function(S)
    out = np.empty((mesh.num_vertices(), 3))
    for i in range(3):
        g.vector().set_local(df.assemble(delta_.dx(i) * w * df.dx).get_local() / lump)
        g.vector().apply("insert")
        out[:, i] = g.compute_vertex_values(mesh)
    return out


def _gradient_dolfinx(delta_, Vv):
    import feluga.backend.dolfinx as L
    v = L.TestFunction(Vv)
    n = v.ufl_shape[0]
    lump = L.assemble(sum(v[i] for i in range(n)) * L.dx).get_local()
    G = L.Function(Vv, name="recovered_gradient")
    G.vector().set_local(L.assemble(L.dot(L.grad(delta_), v) * L.dx).get_local() / lump)
    G.vector().apply("insert")
    return G


def _vertex_gradient_dolfinx(mesh, delta_, S):
    import feluga.backend.dolfinx as L
    w = L.TestFunction(S)
    lump = L.assemble(w * L.dx).get_local()
    g = L.Function(S)
    dofs = np.asarray(S.dofmap.list).ravel()
    nodes = np.asarray(mesh.geometry.dofmaps[0]).ravel()
    if np.asarray(S.dofmap.list).shape[1] != mesh.geometry.dofmaps[0].shape[1]:
        raise NotImplementedError("recovered_vertex_gradient: a P1 space on an affine mesh")
    out = np.empty((len(mesh.geometry.x), 3))
    for i in range(3):
        g.vector().set_local(L.assemble(delta_.dx(i) * w * L.dx).get_local() / lump)
        g.vector().apply("insert")
        g.sync()
        out[nodes, i] = g.x.array[dofs]
    return out
