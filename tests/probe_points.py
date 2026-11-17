"""The probe points of tests/test_backend_probes.py and its worker."""
import numpy as np


def points():
    """40 seeded random points inside the unit cube, then one outside it."""
    rng = np.random.default_rng(7)
    return np.vstack((0.02 + 0.96 * rng.random((40, 3)), [[2.0, 0.5, 0.5]]))


def face_points():
    """48 seeded random points on the planes z = 1/4 and z = 1/2, which are faces of the
    4^3 box mesh's cells."""
    rng = np.random.default_rng(11)
    xy = 0.05 + 0.9 * rng.random((48, 2))
    return np.column_stack((xy, np.repeat([0.25, 0.5], 24)))
