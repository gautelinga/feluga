"""A suite that spawns mpirun is also slow. The children run unbound to cores: under
pytest-xdist several mpirun jobs at once would otherwise each bind from core 0."""
import os

import pytest

os.environ.setdefault("OMPI_MCA_hwloc_base_binding_policy", "none")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("mpi") and not item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)
