"""feluga.dolfinx's C++ extensions (folder cpp/), each built on first use.

    module(comm, "plaza_periodic", what)     the extension cpp/plaza_periodic.cpp

An extension is built with CMake (cpp/CMakeLists.txt) against the installed DOLFINx and
nanobind, into feluga' cache folder (FELUGA_JIT_DIR, else ~/.cache/feluga), under
a name that hashes its sources and the DOLFINx and Python versions: rank 0 builds into a
fresh folder renamed into place when complete while the others wait, and a failed build
or load raises on every rank (feluga.jit). Concurrent jobs share the cache.

A build takes tens of seconds, which a large job spends waiting on rank 0, and compute
nodes may have no CMake; build them all beforehand, with the job's FELUGA_JIT_DIR:

    python3 -m feluga.dolfinx.compiled
"""
import hashlib
import importlib.util
import os
import subprocess
import sys

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp")
# What each extension's build reads besides CMakeLists.txt and its own source
_HEADERS = {"plaza_periodic": [os.path.join(HERE, "..", "..", "cpp", "plaza_periodic_rules.h")]}
_LOADED = {}
EXTENSIONS = ("folded_assembly", "plaza_periodic")


def module(comm, name, what):
    """The extension cpp/`name`.cpp, built if not yet in the cache and loaded once.
    `what` names the caller in errors. Collective."""
    if name in _LOADED:
        return _LOADED[name]
    import tempfile

    import dolfinx

    from feluga.jit import install_dir, on_every_rank, on_rank0
    sources = [os.path.join(HERE, f) for f in (name + ".cpp", "CMakeLists.txt")]
    h = hashlib.sha1(name.encode())
    for path in sources + _HEADERS.get(name, []):
        with open(path, "rb") as f:
            h.update(f.read())
    # everything the binary depends on: two builds of one version (real or complex,
    # 32- or 64-bit PetscInt, a container and its host sharing ~/.cache) must not share it
    from petsc4py import PETSc
    try:
        import nanobind
        nb = nanobind.__version__
    except ImportError:
        nb = ""
    import dolfinx.common
    h.update("|".join([dolfinx.__version__, getattr(dolfinx.common, "git_commit_hash", ""),
                       os.path.dirname(dolfinx.__file__), sys.version,
                       str(np.dtype(PETSc.IntType)), str(np.dtype(PETSc.ScalarType)),
                       ".".join(map(str, PETSc.Sys.getVersion())), nb]).encode())
    root = os.environ.get("FELUGA_JIT_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "feluga")
    target = name + "_x"
    folder = target + "_" + h.hexdigest()[:16]

    def built_in(path):
        found = [f for f in os.listdir(path) if f.startswith(target) and f.endswith(".so")]
        if len(found) != 1:
            raise RuntimeError("{} holds {} extension libraries, not one".format(
                path, len(found)))
        return os.path.join(path, found[0])

    def build():
        final = os.path.join(root, folder)
        if os.path.exists(os.path.join(final, "done")):
            return built_in(final)
        os.makedirs(root, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=folder + ".", dir=root)
        steps = [["cmake", "-S", HERE, "-B", tmp, "-DCMAKE_BUILD_TYPE=Release",
                  "-DMODULE=" + name, "-DPython_EXECUTABLE=" + sys.executable],
                 ["cmake", "--build", tmp, "-j", str(min(8, os.cpu_count() or 1))]]
        for step in steps:
            proc = subprocess.run(step, capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError("{} failed against DOLFINx {}:\n{}".format(
                    " ".join(step[:2]), dolfinx.__version__,
                    (proc.stdout + proc.stderr)[-4000:]))
        open(os.path.join(tmp, "done"), "w").close()
        return built_in(install_dir(tmp, final))

    so = on_rank0(comm, build, "{}: building the extension against DOLFINx {}".format(
        what, dolfinx.__version__))

    def load():
        spec = importlib.util.spec_from_file_location(target, so)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded

    _LOADED[name] = on_every_rank(comm, load, "{}: loading {}".format(what, so),
                                  wait_for=so)
    return _LOADED[name]


def build_all():
    """Build (or find in the cache) every extension, in this process alone."""
    return {name: module(None, name, "feluga.dolfinx.compiled").__file__
            for name in EXTENSIONS}


if __name__ == "__main__":
    for name, path in build_all().items():
        print("{}: {}".format(name, path))
