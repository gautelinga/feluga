"""dolfin's Expression on DOLFINx: C++ code compiled, as dolfin compiles it.

    e = CompiledExpression("tanh((x[1] - y0)/(sqrt(2)*eps))", y0=0.5, eps=0.02)
    u.interpolate(e)                     # DOLFINx calls e(x), x of shape (3, n)
    e.y0 = 0.6                           # parameters by name, as in dolfin

    v = CompiledExpression(("x[1]", "-x[0]"))     # a vector: one string a component

Each component is a C++ expression in `x[0]`, `x[1]`, `x[2]` and the parameters, with
what dolfin's Expressions see: `<cmath>` with `using namespace std` (sqrt, pow, tanh,
abs, max, ...), `pi` and `M_PI`, `DOLFIN_EPS`, `DOLFIN_PI` and `near(a, b[, eps])`. Being
C++, the conditional operator, `&&`, `||` and integer division mean what they mean in
dolfin.

The code is compiled with the C++ compiler (`CXX`, else `c++`) into a shared library
kept in a cache folder (`FEMTOOLS_JIT_DIR`, else `~/.cache/femtools/expressions`) under a
hash of the code and the compiler flags, and loaded with ctypes; a cached library is
reused, in later runs too. Under MPI, rank 0 of `comm` compiles and the others wait for
it: making a CompiledExpression is collective on `comm` (MPI.COMM_WORLD by default),
as dolfin's JIT is.
"""
import ctypes
import hashlib
import os
import subprocess
import tempfile

import numpy as np

FLAGS = ["-O2", "-std=c++17", "-shared", "-fPIC"]

TEMPLATE = r"""
#include <cmath>
#include <algorithm>
using namespace std;
static const double DOLFIN_EPS = 3.0e-16;
static const double DOLFIN_PI = M_PI;
static const double pi = M_PI;
static inline bool near(double a, double b, double eps = 3.0e-16) {{ return std::abs(a - b) <= eps; }}

extern "C" void eval(const double* femtools_X_, long femtools_n_, const double* femtools_p_,
                     double* femtools_out_)
{{
{params}
  for (long femtools_i_ = 0; femtools_i_ < femtools_n_; ++femtools_i_)
  {{
    const double* x = femtools_X_ + 3 * femtools_i_;
{components}
  }}
}}
"""

_LIBRARIES = {}


def _cache_dir():
    d = os.environ.get("FEMTOOLS_JIT_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "femtools", "expressions")
    os.makedirs(d, exist_ok=True)
    return d


def _source(codes, names):
    if "x" in names:
        raise ValueError("CompiledExpression: a parameter cannot be named x")
    params = "\n".join("  const double {} = femtools_p_[{}];".format(n, k)
                       for k, n in enumerate(names))
    comps = "\n".join("    femtools_out_[{w} * femtools_i_ + {k}] = ({c});".format(
        w=len(codes), k=k, c=c) for k, c in enumerate(codes))
    return TEMPLATE.format(params=params, components=comps)


def _compile(source, comm):
    """The loaded library of `source`, compiled once (by rank 0 of comm) and cached. A
    failure on rank 0 -- code that does not compile, no compiler, a cache folder it may
    not write -- raises on every rank (femtools.jit)."""
    from femtools.jit import on_every_rank, on_rank0
    cxx = os.environ.get("CXX", "c++")
    key = hashlib.sha1((source + "\0" + cxx + " " + " ".join(FLAGS)).encode()).hexdigest()
    if key in _LIBRARIES:
        return _LIBRARIES[key]

    def build():
        d = _cache_dir()
        path = os.path.join(d, "expr_" + key + ".so")
        if os.path.exists(path):
            return path
        with tempfile.TemporaryDirectory(dir=d) as tmp:
            src = os.path.join(tmp, "expr.cpp")
            lib = os.path.join(tmp, "expr.so")
            with open(src, "w") as f:
                f.write(source)
            proc = subprocess.run([cxx] + FLAGS + [src, "-o", lib],
                                  capture_output=True, text=True)
            if proc.returncode:
                raise ValueError("the C++ code does not compile:\n"
                                 + (proc.stderr.strip() or proc.stdout.strip()))
            os.replace(lib, path)       # atomic: no rank loads a partial file
        return path

    path = on_rank0(comm, build, "CompiledExpression")

    def load():
        lib = ctypes.CDLL(path)
        lib.eval.restype = None
        lib.eval.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_long,
                             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        return lib

    lib = on_every_rank(comm, load, "CompiledExpression: loading " + path, wait_for=path)
    _LIBRARIES[key] = lib
    return lib


class CompiledExpression:
    """A dolfin Expression compiled from C++ (see the module's docstring): `cpp_code` a
    string, or a tuple of them for a vector, the parameters by keyword. Calling it with
    points x (shape (3, n)) gives the values, shape (n,) or (components, n), as
    DOLFINx's Function.interpolate wants them."""

    def __init__(self, cpp_code, comm=None, **params):
        from mpi4py import MPI
        object.__setattr__(self, "codes",
                           [cpp_code] if isinstance(cpp_code, str) else list(cpp_code))
        object.__setattr__(self, "_names", list(params))
        object.__setattr__(self, "_values", np.array([float(v) for v in params.values()]))
        comm = MPI.COMM_WORLD if comm is None else comm
        object.__setattr__(self, "_lib", _compile(_source(self.codes, self._names),
                                                  comm if comm.size > 1 else None))

    @property
    def value_size(self):
        return len(self.codes)

    @property
    def parameters(self):
        return dict(zip(self._names, self._values.tolist()))

    def __getattr__(self, key):
        names = self.__dict__.get("_names", [])
        if key in names:
            return float(self._values[names.index(key)])
        raise AttributeError(key)

    def __setattr__(self, key, value):
        if key in self._names:
            self._values[self._names.index(key)] = float(value)
        else:
            raise AttributeError("CompiledExpression has no parameter {!r}".format(key))

    def __call__(self, x):
        X = np.ascontiguousarray(np.asarray(x, dtype=np.float64).T)    # (n, 3)
        n = X.shape[0]
        out = np.empty((n, self.value_size))
        p = self._values if len(self._values) else np.zeros(1)
        dp = ctypes.POINTER(ctypes.c_double)
        self._lib.eval(X.ctypes.data_as(dp), n, p.ctypes.data_as(dp), out.ctypes.data_as(dp))
        return out[:, 0].copy() if self.value_size == 1 else np.ascontiguousarray(out.T)
