"""Build configuration for the optional native MARL2D C extension.

The project can run many development workflows without compiling native code, so the
extension is marked optional to keep editable installs usable on machines that do not have
`gcc` available. Training and simulation still require the compiled module; the runtime
fallback in `puffer_soccer.envs.marl2d.csrc.binding` raises a clear message when native code
is missing.
"""

from setuptools import Extension, setup
import numpy

ext_modules = [
    Extension(
        "puffer_soccer.envs.marl2d.csrc.binding",
        sources=["src/puffer_soccer/envs/marl2d/csrc/binding.c"],
        include_dirs=[numpy.get_include(), "src/puffer_soccer/envs/marl2d/csrc"],
        extra_compile_args=["-O3"],
        optional=True,
    )
]

setup(ext_modules=ext_modules)
