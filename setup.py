from setuptools import Extension, setup
import numpy

ext_modules = [
    Extension(
        "puffer_soccer.envs.marl2d.csrc.binding",
        sources=["src/puffer_soccer/envs/marl2d/csrc/binding.c"],
        include_dirs=[numpy.get_include(), "src/puffer_soccer/envs/marl2d/csrc"],
        extra_compile_args=["-O3"],
    )
]

setup(ext_modules=ext_modules)
