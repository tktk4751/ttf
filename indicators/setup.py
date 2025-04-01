from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Cythonモジュールの設定
extensions = [
    Extension(
        "kernel_ma_cy",
        ["kernel_ma_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

# セットアップ設定
setup(
    name="kernel_ma_optimized",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
    ),
    include_dirs=[np.get_include()],
) 