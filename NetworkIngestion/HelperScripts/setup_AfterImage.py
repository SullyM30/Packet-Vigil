"""
Build Cython-optimized AfterImage module.

Usage:
    python setup.py build_ext --inplace

This compiles NetworkIngestion/AfterImage_cy.pyx into a shared library
(.pyd on Windows, .so on Linux) that can be imported directly.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "NetworkIngestion.AfterImage_cy",
        ["NetworkIngestion/AfterImage_cy.pyx"],
    ),
]

setup(
    name="AfterImage_cy",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    ),
)
