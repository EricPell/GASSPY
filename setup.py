import setuptools
import sys
import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "gasspy.physics.databasing.cgasspy_tree",
        ["gasspy/physics/databasing/cgasspy_tree.cpp"]
    ),
]
print(ext_modules[0])
setup(ext_modules=ext_modules)

