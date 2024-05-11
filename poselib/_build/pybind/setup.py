from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys
import os

__version__ = "2.0.0"

ext_modules = [
    Pybind11Extension("poselib",[os.path.join("/media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib",
                                              "pybind",
                                              "pyposelib.cpp")],
    extra_objects=[os.path.join("/media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_install",
                                "lib",
                                "libPoseLib.a")],
    extra_compile_args=['-std=c++17','-march=native','-ffast-math','-fno-unsafe-math-optimizations']),
]

setup(
    name="poselib",
    version=__version__,
    include_dirs=[os.path.join("/media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_install",
                               "include"),
                  "/usr/local/include/eigen3"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
