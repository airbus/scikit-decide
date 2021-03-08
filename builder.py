"""
Adapted from https://github.com/pybind/cmake_example
"""
import fileinput
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
from distutils.version import LooseVersion
from typing import Any, Dict, List

from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


class CMakeExtension(Extension):
    def __init__(
        self,
        name: str,
        install_prefix: str,
        disable_editable: bool = False,
        cmake_configure_options: List[str] = (),
        source_dir: str = str(Path(".").absolute()),
        cmake_build_type: str = "Release",
        cmake_component: str = None,
        cmake_depends_on: List[str] = (),
    ) -> None:

        """
        Custom setuptools extension that configures a CMake project.
        Args:
            name: The name of the extension.
            install_prefix: The path relative to the site-package directory where the CMake
                project is installed (typically the name of the Python package).
            disable_editable: Skip this extension in editable mode.
            source_dir: The location of the main CMakeLists.txt.
            cmake_build_type: The default build type of the CMake project.
            cmake_component: The name of component to install. Defaults to all
                components.
            cmake_depends_on: List of dependency packages containing required CMake projects.
        """

        super().__init__(name=name, sources=[])

        if not Path(source_dir).is_absolute():
            source_dir = str(Path(".").absolute() / source_dir)

        if not Path(source_dir).absolute().is_dir():
            raise ValueError(f"Directory '{source_dir}' does not exist")

        self.install_prefix = install_prefix
        self.cmake_build_type = cmake_build_type
        self.disable_editable = disable_editable
        self.cmake_depends_on = cmake_depends_on
        self.source_dir = str(Path(source_dir).absolute())
        self.cmake_configure_options = cmake_configure_options
        self.cmake_component = cmake_component


class ExtensionBuilder(build_ext):
    def run(self) -> None:
        self.validate_cmake()
        super().run()

    def build_extension(self, ext: Extension) -> None:
        if isinstance(ext, CMakeExtension):
            self.build_cmake_extension(ext)
        else:
            super().build_extension(ext)

    def validate_cmake(self) -> None:
        cmake_extensions = [x for x in self.extensions if isinstance(x, CMakeExtension)]
        if len(cmake_extensions) > 0:
            try:
                out = subprocess.check_output(["cmake", "--version"])
            except OSError:
                raise RuntimeError(
                    "CMake must be installed to build the following extensions: "
                    + ", ".join(e.name for e in cmake_extensions)
                )
            if platform.system() == "Windows":
                cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))  # type: ignore
                if cmake_version < "3.1.0":
                    raise RuntimeError("CMake >= 3.1.0 is required on Windows")

    def build_cmake_extension(self, ext: CMakeExtension) -> None:
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DONLY_PYTHON=ON",
        ]

        cfg = "Debug" if self.debug else "Release"
        # cfg = 'Debug'
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            cmake_args += [
                "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j4"]
        cmake_args += ["-DPYTHON_INCLUDE_DIR={}".format(sysconfig.get_path("include"))]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.source_dir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


def build(setup_kwargs: Dict[str, Any]) -> None:
    cython_modules = []

    cmake_modules = [
        CMakeExtension(name="skdecide/hub/", source_dir="cpp", install_prefix="build")
    ]
    ext_modules = cython_modules + cmake_modules

    f = "cpp/deps/gecode/CMakeLists.txt"
    if os.path.isfile(f):
        with fileinput.FileInput(os.path.abspath(f), inplace=True) as file:
            for line in file:
                print(
                    line.replace(
                        "CMAKE_RUNTIME_OUTPUT_DIRECTORY",
                        "_CMAKE_RUNTIME_OUTPUT_DIRECTORY_",
                    ),
                    end="",
                )

    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": dict(build_ext=ExtensionBuilder),
            "zip_safe": False,
        }
    )
