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
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        cmake_install_prefix = ext_dir / ext.install_prefix

        configure_args = [
            f"-DCMAKE_BUILD_TYPE={ext.cmake_build_type}",
            f"-DCMAKE_INSTALL_PREFIX:PATH={cmake_install_prefix}",
        ]

        # Extend configure arguments with those from the extension
        configure_args += ext.cmake_configure_options

        # Set build arguments
        build_args = ["--config", ext.cmake_build_type]

        if platform.system() == "Windows":
            if sys.maxsize > 2 ** 32:
                configure_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            build_args += ["--", "-j4"]

        """
        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        """

        # Get the absolute path to the build folder
        build_folder = str(Path('.').absolute() / f"{self.build_temp}_{ext.name}")

        # Make sure the build folder exists
        Path(build_folder).mkdir(exist_ok=True, parents=True)

        configure_command = ['cmake', '-S', ext.source_dir, '-B', build_folder] + configure_args

        build_command = ['cmake', '--build', build_folder] + build_args

        install_command = ['cmake', '--install',  build_folder]
        if ext.cmake_component is not None:
            install_command.extend(['--component', ext.cmake_component])

        print(f"$ {' '.join(configure_command)}")
        print(f"$ {' '.join(build_command)}")
        print(f"$ {' '.join(install_command)}")

        # Generate the project
        subprocess.check_call(configure_command)
        # Build the project
        subprocess.check_call(build_command)
        # Install the project
        subprocess.check_call(install_command)

    def copy_extensions_to_source(self):
        original_extensions = list(self.extensions)
        self.extensions = [
            ext for ext in self.extensions
            if not isinstance(ext, CMakeExtension) or not ext.disable_editable
        ]
        super().copy_extensions_to_source()
        self.extensions = original_extensions


def build(setup_kwargs: Dict[str, Any]) -> None:
    cython_modules = []

    cmake_modules = [
        CMakeExtension(
            name="skdecide/hub/__skdecide_hub_cpp",
            source_dir="cpp",
            install_prefix="",
            cmake_configure_options=[
                           f"-DPYTHON_EXECUTABLE={Path(sys.executable)}",
                           f"-DONLY_PYTHON=ON",
                       ]),
    ]
    if 'SKDECIDE_SKIP_DEPS' not in os.environ or os.environ['SKDECIDE_SKIP_DEPS'] == '0':
        cmake_modules.extend([
            CMakeExtension(
                name="chuffed",
                disable_editable=True,
                source_dir="cpp/deps/chuffed",
                install_prefix="skdecide/hub",
                cmake_configure_options=[
                           f"-DBISON_EXECUTABLE=false",
                           f"-DFLEX_EXECUTABLE=false",
                           ]),
            CMakeExtension(
                name="gecode",
                disable_editable=True,
                source_dir="cpp/deps/gecode",
                install_prefix="skdecide/hub",
                cmake_configure_options=[
                           ]),
            CMakeExtension(
                name="libminizinc",
                disable_editable=True,
                source_dir="cpp/deps/libminizinc",
                install_prefix="skdecide/hub",
                cmake_configure_options=[
                               f"-DBUILD_SHARED_LIBS:BOOL=OFF",
                           ]),
        ])
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
