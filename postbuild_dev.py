"""Post-build script to be run to setup developer environment.

After `poetry install`, some further steps are required to have the poetry environment working properly,
similarly to a normal install from pypi.

Currently, this steps are:
- finish minizinc setup: the embedded minizinc needs the gecode.msc configuration file
  to be placed near the corresponding executables and libraries (as paths within are relative to it)

"""

import os
import shutil

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


def configure_minizinc():
    print("Finish minizinc configuration")

    class ExtensionBuilder(build_ext):
        def run(self):
            print(f"Copying gecode.msc to {self.build_lib} directory")
            root_dir = os.path.dirname(os.path.abspath(__file__))
            skdecidehub_relativepath = "skdecide/hub"
            gecodeconfig_filename = "gecode.msc"
            gecodeconfig_retativepath = (
                f"{skdecidehub_relativepath}/{gecodeconfig_filename}"
            )
            srcfile = f"{root_dir}/{gecodeconfig_retativepath}"
            detsfile = f"{root_dir}/{self.build_lib}/{gecodeconfig_retativepath}"
            shutil.copyfile(srcfile, detsfile)

    setup(
        cmdclass=dict(build_ext=ExtensionBuilder),
        ext_modules=[Extension(name="", sources=[""])],
        zip_safe=False,
        script_args=["build"],
    )


if __name__ == "__main__":
    configure_minizinc()
