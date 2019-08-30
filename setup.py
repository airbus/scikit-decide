import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DONLY_PYTHON=ON']
        
        if os.path.isfile('airlaps_cmake_toolchain.txt'):
            cmake_args += ['-DCMAKE_TOOLCHAIN_FILE=' + os.path.abspath('airlaps_cmake_toolchain.txt')]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# TODO: import the following from python/setup.py
setup(
    name='airlaps',
    version='0.2.1',
    package_dir = {'': 'python'},
    packages=[
        'airlaps',
        'airlaps.catalog',
        'airlaps.catalog.domain',
        'airlaps.catalog.solver',
        'airlaps.builders',
        'airlaps.builders.domain',
        'airlaps.builders.solver',
        'airlaps.wrappers',
        'airlaps.wrappers.space',
        'airlaps.wrappers.domain',
        'airlaps.wrappers.solver'
    ],
    extras_require={
        'wrappers': [
            'gym==0.13.0',
            'stable-baselines==2.6.0',
            'tensorflow==1.14.0'
        ],
    },
    url='www.airbus.com',
    license='MIT',
    author='Airbus',
    description='AIRLAPS is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.',
    ext_modules=[CMakeExtension(name='airlaps/airlaps', sourcedir='cpp')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
