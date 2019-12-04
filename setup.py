# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys
import platform
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel
from distutils.version import LooseVersion

__version__ = '0.3.4'


cpp_extension = False
cxx_compiler = None
cmake_options = None

class BDistWheelCommand(bdist_wheel):
    user_options = install.user_options + [
        ('cpp-extension', None, 'Compile the C++ hub extension'),
        ('cxx-compiler=', None, 'Path to the C++ compiler'),
        ('cmake-options=', None, 'Options to pass to cmake')
    ]

    def initialize_options(self):
        bdist_wheel.initialize_options(self)
        self.cpp_extension = False
        self.cxx_compiler = None
        self.cmake_options = None

    def finalize_options(self):
        global cpp_extension, cxx_compiler, cmake_options
        bdist_wheel.finalize_options(self)
        cpp_extension = cpp_extension or self.cpp_extension
        cxx_compiler = self.cxx_compiler if self.cxx_compiler is not None else cxx_compiler
        cmake_options = self.cmake_options if self.cmake_options is not None else cmake_options

    def run(self):
        bdist_wheel.run(self)

class InstallCommand(install):
    user_options = install.user_options + [
        ('cpp-extension', None, 'Compile the C++ hub extension'),
        ('cxx-compiler=', None, 'Path to the C++ compiler'),
        ('cmake-options=', None, 'Options to pass to cmake')
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.cpp_extension = False
        self.cxx_compiler = None
        self.cmake_options = None

    def finalize_options(self):
        global cpp_extension, cxx_compiler, cmake_options
        install.finalize_options(self)
        cpp_extension = cpp_extension or self.cpp_extension
        cxx_compiler = self.cxx_compiler if self.cxx_compiler is not None else cxx_compiler
        cmake_options = self.cmake_options if self.cmake_options is not None else cmake_options

    def run(self):
        install.run(self)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):

    user_options = build_ext.user_options + [
        ('cpp-extension', None, 'Compile the C++ hub extension'),
        ('cxx-compiler=', None, 'Path to the C++ compiler'),
        ('cmake-options=', None, 'Options to pass to cmake')
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.cpp_extension = False
        self.cxx_compiler = None
        self.cmake_options = None
    
    def finalize_options(self):
        global cpp_extension, cxx_compiler, cmake_options
        build_ext.finalize_options(self)
        cpp_extension = cpp_extension or self.cpp_extension
        cxx_compiler = self.cxx_compiler if self.cxx_compiler is not None else cxx_compiler
        cmake_options = self.cmake_options if self.cmake_options is not None else cmake_options

    def run(self):
        global cpp_extension
        
        if cpp_extension:
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
        global cxx_compiler, cmake_options

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DONLY_PYTHON=ON']
        
        if cxx_compiler is not None:
            cmake_args += ['-DCMAKE_CXX_COMPILER=' + cxx_compiler]
        
        if cmake_options is not None:
            cmake_args += [cmake_options]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
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


def find_version(*filepath):
    # Extract version information from filepath
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, *filepath)) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


extras_require = {
    'domains': [
        'gym>=0.13.0',
        'numpy>=1.16.4',
        'matplotlib>=3.1.0'
    ],
    'solvers': [
        'gym>=0.13.0',
        'numpy>=1.16.4',
        'joblib>=0.13.2',
        'stable-baselines~=2.6',
        'ray[rllib,debug]~=0.7.3'
    ]
}

# Add 'all' to extras_require to install them all at once
all_extra = []
for extra in extras_require.values():
    all_extra += extra
extras_require['all'] = all_extra


# TODO: import the following from python/setup.py
setup(
    name='airlaps',
    version=__version__,
    package_dir={'': 'python'},
    packages=find_packages(where='./python'),
    install_requires=[
       'simplejson>=3.16.0'
    ],
    url='www.airbus.com',
    license='MIT',
    author='Airbus',
    description='AIRLAPS is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.',
    entry_points={
        'airlaps.domains': [
            'GymDomain = airlaps.hub.domain.gym:GymDomain [domains]',
            'DeterministicGymDomain = airlaps.hub.domain.gym:DeterministicGymDomain [domains]',
            'CostDeterministicGymDomain = airlaps.hub.domain.gym:CostDeterministicGymDomain [domains]',
            'GymPlanningDomain = airlaps.hub.domain.gym:GymPlanningDomain [domains]',
            'GymWidthPlanningDomain = airlaps.hub.domain.gym:GymWidthPlanningDomain [domains]',
            'MasterMind = airlaps.hub.domain.mastermind:MasterMind [domains]',
            'Maze = airlaps.hub.domain.maze:Maze [domains]',
            'RockPaperScissors = airlaps.hub.domain.rock_paper_scissors:RockPaperScissors [domains]',
            'SimpleGridWorld = airlaps.hub.domain.simple_grid_world:SimpleGridWorld [domains]'
        ],
        'airlaps.solvers': [
            'AOstar = airlaps.hub.solver.aostar:AOstar',
            'Astar = airlaps.hub.solver.astar:Astar',
            'AugmentedRandomSearch = airlaps.hub.solver.ars:AugmentedRandomSearch [solvers]',
            'BFWS = airlaps.hub.solver.bfws:BFWS',
            'CGP = airlaps.hub.solver.cgp:CGP [solvers]',
            'IW = airlaps.hub.solver.iw:IW',
            'LazyAstar = airlaps.hub.solver.lazy_astar:LazyAstar',
            'POMCP = airlaps.hub.solver.pomcp:POMCP',
            'RayRLlib = airlaps.hub.solver.ray_rllib:RayRLlib [solvers]',
            'SimpleGreedy = airlaps.hub.solver.simple_greedy:SimpleGreedy',
            'StableBaseline = airlaps.hub.solver.stable_baselines:StableBaseline [solvers]'
        ]
    },
    extras_require=extras_require,
    ext_modules=[CMakeExtension(name='airlaps/hub/', sourcedir='cpp')],
    cmdclass=dict(build_ext=CMakeBuild, install=InstallCommand, bdist_wheel=BDistWheelCommand)
)
