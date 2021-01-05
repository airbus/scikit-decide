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

################################################################################
# Version, create_version_file, and package_name
################################################################################
#version = open('version.txt', 'r').read().strip()

def _get_version_hash():
    """Talk to git and find out the tag/hash of our latest commit"""
    try:
        ver = subprocess.check_output(["git", "describe", "--tags", "--always"], encoding="utf-8")
    except OSError:
        print("Couldn't run git to get a version number for setup.py")
        return
    return ver.strip()

version = _get_version_hash()

if version[:1] == 'v':
    version = version[1:]

print("version: {}".format(version))

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
        'tensorflow>=1.8.0,<2.0.0',
        'stable-baselines>=2.6',
        'ray[rllib,debug]>=0.8.6'
    ],
    # TODO add do dependancies
    'discrete_optimization':
    [
        'shapely>=1.7',
        'mip==1.9',
        'minizinc>=0.3',
        'deap>=1.3',
        'networkx>=2.4',
        'numba>=0.50',
        'matplotlib>=3.1',
        "seaborn>=0.10.1",
        "pymzn>=0.18.3",
        "ortools>=8.0"
    ]
}

# Add 'all' to extras_require to install them all at once

print(sys.platform)

if sys.platform == "win32":
    extras_require = {
        'domains': [
            'gym>=0.13.0',
            'numpy>=1.16.4',
            'matplotlib>=3.1.0',
            'simplejson>=3.16.0'
        ],
        'solvers': [
            'gym>=0.13.0',
            'numpy>=1.16.4',
            'joblib>=0.13.2',
            'ray[rllib,debug]>=1.0.0',
            'stable-baselines3>=0.9.0'
        ],
        'discrete_optimization':
            [
                'shapely>=1.7',
                'mip==1.9',
                'minizinc>=0.3',
                'deap>=1.3',
                'networkx>=2.4',
                'numba>=0.50',
                'matplotlib>=3.1',
                "seaborn>=0.10.1",
                "pymzn>=0.18.3",
                "ortools>=8.0"
            ]
    }
else:
    extras_require = {
        'domains': [
            'gym>=0.13.0',
            'numpy>=1.16.4',
            'matplotlib>=3.1.0',
            'simplejson>=3.16.0'
        ],
        'solvers': [
            'gym>=0.13.0',
            'numpy>=1.16.4',
            'joblib>=0.13.2',
            'ray[rllib,debug]>=1.0.0',
            'torch==1.6.0',
            'stable-baselines3>=0.9.0'
        ],
        'discrete_optimization':
            [
                'shapely>=1.7',
                'mip==1.9',
                'minizinc>=0.3',
                'deap>=1.3',
                'networkx>=2.4',
                'numba>=0.50',
                'matplotlib>=3.1',
                "seaborn>=0.10.1",
                "pymzn>=0.18.3",
                "ortools>=8.0"
            ]
    }

all_extra = []
for extra in extras_require.values():
    all_extra += extra
extras_require['all'] = all_extra
from pathlib import Path
data_packages = ['{}'.format(p).replace('/', '.')
                 for p in list(Path('skdecide/builders/discrete_optimization/data').glob('**'))
                 + list(Path('skdecide/builders/discrete_optimization/').glob('**/minizinc'))]

with open('requirements.txt') as f:
    INSTALL_REQUIRES = []
    for l in f:
        if l:
            l = l.strip()
            i = l.find('-f')
            if i != -1:
                l = l[:i]
            else:
                INSTALL_REQUIRES.append(l)

with open('requirements-dev.txt') as f:
    TEST_REQUIRES = [l.strip() for l in f.readlines() if l]

print(platform.system(), INSTALL_REQUIRES)
print(platform.system(), TEST_REQUIRES)
print(platform.system(), extras_require)


setup(
    name='scikit-decide',
    version=version,
    packages=find_packages()+data_packages,
    include_package_data=True,
    package_data = {'': ['*']},
    install_requires=INSTALL_REQUIRES,
    setup_requires=[],
    tests_require=TEST_REQUIRES,
    url='https://github.com/airbus/scikit-decide',
    license='MIT',
    author='Airbus',
    description='Scikit-decide is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.',
    entry_points={
        'skdecide.domains': [
            'GymDomain = skdecide.hub.domain.gym:GymDomain [domains]',
            'DeterministicGymDomain = skdecide.hub.domain.gym:DeterministicGymDomain [domains]',
            'CostDeterministicGymDomain = skdecide.hub.domain.gym:CostDeterministicGymDomain [domains]',
            'GymPlanningDomain = skdecide.hub.domain.gym:GymPlanningDomain [domains]',
            'GymWidthPlanningDomain = skdecide.hub.domain.gym:GymWidthPlanningDomain [domains]',
            'MasterMind = skdecide.hub.domain.mastermind:MasterMind [domains]',
            'Maze = skdecide.hub.domain.maze:Maze [domains]',
            'RockPaperScissors = skdecide.hub.domain.rock_paper_scissors:RockPaperScissors [domains]',
            'SimpleGridWorld = skdecide.hub.domain.simple_grid_world:SimpleGridWorld [domains]'
        ],
        'skdecide.solvers': [
            'AOstar = skdecide.hub.solver.aostar:AOstar',
            'Astar = skdecide.hub.solver.astar:Astar',
            'LRTAstar = skdecide.hub.solver.lrtastar:LRTAstar',
            'MCTS = skdecide.hub.solver.mcts:MCTS',
            'UCT = skdecide.hub.solver.mcts:UCT',
            'AugmentedRandomSearch = skdecide.hub.solver.ars:AugmentedRandomSearch [solvers]',
            'BFWS = skdecide.hub.solver.bfws:BFWS',
            'CGP = skdecide.hub.solver.cgp:CGP [solvers]',
            'IW = skdecide.hub.solver.iw:IW',
            'RIW = skdecide.hub.solver.riw:RIW',
            'LRTDP = skdecide.hub.solver.lrtdp:LRTDP',
            'ILAOstar = skdecide.hub.solver.ilaostar:ILAOstar',
            'LazyAstar = skdecide.hub.solver.lazy_astar:LazyAstar',
            'MaxentIRL = skdecide.hub.solver.maxent_irl:MaxentIRL [solvers]',
            'POMCP = skdecide.hub.solver.pomcp:POMCP',
            'RayRLlib = skdecide.hub.solver.ray_rllib:RayRLlib [solvers]',
            'SimpleGreedy = skdecide.hub.solver.simple_greedy:SimpleGreedy',
            'StableBaseline = skdecide.hub.solver.stable_baselines:StableBaseline [solvers]'
        ]
    },
    extras_require=extras_require,
    ext_modules=[CMakeExtension(name='skdecide/hub/', sourcedir='cpp')],
    cmdclass=dict(build_ext=CMakeBuild, install=InstallCommand, bdist_wheel=BDistWheelCommand)
)
