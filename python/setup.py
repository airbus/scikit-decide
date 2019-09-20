from setuptools import setup, find_packages

__version__ = '0.3.4'

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

# Setup definition
setup(
    name='airlaps',
    version=__version__,
    packages=find_packages(),
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
    extras_require=extras_require
)
