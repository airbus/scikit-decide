from setuptools import setup

setup(
    name='airlaps',
    version='0.1.0',
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
        'airlaps.wrappers.solver',
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
    description='AIRLAPS is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.'
)
