from setuptools import setup

setup(
    name='airlaps',
    version='0.2.1',
    packages=[
        'airlaps',
        'airlaps.builders',
        'airlaps.builders.domain',
        'airlaps.builders.solver'
    ],
    url='www.airbus.com',
    license='MIT',
    author='Airbus',
    description='AIRLAPS is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.'
)
