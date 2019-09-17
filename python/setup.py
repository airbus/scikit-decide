from setuptools import setup


__version__ = "0.2.3"


setup(
    name='airlaps',
    version=__version__,
    packages=[
        'airlaps',
        'airlaps.builders',
        'airlaps.builders.domain',
        'airlaps.builders.solver'
    ],
    install_requires=[
       'simplejson>=3.16.0'
    ],
    url='www.airbus.com',
    license='MIT',
    author='Airbus',
    description='AIRLAPS is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.'
)
