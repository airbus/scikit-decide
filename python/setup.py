from setuptools import setup

setup(
    name='airlaps',
    version='0.0.2',
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
        'airlaps.wrappers.distribution'
        ],
    extras_require={
            'wrappers': [
                'scipy==1.1.0',
                'gym==0.12.1',
                'stable-baselines==2.5.0',
                'tensorflow==1.12.0'
            ],
    },
    url='www.airbus.com',
    license='NA',
    author='airbus',
    author_email='airlaps@airbus.com',
    description='AIRLAPS is an AI toolbox for Reinforcement Learning, Automated Planning and Scheduling.'
)
