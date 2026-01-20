from setuptools import setup, find_packages

setup(
    name='nsflow',
    version='0.1.0',
    packages=find_packages(exclude=['data'])
)
