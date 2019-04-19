# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='sentiment',
    version='0.1.0',
    description='Text Analytics API',
    long_description=readme,
    author='Vo Tam Van',
    author_email='vtvan2k1@gmail.com',
    url='https://github.com/votamvan/text-analytics',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'resources', 'notebooks', 'scripts'))
)

