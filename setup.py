#!/usr/bin/env python
from setuptools import setup

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
    name = 'rnadeep',
    version = '0.1',
    description = 'Training repository for RNA folding',
    long_description = LONG_DESCRIPTION,
    long_description_content_type = 'text/markdown',
    license = 'MIT',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Science/Research',
        ],
    python_requires = '>=3.7',
    install_requires = [
        'numpy',
        'tensorflow'],
    packages = ['rnadeep']
)

