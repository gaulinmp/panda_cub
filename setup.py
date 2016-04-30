"""Panda Cub is a toolkit for working with pandas and other scientific packages.

COPYRIGHT: MIT
"""

import os
import re
from codecs import open

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Get the version from panda_cub/__init__.py
_version_re = re.compile(r'^__version__\s*=\s*(.*)$', re.M)

with open('panda_cub/__init__.py', 'r') as fh:
    version_re = _version_re.search(fh.read())
    if version_re:
        version = version_re.group(1)
    else:
        version = "0.0.0dev1"

setup(
    name='panda_cub',
    version=version,
    description='Python toolkit for data analysis code snippets',
    long_description=long_description,
    keywords='pandas, dataframe, data analysis',
    url='https://github.com/gaulinmp/panda_cub',
    author='Mac Gaulin',
    author_email='git@mgaulin.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    packages=['panda_cub',],
    install_requires=['numpy', 'scipy', 'pandas>=0.10'],
    extras_require={
        # 'dev': ['statsmodel'],
        # 'test': ['coverage'],
    },
)
