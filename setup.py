# -*- coding: utf-8 -*-

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '2.0.9'
PACKAGE_NAME = 'FuzzyTM'
AUTHOR = 'Emil Rijcken'
AUTHOR_EMAIL = 'emil.rijcken@gmail.com'
URL = 'https://github.com/ERijck/FuzzyTM'

LICENSE = 'GNU General Public License v3.0'
DESCRIPTION = 'A Python package for Fuzzy Topic Models'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'pandas',
      'scipy',
      'pyfume',
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )