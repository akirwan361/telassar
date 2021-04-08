#!/usr/bin/env python

from setuptools import setup
from configparser import ConfigParser
import sys

conf = ConfigParser()
conf.read(["setup.cfg"])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata.get('package_name', 'packagename')
DESCRIPTION = metadata.get('description', '')
AUTHOR = metadata.get('author', '')
AUTHOR_EMAIL = metadata.get('author_email', '')
LICENSE = metadata.get('license', '')

__import__(PACKAGENAME)
package = sys.modules[PACKAGENAME]
LONG_DESCRIPTION = package.__doc__

VERSION = '0.0.1'

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      install_requires=['astropy', 'matplotlib', 'lmfit'],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,
)
