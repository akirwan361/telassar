#!/usr/bin/env python
"""
Modified from:
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, "README.md"), encoding='utf-8' as dfile:
    long_description = dfile.read()

setup(
    name='telassar',  # Required
    version='0.0.1',  # Required
    description='Two dimEnsionaL spectrAl analysiS for muSe And xshooteR',  # Optional
    long_description=long_description,  # Optional
    url='https://github.com/amiller361/telassar',  # Optional
    author='Andrew Miller',  # Optional
    author_email='andrew.miller@mu.ie',  # Optional
    keywords='astronomy, spectroscopy, astrophysics',  # Optional
    package_dir={'': 'telassar'}
    packages=find_packages('telassar')
    python_requires='>=3.6',
    install_requires=[
        'astropy',
        'matplotlib',
        'lmfit',
        'photutils',
    ],  # Optional

    # include sample data?
    data_files=[('hd163296_pvslice.fits', ['data/hd163296_pvslice.fits'])],        

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/amiller361/telassar/issues',
#        'Say Thanks!': 'http://saythanks.io/to/example',
        'Source': 'https://github.com/amiller361/telassar/',
    },
)
