"""
Modified from:
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


setup(
    name='telassar',  # Required
    version='0.0.1',  # Required
    description='Two dimEnsionaL spectrAl analysiS for muSe And xshooteR',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/amiller361/telassar',  # Optional
    author='Andrew Miller',  # Optional
    author_email='andrew.miller@mu.ie',  # Optional
    keywords='astronomy, spectroscopy, astrophysics',  # Optional
    package_dir={'': 'telassar'},  # Optional
    packages=find_packages(where='telassar'),  # Required
    python_requires='>=3.5',
    install_requires=[
        'astropy',
        'matplotlib',
        'lmfit',
        'photutils',
    ],  # Optional

    # include sample data?
    data_files=[('dgtau_OI6300_pvslice.fits', ['data/dgtau_OI6300_pvslice.fits'])],        
#    entry_points={  # Optional
#        'console_scripts': [
#            'sample=sample:main',
#        ],
#    },

     project_urls={  # Optional
        'Bug Reports': 'https://github.com/amiller361/telassar/issues',
#        'Say Thanks!': 'http://saythanks.io/to/example',
        'Source': 'https://github.com/amiller361/telassar/',
    },
)
