from setuptools import setup, find_packages

setup(
    name='telassar',
    version='0.0.1',
    author='Andrew Miller',
    author_email='andrew.miller@mu.ie',
    description='Two dimEnsionaL spectrAl analysiS for muSe '
    'And xshooteR is a hideous monstrosity for 2D astro data',
    python_requires='>=3.6',
    install_requires=[
        'requests',
        'numpy>=1.10.0',
        'scipy',
        'matplotlib',
        'astropy>=1.0',
        'lmfit>=1.0.0'
    ],
    package_dir={"": "telassar"},
    packages=find_packages(where='telassar'),
)
