"""
Ncpol2SDPA: A converter from commutative and noncommutative polynomial
optimization problems to sparse SDP input formats.
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
setup(
    name='ncpol2sdpa',
    version='1.11.0',
    author='Peter Wittek',
    author_email='peterwittek@users.noreply.github.com',
    packages=['ncpol2sdpa'],
    url='http://ncpol2sdpa.readthedocs.io/',
    keywords=[
        'sdp',
        'semidefinite programming',
        'relaxation',
        'polynomial optimization problem',
        'noncommuting variable',
        'sdpa'],
    license='LICENSE',
    description='A converter from polynomial optimization problems of\
                 noncommutative variables to sparse SDPA input format.',
    long_description=open('README.rst').read(),
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python'
    ],
    install_requires=[
        "sympy >= 0.7.2",
        "numpy"
    ],
    test_suite="tests"
)
