from distutils.core import setup

setup(
    name='ncpol2sdpa',
    version='1.0.0',
    author='P. Wittek',
    packages=['ncpol2sdpa'],
    url='http://peterwittek.github.io/ncpol2sdpa/',
    license='LICENSE',
    description='A converter from noncommutative polynomial optimization problems to sparse SDPA input format.',
    long_description=open('README.md').read(),
    install_requires=[
        "sympy >= 0.7.2"
    ],
) 
