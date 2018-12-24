#!/usr/bin/env python

from setuptools import setup

try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements

from pynpoint import __author__, __version__, __license__, __email__

reqs = parse_requirements('requirements.txt', session='hack')
reqs = [str(ir.req) for ir in reqs]

setup(
    name='pynpoint',
    version=__version__,
    description='Pipeline for processing and analysis of high-contrast imaging data',
    long_description=open('README.rst').read(),
    author=__author__,
    author_email=__email__,
    url='http://pynpoint.ethz.ch',
    packages=['pynpoint',
              'pynpoint.core',
              'pynpoint.readwrite',
              'pynpoint.processing',
              'pynpoint.util'],
    include_package_data=True,
    install_requires=reqs,
    license=__license__,
    zip_safe=False,
    keywords='pynpoint',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    tests_require=['pytest'],
)
