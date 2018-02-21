#!/usr/bin/env python

import os
import sys

from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

readme = open('README.rst').read()

packages = ['PynPoint',
            'PynPoint.Core',
            'PynPoint.IOmodules',
            'PynPoint.OldVersion',
            'PynPoint.ProcessingModules',
            'PynPoint.Util',
            'PynPoint.Wrapper']

setup(
    name='PynPoint-exoplanet',
    version='1.0.0',
    description='Python package for processing and analyzing of high-contrast imaging data',
    long_description=readme,
    author='Tomas Stolker, Markus Bonse, Adam Amara',
    author_email='tomas.stolker@phys.ethz.ch, mbonse@tuebingen.mpg.de, adam.amara@phys.ethz.ch',
    url='http://pynpoint.ethz.ch',
    packages=packages,
    package_dir={'PynPoint': 'PynPoint'},
    include_package_data=True,
    install_requires=['astropy'],
    license='GPLv3',
    zip_safe=False,
    keywords='PynPoint',
    entry_points={'console_scripts': ['PynPoint = PynPoint._Cli:run',]},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    tests_require=['pytest>=2.3'],
    cmdclass = {'test': PyTest},
)
