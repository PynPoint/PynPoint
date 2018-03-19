#!/usr/bin/env python

import os
import sys

from setuptools import setup

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist bdist_wheel upload')
    sys.exit()

readme = open('README.rst').read()

packages = ['PynPoint',
            'PynPoint.Core',
            'PynPoint.IOmodules',
            'PynPoint.OldVersion',
            'PynPoint.ProcessingModules',
            'PynPoint.Util',
            'PynPoint.Wrapper']

requirements = ['configparser',
                'h5py',
                'numpy',
                'numba',
                'scipy',
                'astropy',
                'photutils',
                'scikit-image',
                'scikit-learn',
                'opencv-python',
                'statsmodels',
                'PyWavelets',
                'mlpy',
                'matplotlib',
                'emcee']

setup(
    name='PynPoint',
    version='1.0.0',
    description='Python package for the processing and analysis of high-contrast imaging data',
    long_description=readme,
    author='Tomas Stolker, Markus Bonse, Sascha Quanz, Adam Amara',
    author_email='tomas.stolker@phys.ethz.ch, mbonse@tuebingen.mpg.de, sascha.quanz@phys.ethz.ch, adam.amara@phys.ethz.ch',
    url='http://pynpoint.ethz.ch',
    packages=packages,
    package_dir={'PynPoint': 'PynPoint'},
    include_package_data=True,
    install_requires=requirements,
    license='GPLv3',
    zip_safe=False,
    keywords='PynPoint',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy'
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    tests_require=['pytest'],
)
