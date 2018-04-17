#!/usr/bin/env python

import sys
from setuptools import setup

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
                'matplotlib',
                'emcee',
                'ephem']

setup(
    name='PynPoint',
    version='0.4.0',
    description='Python package for processing and analysis of high-contrast imaging data',
    long_description=readme,
    author='Tomas Stolker, Markus Bonse, Sascha Quanz, Adam Amara',
    author_email='tomas.stolker@phys.ethz.ch',
    url='http://pynpoint.ethz.ch',
    packages=packages,
    package_dir={'PynPoint': 'PynPoint'},
    include_package_data=True,
    install_requires=requirements,
    license='GPLv3',
    zip_safe=False,
    keywords='PynPoint',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy'
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    tests_require=['pytest'],
)
