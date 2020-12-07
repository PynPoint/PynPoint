#!/usr/bin/env python

from setuptools import setup

import pathlib
import pkg_resources

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='pynpoint',
    version='0.8.3',
    description='Pipeline for processing and analysis of high-contrast imaging data',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    author='Tomas Stolker & Markus Bonse',
    author_email='tomas.stolker@phys.ethz.ch',
    url='https://github.com/PynPoint/PynPoint',
    project_urls={'Documentation': 'https://pynpoint.readthedocs.io'},
    packages=['pynpoint',
              'pynpoint.core',
              'pynpoint.readwrite',
              'pynpoint.processing',
              'pynpoint.util'],
    include_package_data=True,
    install_requires=install_requires,
    license='GPLv3',
    zip_safe=False,
    keywords='pynpoint',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    tests_require=['pytest'],
)
