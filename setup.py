#!/usr/bin/env python

from setuptools import setup

from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements

reqs = parse_requirements('requirements.txt', session=PipSession())
reqs = [str(req.requirement) for req in reqs]

setup(
    name='pynpoint',
    version='0.10.0',
    description='Pipeline for processing and analysis of high-contrast imaging data',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    author='Tomas Stolker & Markus Bonse',
    author_email='stolker@strw.leidenuniv.nl',
    url='https://github.com/PynPoint/PynPoint',
    project_urls={'Documentation': 'https://pynpoint.readthedocs.io'},
    packages=['pynpoint',
              'pynpoint.core',
              'pynpoint.readwrite',
              'pynpoint.processing',
              'pynpoint.util'],
    include_package_data=True,
    install_requires=reqs,
    license='GPLv3',
    zip_safe=False,
    keywords='pynpoint',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    tests_require=['pytest'],
)
