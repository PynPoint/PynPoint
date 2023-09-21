#!/usr/bin/env python

import pkg_resources
import setuptools

with open('requirements.txt') as req_txt:
    parse_req = pkg_resources.parse_requirements(req_txt)
    install_requires = [str(req) for req in parse_req]

setuptools.setup(
    name='pynpoint',
    version='0.11.0',
    description='Pipeline for processing and analysis of high-contrast imaging data',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    author='Tomas Stolker & Markus Bonse',
    author_email='stolker@strw.leidenuniv.nl',
    url='https://github.com/PynPoint/PynPoint',
    project_urls={'Documentation': 'https://pynpoint.readthedocs.io'},
    packages=setuptools.find_packages(include=['pynpoint', 'pynpoint.*']),
    install_requires=install_requires,
    tests_require=['pytest'],
    license='GPLv3',
    zip_safe=False,
    keywords='pynpoint',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
