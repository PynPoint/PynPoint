PynPoint
========

**Python package for the processing and analysis of high-contrast imaging data**

.. image:: https://img.shields.io/badge/GitHub-PynPoint-blue.svg?style=flat
    :target: https://github.com/PynPoint/PynPoint

.. image:: https://travis-ci.org/PynPoint/PynPoint.svg?branch=master
    :target: https://travis-ci.org/PynPoint/PynPoint

.. image:: https://img.shields.io/badge/Python-2.7-yellow.svg?style=flat
    :target: https://pypi.python.org/pypi/PynPoint-exoplanet

.. image:: https://img.shields.io/aur/license/yaourt.svg?style=flat
    :target: https://github.com/PynPoint/PynPoint/blob/master/LICENSE

.. image:: http://img.shields.io/badge/arXiv-1207.6637-orange.svg?style=flat
    :target: http://arxiv.org/abs/1207.6637


PynPoint is an end-to-end pipeline for the data reduction of high-contrast imaging data of planetary and substellar companions, as well as circumstellar disks in scattered light.

The pipeline has a modular architecture with a central data storage in which the reduction steps are stored by the processing modules. These modules have specific tasks such as the subtraction of the background, frame selection, centering, PSF subtraction, and photometric and astrometric measurements. The tags from the central data storage can be written to FITS, HDF5, and text files with the available IO modules.

PynPoint is under continuous development and the latest implementations can be pulled from Github repository. Bug reports, requests for new features, and contributions in the form of new processing modules are highly appreciated. Instructions for writing of modules are provided in the documentation. Bug reports and functionality requests can be provided by creating an `issue <https://github.com/PynPoint/PynPoint/issues>`_ on the Github page.

Documentation
-------------

Documentation can be found at `pythonhosted.org/PynPoint-exoplanet <http://pythonhosted.org/PynPoint-exoplanet/>`_, including installation instructions, details on the architecture of PynPoint, end-to-end examples for data obtained with dithering and nodding, and a description of the various processing modules and parameters.

Attribution
-----------

Please cite `Amara & Quanz (2012) <http://adsabs.harvard.edu/abs/2012MNRAS.427..948A>`_ when results obtained with PynPoint are published.

License
-------

Copyright 2014-2018 Tomas Stolker, Markus Bonse, Sascha Quanz, Adam Amara, and contributors.

PynPoint is free software and distributed under the GNU General Public License v3. See the LICENSE file for the terms and conditions.