PynPoint
========

**Python package for processing and analysis of high-contrast imaging data**

.. image:: https://badge.fury.io/py/pynpoint.svg
    :target: https://badge.fury.io/py/pynpoint

.. .. image:: https://img.shields.io/badge/GitHub-PynPoint-blue.svg?style=flat
..     :target: https://github.com/PynPoint/PynPoint

.. image:: https://travis-ci.org/PynPoint/PynPoint.svg?branch=master
    :target: https://travis-ci.org/PynPoint/PynPoint

.. .. image:: https://codecov.io/gh/PynPoint/PynPoint/branch/master/graph/badge.svg
..     :target: https://codecov.io/gh/PynPoint/PynPoint

.. image:: https://coveralls.io/repos/github/PynPoint/PynPoint/badge.svg?branch=master
    :target: https://coveralls.io/github/PynPoint/PynPoint?branch=master

.. image:: https://www.codefactor.io/repository/github/pynpoint/pynpoint/badge
    :target: https://www.codefactor.io/repository/github/pynpoint/pynpoint

.. image:: https://readthedocs.org/projects/pynpoint/badge/?version=latest
    :target: http://pynpoint.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Python-2.7-yellow.svg?style=flat
    :target: https://pypi.python.org/pypi/pynpoint

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://github.com/PynPoint/PynPoint/blob/master/LICENSE

.. image:: http://img.shields.io/badge/arXiv-1207.6637-orange.svg?style=flat
    :target: http://arxiv.org/abs/1207.6637

PynPoint is an end-to-end pipeline for the data reduction of high-contrast imaging data of planetary and substellar companions, as well as circumstellar disks in scattered light.

The pipeline has a modular architecture with a central data storage in which the reduction steps are stored by the processing modules. These modules have specific tasks such as the subtraction of the background, frame selection, centering, PSF subtraction, and photometric and astrometric measurements. The tags from the central data storage can be written to FITS, HDF5, and text files with the available IO modules.

PynPoint is under continuous development and the latest implementations can be pulled from Github repository. Bug reports, requests for new features, and contributions in the form of new processing modules are highly appreciated. Instructions for writing of modules are provided in the documentation. Bug reports and functionality requests can be provided by creating an `issue <https://github.com/PynPoint/PynPoint/issues>`_ on the Github page.

Documentation
-------------

Documentation can be found at `http://pynpoint.readthedocs.io <http://pynpoint.readthedocs.io>`_, including installation instructions, details on the architecture of PynPoint, and end-to-end example for data obtained with dithering, and a description of the various processing modules and parameters.

Attribution
-----------

If you use PynPoint in your publication then please cite Stolker et al. subm. Please also cite `Amara & Quanz (2012) <http://adsabs.harvard.edu/abs/2012MNRAS.427..948A>`_ as the origin of PynPoint, which focused initially on the use of principal component analysis (PCA) as a PSF subtraction method. In case you use specifically the PCA-based background subtraction module or the wavelet based speckle suppression module, please give credit to `Hunziker et al. (2018) <http://adsabs.harvard.edu/abs/2018A%26A...611A..23H>`_ or `Bonse, Quanz & Amara (2018) <http://adsabs.harvard.edu/abs/2018arXiv180405063B>`_, respectively.

License
-------

Copyright 2014-2018 Tomas Stolker, Markus Bonse, Sascha Quanz, Adam Amara, and contributors.

PynPoint is free software and distributed under the GNU General Public License v3. See the LICENSE file for the terms and conditions.

Acknowledgements
----------------

The PynPoint logo was designed by `Atlas Infographics <https://atlas-infographics.nl>`_.
