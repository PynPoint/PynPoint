PynPoint
========

**Pipeline for processing and analysis of high-contrast imaging data**

.. image:: https://badge.fury.io/py/pynpoint.svg
    :target: https://pypi.python.org/pypi/pynpoint

.. image:: https://img.shields.io/badge/Python-3.6%2C%203.7-yellow.svg?style=flat
    :target: https://pypi.python.org/pypi/pynpoint

.. image:: https://travis-ci.org/PynPoint/PynPoint.svg?branch=master
    :target: https://travis-ci.org/PynPoint/PynPoint

.. image:: https://readthedocs.org/projects/pynpoint/badge/?version=latest
    :target: http://pynpoint.readthedocs.io/en/latest/?badge=latest

.. image:: https://coveralls.io/repos/github/PynPoint/PynPoint/badge.svg?branch=master
    :target: https://coveralls.io/github/PynPoint/PynPoint?branch=master

.. image:: https://www.codefactor.io/repository/github/pynpoint/pynpoint/badge
    :target: https://www.codefactor.io/repository/github/pynpoint/pynpoint

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://github.com/PynPoint/PynPoint/blob/master/LICENSE

.. image:: http://img.shields.io/badge/arXiv-1811.03336-orange.svg?style=flat
    :target: http://arxiv.org/abs/1811.03336

PynPoint is a generic, end-to-end pipeline for the data reduction and analysis of high-contrast imaging data of planetary and substellar companions, as well as circumstellar disks in scattered light. The package is stable, has been extensively tested, and is available on `PyPI <https://pypi.org/project/pynpoint/>`_. PynPoint is under continuous development so the latest implementations can be pulled from Github repository.

The pipeline has a modular architecture with a central data storage in which all results are stored by the processing modules. These modules have specific tasks such as the subtraction of the thermal background emission, frame selection, centering, PSF subtraction, and photometric and astrometric measurements. The tags from the central data storage can be written to FITS, HDF5, and text files with the available I/O modules.

To get a first impression, there is an end-to-end example available of a `SPHERE/ZIMPOL <https://www.eso.org/sci/facilities/paranal/instruments/sphere.html>`_ H-alpha data set of the accreting M dwarf companion of `HD 142527 <http://ui.adsabs.harvard.edu/abs/2019A%26A...622A.156C>`_, which can be downloaded `here <https://people.phys.ethz.ch/~stolkert/pynpoint/hd142527_zimpol_h-alpha.tgz>`_.

Documentation
-------------

Documentation can be found at `http://pynpoint.readthedocs.io <http://pynpoint.readthedocs.io>`_, including installation instructions, details on the architecture of PynPoint, and a description of all the pipeline modules and their input parameters.

Mailing list
------------

Please subscribe to the `mailing list <https://pynpoint.readthedocs.io/en/latest/mailing.html>`_ if you want to be informed about new functionalities, pipeline modules, releases, and other PynPoint related news.

Attribution
-----------

If you use PynPoint in your publication then please cite `Stolker et al. (2019) <http://ui.adsabs.harvard.edu/abs/2019A%26A...621A..59S>`_. Please also cite `Amara & Quanz (2012) <http://ui.adsabs.harvard.edu/abs/2012MNRAS.427..948A>`_ as the origin of PynPoint, which focused initially on the use of principal component analysis (PCA) as a PSF subtraction method. In case you use specifically the PCA-based background subtraction module or the wavelet based speckle suppression module, please give credit to `Hunziker et al. (2018) <http://ui.adsabs.harvard.edu/abs/2018A%26A...611A..23H>`_ or `Bonse, Quanz & Amara (2018) <http://ui.adsabs.harvard.edu/abs/2018arXiv180405063B>`_, respectively.

Contributing
------------

Contributions in the form of bug fixes, new or improved functionalities, and additional pipeline modules are highly appreciated. Please consider forking the repository and creating a pull request to help improve and extend the package. Instructions for writing of modules are provided in the documentation. Bug reports can be provided by creating an `issue <https://github.com/PynPoint/PynPoint/issues>`_ on the Github page.

License
-------

Copyright 2014-2020 Tomas Stolker, Markus Bonse, Sascha Quanz, Adam Amara, and contributors.

PynPoint is distributed under the GNU General Public License v3. See the LICENSE file for the terms and conditions.

Acknowledgements
----------------

The PynPoint logo was designed by `Atlas Interactive <https://atlas-interactive.nl>`_ and is `available <https://quanz-group.ethz.ch/research/algorithms/pynpoint.html>`_ for use in presentations.
