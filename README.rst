PynPoint
========

**Pipeline for processing and analysis of high-contrast imaging data**

.. image:: https://badge.fury.io/py/pynpoint.svg
    :target: https://pypi.python.org/pypi/pynpoint

.. image:: https://img.shields.io/badge/Python-3.6%2C%203.7%2C%203.8-yellow.svg?style=flat
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

PynPoint is a generic, end-to-end pipeline for the data reduction and analysis of high-contrast imaging data of exoplanets. The pipeline uses principal component analysis (PCA) for the subtraction of the stellar PSF and supports post-processing with ADI, RDI, and SDI techniques. The package is stable, extensively tested, and actively maintained.

For a first impression, `this example <https://home.strw.leidenuniv.nl/~stolker/pynpoint/hd142527_zimpol_h-alpha.tgz>`_ of `HD 142527B <https://ui.adsabs.harvard.edu/abs/2019A%26A...622A.156C/abstract>`_ shows a typical workflow.

Documentation
-------------

Documentation is available at `http://pynpoint.readthedocs.io <http://pynpoint.readthedocs.io>`_, including installation instructions, details on the pipeline architecture, and a description of all the pipeline modules and input parameters.

Mailing list
------------

Please subscribe to the `mailing list <https://pynpoint.readthedocs.io/en/latest/mailing.html>`_ if you want to be informed about PynPoint related news.

Attribution
-----------

If you use PynPoint in your publication then please cite `Stolker et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019A%26A...621A..59S/abstract>`_. Please also cite `Amara & Quanz (2012) <https://ui.adsabs.harvard.edu/abs/2012MNRAS.427..948A/abstract>`_ as the origin of PynPoint, which focused initially on the use of PCA as a PSF subtraction method. In case you use specifically the PCA-based background subtraction module or the wavelet based speckle suppression module, please give credit to `Hunziker et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018A%26A...611A..23H/abstract>`_ or `Bonse et al. (preprint) <https://ui.adsabs.harvard.edu/abs/2018arXiv180405063B/abstract>`_, respectively.

Contributing
------------

Contributions in the form of bug fixes, new or improved functionalities, and pipeline modules are highly appreciated. Please consider forking the repository and creating a pull request to help improve and extend the package. Instructions for `coding of a pipeline module <https://pynpoint.readthedocs.io/en/latest/coding.html>`_ are available in the documentation. Bugs can be reported by creating an `issue <https://github.com/PynPoint/PynPoint/issues>`_ on the Github page.

License
-------

Copyright 2014-2021 Tomas Stolker, Markus Bonse, Sascha Quanz, Adam Amara, and contributors.

PynPoint is distributed under the GNU General Public License v3. See the LICENSE file for the terms and conditions.

Acknowledgements
----------------

The PynPoint logo was designed by `Atlas Interactive <https://atlas-interactive.nl>`_ and is `available <https://quanz-group.ethz.ch/research/algorithms/pynpoint.html>`_ for use in presentations.
