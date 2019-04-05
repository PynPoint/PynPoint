.. _near_intro:

Introduction
============

This guide shows the basic steps to process NEAR data with PynPoint. First we describe the steps required to install PynPoint.  After that, the basic steps how to get PCA working with PynPoint are shown. Python version 3 is used.

.. _near_install:

Install PynPoint
----------------

PynPoint is available in the |pypi| and on |github|. We recommend using a Python virtual environment to install and run PynPoint such that the correct versions of the dependencies can be installed without affecting other installed Python packages. First install `virtualenv` with the |pip|::

    $ python3 -m pip install virtualenv

Then create a virtual environment, for example::

    $ virtualenv folder_name

And activate the environment with::

    $ source folder_name/bin/activate

PynPoint can now be installed with pip::

    $ python3 -m pip install pynpoint

If you do not use a virtual environment, you might need to add the '`- - user`' argument::

    $ python3 -m pip install --user pynpoint

The installation can be tested by starting Python in interactive mode and printing the PynPoint version::

    >>> import pynpoint
    >>> pynpoint.__version__

A virtual environment is deactivate with::

    $ deactivate

.. |pypi| raw:: html

   <a href="https://pypi.org/project/pynpoint/" target="_blank">PyPI repository</a>

.. |github| raw:: html

   <a href="https://github.com/PynPoint/PynPoint" target="_blank">Github</a>

.. |pip| raw:: html

   <a href="https://packaging.python.org/tutorials/installing-packages/" target="_blank">pip package manager</a>

.. _near_attribution:

Attribution
-----------

If you use PynPoint in your publication then please cite `Stolker et al. (2019) <http://adsabs.harvard.edu/abs/2019A%26A...621A..59S>`_. Please also cite `Amara & Quanz (2012) <http://adsabs.harvard.edu/abs/2012MNRAS.427..948A>`_ as the origin of PynPoint, which focused initially on the use of principal component analysis (PCA) as a PSF subtraction method. In case you use specifically the PCA-based background subtraction module or the wavelet based speckle suppression module, please give credit to `Hunziker et al. (2018) <http://adsabs.harvard.edu/abs/2018A%26A...611A..23H>`_ or `Bonse, Quanz & Amara (2018) <http://adsabs.harvard.edu/abs/2018arXiv180405063B>`_, respectively.
