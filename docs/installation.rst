.. _installation:

Installation
============

Virtual Environment
-------------------

PynPoint is available in the |pypi| and on |github|. We recommend using a Python virtual environment to install and run PynPoint such that the correct versions of the dependencies can be installed without affecting other installed Python packages. First install `virtualenv`, for example with the |pip|::

    $ pip install virtualenv

Then create a virtual environment::

    $ virtualenv folder_name

And activate the environment with::

    $ source folder_name/bin/activate

A virtual environment can be deactivated with::

    $ deactivate

.. important::
   Make sure to adjust the path where the virtual environment is installed and activated.

Installation from PyPI
----------------------

PynPoint can now be installed with pip::

    $ pip install pynpoint

Installation from Github
------------------------

The repository can also be cloned from Github, which contains the most recent implementations::

    $ git clone git@github.com:PynPoint/PynPoint.git

In that case, the dependencies can be installed from the PynPoint folder::

    $ pip install -r requirements.txt

By adding the path of the repository to the ``PYTHONPATH`` environment variable enables PynPoint to be imported from any location::

    $ echo "export PYTHONPATH='$PYTHONPATH:/path/to/pynpoint'" >> folder_name/bin/activate

.. important::
   Make sure to adjust local path in which PynPoint will be cloned from the Github repository.

Do you want to makes changes to the code? Then please fork the PynPoint repository on the Github page and clone your own fork instead of the main repository. We very much welcome active contributions and pull requests (see :ref:`contributing` section).

Testing Pynpoint
----------------

The installation can be tested by starting Python in interactive mode and printing the PynPoint version::

    >>> import pynpoint
    >>> pynpoint.__version__

.. tip::
   If the PynPoint package is not find by Python then possibly the path was not set correctly. The list of folders that are searched by Python for modules can be printed in interactive mode as::

      >>> import sys
      >>> sys.path

   The result should contain the folder in which the Github repository was cloned or the folder in which Python modules are installed with pip.

.. |pypi| raw:: html

   <a href="https://pypi.org/project/pynpoint/" target="_blank">PyPI repository</a>

.. |github| raw:: html

   <a href="https://github.com/PynPoint/PynPoint" target="_blank">Github</a>

.. |pip| raw:: html

   <a href="https://packaging.python.org/tutorials/installing-packages/" target="_blank">pip package manager</a>
