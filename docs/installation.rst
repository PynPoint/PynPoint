.. _installation:

Installation
============

PynPoint is compatible with Python 3.6/3.7/3.8. Earlier versions (up to v0.7.0) are also compatible with Python 2.7, but we highly recommend using Python 3 since several key Python projects have already |python| Python 2.

Virtual Environment
-------------------

PynPoint is available in the |pypi| and on |github|. We recommend using a Python virtual environment to install and run PynPoint such that the correct versions of the dependencies can be installed without affecting other installed Python packages. First install `virtualenv`, for example with the |pip|:

.. code-block:: console

    $ pip install virtualenv

Then create a virtual environment for Python 3:

.. code-block:: console

    $ virtualenv -p python3 folder_name

And activate the environment with:

.. code-block:: console

    $ source folder_name/bin/activate

A virtual environment can be deactivated with:

.. code-block:: console

    $ deactivate

.. important::
   Make sure to adjust the path where the virtual environment is installed and activated.

Installation from PyPI
----------------------

PynPoint can now be installed with pip:

.. code-block:: console

    $ pip install pynpoint

If you do not use a virtual environment then you may have to add the ``--user`` argument:

.. code-block:: console

    $ pip install --user pynpoint

To update the installation to the most recent version:

.. code-block:: console

   $ pip install --upgrade PynPoint

.. _installation_github:

Installation from Github
------------------------

The repository can be cloned from Github, which contains the most recent implementations:

.. code-block:: console

    $ git clone git@github.com:PynPoint/PynPoint.git

In that case, the dependencies can be installed from the PynPoint folder:

.. code-block:: console

    $ pip install -r requirements.txt

And to update the dependencies to the latest versions with which PynPoint is compatible:

.. code-block:: console

    $ pip install --upgrade -r requirements.txt 

Once a local copy of the repository exists, new commits can be pulled from Github with:

.. code-block:: console

    $ git pull origin master

By adding the path of the repository to the ``PYTHONPATH`` environment variable enables PynPoint to be imported from any location:

.. code-block:: console

    $ echo "export PYTHONPATH='$PYTHONPATH:/path/to/pynpoint'" >> folder_name/bin/activate

.. important::
   Make sure to adjust local path in which PynPoint will be cloned from the Github repository.

Do you want to makes changes to the code? Then please fork the PynPoint repository on the Github page and clone your own fork instead of the main repository. We very much welcome contributions and pull requests (see :ref:`contributing` section).

Testing Pynpoint
----------------

The installation can be tested by starting Python in interactive mode and printing the PynPoint version:

.. code-block:: python

    >>> import pynpoint
    >>> pynpoint.__version__

.. tip::
   If the PynPoint package is not find by Python then possibly the path was not set correctly. The list of folders that are searched by Python for modules can be printed in interactive mode as:

      .. code-block:: python

         >>> import sys
         >>> sys.path

   The result should contain the folder in which the Github repository was cloned or the folder in which Python modules are installed with pip.

.. |python| raw:: html

   <a href="https://python3statement.org/" target="_blank">stopped supporting</a>

.. |pypi| raw:: html

   <a href="https://pypi.org/project/pynpoint/" target="_blank">PyPI repository</a>

.. |github| raw:: html

   <a href="https://github.com/PynPoint/PynPoint" target="_blank">Github</a>

.. |pip| raw:: html

   <a href="https://packaging.python.org/tutorials/installing-packages/" target="_blank">pip package manager</a>
