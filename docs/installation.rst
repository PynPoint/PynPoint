.. _installation:

Installation
============

PynPoint is compatible with Python 3.7/3.8/3.9. Earlier versions (up to v0.7.0) are also compatible with Python 2.7.

.. _virtual_environment:

Virtual Environment
-------------------

PynPoint is available in the `PyPI repository <https://pypi.org/project/pynpoint/>`_ and on `Github <https://github.com/PynPoint/PynPoint>`_. We recommend using a Python virtual environment to install and run PynPoint such that the correct versions of the dependencies can be installed without affecting other installed Python packages. First install ``virtualenv``, for example with the `pip package manager <https://packaging.python.org/tutorials/installing-packages/>`_:

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

.. _installation_pypi:

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

Using pip
^^^^^^^^^

The Github repository contains the latest commits. Installation from Github is also possible with ``pip``:

.. code-block:: console

   $ pip install git+git://github.com/PynPoint/PynPoint.git@main

This will also install the required dependencies.

Cloning the repository
^^^^^^^^^^^^^^^^^^^^^^

Alternatively, the Github repository can be cloned, which is in particular useful if you want to look into and/or make changes to the code:

.. code-block:: console

    $ git clone git@github.com:PynPoint/PynPoint.git

PynPoint and the dependencies can be installed by running the setup script:

.. code-block:: console

    $ python setup.py install

Instead of running ``setup.py``, the path of the repository can also be added to the ``PYTHONPATH`` environment variable such that PynPoint can be imported from any working folder. When using a ``virtualenv``, the ``PYTHONPATH`` can be added to the activation script:

.. code-block:: console

    $ echo "export PYTHONPATH='$PYTHONPATH:/path/to/pynpoint'" >> folder_name/bin/activate

With this last approach, the dependencies need to be installed manually.

.. important::
   Make sure to adjust the path to the PynPoint folder and the virtual environment.

Once a local copy of the repository exists, new commits can be pulled from Github with:

.. code-block:: console

    $ git pull origin main

Do you want to makes changes to the code? Please fork the PynPoint repository on the Github page and clone your own fork instead of the main repository. We very much welcome contributions and pull requests (see :ref:`contributing` section).

Dependencies
^^^^^^^^^^^^

If needed, the dependencies can be manually installed from the PynPoint folder:

.. code-block:: console

    $ pip install -r requirements.txt

Or updated to the latest versions with which PynPoint is compatible:

.. code-block:: console

    $ pip install --upgrade -r requirements.txt

.. _testing_pynpoint:

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
