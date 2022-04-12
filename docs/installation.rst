.. _installation:

Installation
============

PynPoint is compatible with `Python <https://www.python.org>`_ versions 3.8/3.9/3.10.

.. _virtual_environment:

Virtual Environment
-------------------

PynPoint is available in the `PyPI repository <https://pypi.org/project/pynpoint/>`_ and on `Github <https://github.com/PynPoint/PynPoint>`_. We recommend using a Python virtual environment to install and run PynPoint such that the correct dependency versions are installed without affecting other Python installations. First install ``virtualenv``, for example with the `pip package manager <https://packaging.python.org/tutorials/installing-packages/>`_:

.. code-block:: console

    $ pip install virtualenv

Then create a virtual environment:

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

   $ pip install git+git://github.com/PynPoint/PynPoint.git

Cloning the repository
^^^^^^^^^^^^^^^^^^^^^^

Alternatively, the Github repository can be cloned, which is in particular useful if you want to look into the code:

.. code-block:: console

    $ git clone git@github.com:PynPoint/PynPoint.git

The package is installed by running ``pip`` in the local repository folder:

.. code-block:: console

    $ pip install -e .

Instead of running ``setup.py``, the path of the repository can also be added to the ``PYTHONPATH`` environment variable such that PynPoint can be imported from any working folder. When using a ``virtualenv``, the ``PYTHONPATH`` can be added to the activation script:

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
