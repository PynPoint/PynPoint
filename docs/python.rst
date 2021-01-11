.. _python:

Python guidelines
=================

.. _starting:

Getting started
---------------

The modular architecture of PynPoint allows for easy implementation of new pipeline modules and we welcome contributions from users. Before writing a new PynPoint module, it is helpful to have a look at the :ref:`architecture` section. In addition, some basic knowledge on Python is required and some understanding on the following items can be helpful:

    * Python `types <https://docs.python.org/3/library/stdtypes.html>`_ such as lists, tuples, and dictionaries.
    * `Classes <https://docs.python.org/3/tutorial/classes.html>`_ and in particular the concept of inheritance.
    * `Absrtact classes <https://docs.python.org/3/library/abc.html>`_ as interfaces.

.. _conventions:

Conventions
-----------

Before we start writing a new PynPoint module, please take notice of the following style conventions:

    * `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ -- style guide for Python code
    * We recommend using `pylint <https://www.pylint.org>`_ and `pycodestyle <https://pypi.org/project/pycodestyle/>`_ to analyze newly written code in order to keep PynPoint well structured, readable, and documented.
    * Names of class member should start with ``m_``.
    * Images should ideally not be read from and written to the central database at once but in amounts of ``MEMORY``.

Unit tests
----------

PynPoint is a robust pipeline package with 95% of the code covered by `unit tests <https://docs.python.org/3/library/unittest.html>`_. Testing of the package is done by running ``make test`` in the cloned repository. This requires the installation of:

   * `pytest <https://docs.pytest.org/en/latest/getting-started.html>`_
   * `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/readme.html>`_

The unit tests ensure that the output from existing functionalities will not change whenever new code. With these things in mind, we are now ready to code!
