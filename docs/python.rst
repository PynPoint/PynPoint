.. _python:

Python Guidelines
=================

.. _starting:

Getting Started
---------------

The modular architecture of PynPoint allows for easy implementation of new pipeline modules and we welcome contributions from users. Before writing a new PynPoint module, it is helpful to have a look at the :ref:`architecture` section. In addition, some basic knowledge on Python is required and some understanding on the following items can be helpful:

    * Python |types| such as lists, tuples, and dictionaries.
    * |classes|, in particular the concept of inheritance.
    * |abc| as interfaces.

.. |types| raw:: html

   <a href="https://docs.python.org/3/library/stdtypes.html" target="_blank">types</a>

.. |classes| raw:: html

   <a href="https://docs.python.org/3/tutorial/classes.html" target="_blank">Classes</a>

.. |abc| raw:: html

   <a href="https://docs.python.org/2/library/abc.html" target="_blank">Abstract classes</a>

There are three different types of pipeline modules: :class:`~pynpoint.core.processing.ReadingModule`, :class:`~pynpoint.core.processing.WritingModule`, and :class:`~pynpoint.core.processing.ProcessingModule`. The concept is similar for the three types of modules so here we will explain only how to code a processing module.

.. _conventions:

Conventions
-----------

Before we start writing a new PynPoint module, please take notice of the following style conventions:

    * |pep8| -- style guide for Python code
    * We recommend using |pylint| to analyze newly written code in order to keep PynPoint well structured, readable, and documented.
    * Names of class member should start with ``m_``.
    * Images should ideally not be read from and written to the central database at once but in amounts of ``MEMORY``.

.. |pep8| raw:: html

   <a href="https://www.python.org/dev/peps/pep-0008" target="_blank">PEP 8</a>

.. |pylint| raw:: html

   <a href="https://www.pylint.org" target="_blank">pylint</a>

Now we are ready to code!