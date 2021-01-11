.. _running_pynpoint:

Running PynPoint
================

.. _running_intro:

Introduction
------------

The pipeline can be executed with a Python script, in `interactive mode <https://docs.python.org/3/tutorial/interpreter.html#interactive-mode>`_, or with a `Jupyter Notebook <https://jupyter.org/>`_. The main components of PynPoint are the pipeline and the three types of pipeline modules:

1. :class:`~pynpoint.core.pypeline.Pypeline` -- The actual pipeline which capsules a list of pipeline modules.

2. :class:`~pynpoint.core.processing.ReadingModule` -- Module for importing data and relevant header information from FITS, HDF5, or ASCII files into the database.

3. :class:`~pynpoint.core.processing.WritingModule` -- Module for exporting results from the database into FITS, HDF5 or ASCII files.

4. :class:`~pynpoint.core.processing.ProcessingModule` -- Module for processing data with a specific data reduction or analysis recipe.

.. _initiating_pypeline:

Initiating the Pypeline
-----------------------

The pipeline is initiated by creating an instance of :class:`~pynpoint.core.pypeline.Pypeline`:

.. code-block:: python

    pipeline = Pypeline(working_place_in='/path/to/working_place',
                        input_place_in='/path/to/input_place',
                        output_place_in='/path/to/output_place')

PynPoint creates an HDF5 database called ``PynPoin_database.hdf5`` in the ``working_place_in`` of the pipeline. This is the central data storage in which the processing results from a :class:`~pynpoint.core.processing.ProcessingModule` are stored. The advantage of the HDF5 format is that reading of data is much faster than from FITS files and it is also possible to quickly read subsets from large datasets.

Restoring data from an already existing pipeline database can be done by creating an instance of :class:`~pynpoint.core.pypeline.Pypeline` with the ``working_place_in`` pointing to the path of the ``PynPoint_database.hdf5`` file.

.. _running_modules:

Running pipeline modules
------------------------

Input data is read into the central database with a :class:`~pynpoint.core.processing.ReadingModule`. By default, PynPoint will read data from the ``input_place_in`` but setting a manual folder is possible to read data to separate database tags (e.g., dark frames, flat fields, and science data).

For example, to read the images from FITS files that are located in the default input place:

.. code-block:: python

    module = FitsReadingModule(name_in='read',
                               input_dir=None,
                               image_tag='science')

    pipeline.add_module(module)

The images from the FITS files are stored in the database as a dataset with a unique tag. This tag can be used by other pipeline module to read the data for further processing.

The parallactic angles can be read from a text or FITS file and are attached as attribute to a dataset:

.. code-block:: python

    module = ParangReadingModule(name_in='parang',
                                 data_tag='science'
                                 file_name='parang.dat',
                                 input_dir=None)

    pipeline.add_module(module)

Finally, we run all pipeline modules:

.. code-block:: python

    pipeline.run()

Alternatively, it is also possible to run each pipeline module individually by their ``name_in`` value:

.. code-block:: python

    pipeline.run_module('read')
    pipeline.run_module('parang')

.. important::
   Some pipeline modules require pixel coordinates for certain arguments. Throughout PynPoint, pixel coordinates are zero-indexed, meaning that (x, y) = (0, 0) corresponds to the center of the pixel in the bottom-left corner of the image. This means that there is an offset of -1 in both directions with respect to the pixel coordinates of DS9, for which the center of the bottom-left pixel is (x, y) = (1, 1).

.. _hdf5_files:

HDF5 database
-------------

There are several ways to access the datasets in the HDF5 database that is used by PynPoint:

* The :class:`~pynpoint.readwrite.fitswriting.FitsWritingModule` exports a dataset from the database into a FITS file.

* Several methods of the :class:`~pynpoint.core.pypeline.Pypeline` class help to easily retrieve data and attributes from the database. For example:

   * To read a dataset:

     .. code-block:: python

        pipeline.get_data('tag_name')

   * To read an attribute of a dataset:

     .. code-block:: python

        pipeline.get_attribute('tag_name', 'attr_name')

* The `h5py <http://www.h5py.org/>`_ Python package can be used to access the HDF5 file directly.

* There are external tools available such as `HDFCompass <https://support.hdfgroup.org/projects/compass/download.html>`_ or `HDFView <https://support.hdfgroup.org/downloads/index.html>`_ to read, inspect, and visualize data and attributes. HDFCompass is easy to use and has a basic plotting functionality. In HDFCompass, the static PynPoint attributes can be opened with the *Reopen as HDF5 Attributes* option.
