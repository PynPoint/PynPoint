.. _tutorial:

Tutorial
========

.. _introduction:

Introduction
------------

The pipeline can be executed with a Python script, in interactive mode of Python, or with a Jupyter Notebook. The pipeline works with two different components:

1. Pipeline modules which read, write, and process data:

	1.1 :class:`pynpoint.core.processing.ReadingModule` - Reading of the data and relevant header information.

	1.2 :class:`pynpoint.core.processing.WritingModule` - Exporting of results from the database.

	1.3 :class:`pynpoint.core.processing.ProcessingModule` - Processing and analysis of the data.

2. The actual pipeline :class:`pynpoint.core.pypeline.Pypeline` which capsules a list of pipeline modules.

.. important::
   Pixel coordinates are zero-indexed, meaning that (x, y) = (0, 0) corresponds to the center of the pixel in the bottom-left corner of the image. The coordinate of the bottom-left corner is therefore (x, y) = (-0.5, -0.5). This means that there is an offset of -1.0 in both directions with respect to the pixel coordinates of DS9, for which the bottom-left corner is (x, y) = (0.5, 0.5).

.. _data-types:

Data Types
----------

PynPoint currently works with three types of input and output data:

* FITS files
* HDF5 files
* ASCII files

PynPoint creates an HDF5 database called ``PynPoin_database.hdf5`` in the ``working_place_in`` of the pipeline. This is the central data storage in which the results of the processing steps are saved. The advantage of the HDF5 data format is that reading of data is much faster compared to the FITS data format and it is possible to quickly read subsets from very large datasets.

Input data is read into the central database with a :class:`~pynpoint.core.processing.ReadingModule`. By default, PynPoint will read data from the ``input_place_in`` but setting a manual folder is possible to read data to separate database tags (e.g., dark frames, flat fields, and science data). Here we show an example of how to read FITS files and a list of parallactic angles.

First, we need to create an instance of :class:`~pynpoint.core.pypeline.Pypeline`::

    pipeline = Pypeline(working_place_in="/path/to/working_place",
                        input_place_in="/path/to/input_place",
                        output_place_in="/path/to/output_place")

Next, we read the science data from the the default input location::

    module = FitsReadingModule(name_in="read_science",
                               input_dir=None,
                               image_tag="science")

    pipeline.add_module(module)

And we read the flat fields from a separate location::

    module = FitsReadingModule(name_in="read_flat",
                               input_dir="/path/to/flat",
                               image_tag="flat")

    pipeline.add_module(module)

The parallactic angles are read from a text file in the default input folder and attached as attribute to the science data::

    module = ParangReadingModule(file_name="parang.dat",
                                 name_in="parang",
                                 input_dir=None,
                                 data_tag="science")

    pipeline.add_module(module)

Finally, we run all pipeline modules::

    pipeline.run()

Alternatively, it is also possible to run the modules individually by their ``name_in`` value::

    pipeline.run_module("read_science")
    pipeline.run_module("read_flat")
    pipeline.run_module("parang")

The FITS files of the science data and flat fields are read and stored into the central HDF5 database. The data is labelled with a tag which is used by other pipeline module to access data from the database.

Restoring data from an already existing pipeline database can be done by creating an instance of :class:`~pynpoint.core.pypeline.Pypeline` with the ``working_place_in`` pointing to the path of the ``PynPoint_database.hdf5`` file.

PynPoint can also handle the HDF5 format as input and output data. Data and corresponding attributes can be exported as HDF5 file with  :class:`~pynpoint.readwrite.hdf5writing.Hdf5WritingModule`. This data format can be imported into the central database with :class:`~pynpoint.readwrite.hdf5reading.Hdf5ReadingModule`. Have a look at the :ref:`pynpoint-package` section for more information.

.. _hdf5-files:

HDF5 Files
----------

There are several options to access data from the central HDF5 database:

	* Use :class:`~pynpoint.readwrite.fitswriting.FitsWritingModule` to export data to a FITS file, as shown in the :ref:`examples` section.
	* Use the easy access functions of the :class:`pynpoint.core.pypeline` module to retrieve data and attributes from the database:

		* ``pipeline.get_data(tag='tag_name')``

		* ``pipeline.get_attribute(data_tag='tag_name', attr_name='attr_name')``

	* Use an external tool such as |HDFCompass| or |HDFView| to read, inspect, and visualize data and attributes in the HDF5 database. We recommend using HDFCompass because it is easy to use and has a basic plotting functionality, allowing the user to quickly inspect images from a particular database tag. In HDFCompass, the static attributes can be opened with the `Reopen as HDF5 Attributes` option.

.. |HDFCompass| raw:: html

   <a href="https://support.hdfgroup.org/projects/compass/download.html" target="_blank">HDFCompass</a>

.. |HDFView| raw:: html

   <a href="https://support.hdfgroup.org/downloads/index.html" target="_blank">HDFView</a>
